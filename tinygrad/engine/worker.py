import multiprocessing, atexit, signal, sys, threading, contextlib
from multiprocessing.context import SpawnContext, SpawnProcess
from multiprocessing.reduction import ForkingPickler
from tinygrad.helpers import Context, getenv, PARALLEL
from tinygrad.uop.ops import UOp, Ops

# LOCAL PATCH (2026-09-01): UOp graphs pickle recursively (UOp.__reduce__ -> src -> __reduce__ ...) and the C pickler
# has a fixed recursion budget (~5000 frames on python 3.12, NOT raised by sys.setrecursionlimit). The ASTs of large
# kernels (measured: 1024px diffusion matmuls) exceed it, which killed the pool's task feeder with RecursionError and
# every worker with a broken pipe. Graphs cross the process boundary as a flat toposorted table instead; rebuilding goes
# through UOp's hash-consing, so identity/dedup is preserved on the other side.
def _rebuild_uop(rows:list) -> UOp:
  built:list[UOp] = []
  for op, srcs, arg, tag, metadata, *rest in rows: built.append(UOp(op, tuple(built[i] for i in srcs), arg, tag, metadata, *rest))
  return built[-1]
def _reduce_uop(u:UOp):
  order = list(u.toposort())
  idx = {x:i for i,x in enumerate(order)}
  rows = []
  for x in order:
    row:list = [x.op, tuple(idx[s] for s in x.src), x.arg, x.tag, x.metadata]
    if x.op is Ops.BUFFER and x.realized is not None: row.append(x.realized)
    rows.append(tuple(row))
  return _rebuild_uop, (rows,)
ForkingPickler.register(UOp, _reduce_uop)

# generic pool of worker processes for parallel compilation, shared by kernel lowering and BEAM search

# workers should not open devices and should ignore ctrl c and should not launch VIZ
def _init_worker():
  Context(ALLOW_DEVICE_USAGE=0, VIZ=0, TRACK_MATCH_STATS=0).__enter__()
  signal.signal(signal.SIGINT, signal.SIG_IGN)

# spawn normally reimports the user's __main__ before _init_worker. This replays top-level code and can recursively create pools. There is no public
# multiprocessing switch to skip that import, so hide the two attributes used to locate __main__ while each worker (including replacements) starts.
_spawn_lock, _missing = threading.Lock(), object()

# LOCAL PATCH (2026-09-02): hiding __main__ by setting __file__ = None makes every concurrent inspect.getmodule()
# in the parent raise "TypeError: <module '__main__' from None> is a built-in module" - the scheduler's DEBUG>=1
# caller-location line does exactly that, and a pool worker respawn (maxtasksperchild) from the pool's maintenance
# thread happens while the main thread is scheduling. Point __file__ at an empty stub instead: the spawned child
# imports the stub as __mp_main__ (harmless) and inspect keeps working.
_stub_path = None
def _stub_main_file() -> str:
  global _stub_path
  if _stub_path is None:
    import tempfile, os
    _stub_path = os.path.join(tempfile.gettempdir(), "tinygrad_mp_main_stub.py")
    if not os.path.exists(_stub_path):
      with open(_stub_path, "w") as f: f.write("# empty stand-in for __main__ while tinygrad spawns pool workers\n")
  return _stub_path

@contextlib.contextmanager
def _without_main():
  main = sys.modules.get("__main__")
  if main is None:
    yield
    return
  with _spawn_lock:
    saved = {name:getattr(main, name, _missing) for name in ("__file__", "__spec__")}
    try:
      setattr(main, "__file__", _stub_main_file())
      setattr(main, "__spec__", None)
      yield
    finally:
      for name,value in saved.items(): delattr(main, name) if value is _missing else setattr(main, name, value)

class _WorkerProcess(SpawnProcess):
  @staticmethod
  def _Popen(process_obj):
    with _without_main(): return SpawnProcess._Popen(process_obj)

class _WorkerContext(SpawnContext): Process = _WorkerProcess

worker_pool = None
def get_worker_pool():
  global worker_pool
  if multiprocessing.current_process().daemon or PARALLEL == 0: return None
  if worker_pool is None:
    worker_pool = _WorkerContext().Pool(PARALLEL.value, _init_worker, (), getenv("BEAM_MAX_TASKS_PER_CHILD", 16))
    @atexit.register
    def close_pool(pool=worker_pool): pool.close()
  return worker_pool

def terminate_worker_pool():
  global worker_pool
  if worker_pool is not None: worker_pool.terminate()
  worker_pool = None

# LOCAL PATCH (2026-09-01): a worker that dies mid-task never delivers its result and pool.imap_unordered then waits
# forever (measured: one worker hit BrokenPipeError on the result queue during a 1024px compile batch and the ComfyUI
# prompt hung until its 3600 s budget). Bound the wait per result; the caller lowers whatever is missing itself. A pool
# that produced a timeout is torn down so the next get_worker_pool() starts a fresh one.
POOL_TASK_TIMEOUT = getenv("POOL_TASK_TIMEOUT", 1200)
def pool_imap_bounded(pool, fn, tasks:list):
  it = pool.imap_unordered(fn, tasks)
  got = 0
  while got < len(tasks):
    try: r = it.next(timeout=POOL_TASK_TIMEOUT)
    except multiprocessing.TimeoutError:
      print(f"worker pool: no result for {POOL_TASK_TIMEOUT}s with {len(tasks)-got} task(s) outstanding - assuming a dead worker, "
            "finishing in the parent and restarting the pool", flush=True)
      terminate_worker_pool()
      return
    except StopIteration: return
    got += 1
    yield r

import math, time, traceback, signal, threading
from dataclasses import replace
from tinygrad.uop.ops import sym_infer, AxisType, UOp, Ops
from tinygrad.uop.render import pyrender
from tinygrad.device import Device, Buffer
from tinygrad.helpers import prod, flatten, DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, colored, time_to_str
from tinygrad.helpers import IGNORE_BEAM_CACHE
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.engine.realize import time_call
from tinygrad.engine.worker import get_worker_pool, terminate_worker_pool, pool_imap_bounded
from tinygrad.codegen import to_program, finalize_program
from tinygrad.codegen.opt.postrange import Scheduler

actions = [Opt(op=OptOps.UPCAST, axis=axis, arg=amt) for amt in [0,2,3,4,5,7] for axis in range(8)]
actions += [Opt(op=OptOps.UNROLL, axis=axis, arg=amt) for amt in [0,4,7] for axis in range(5)]
actions += [Opt(op=OptOps.LOCAL, axis=axis, arg=amt) for amt in [2,3,4,8,13,16,29] for axis in range(6)]
actions += [Opt(op=OptOps.GROUPTOP, axis=axis, arg=amt) for amt in [13,16,28,29,32,49,64,256] for axis in range(3)]
actions += [Opt(op=OptOps.GROUP, axis=axis, arg=amt) for amt in [0,4,8,16] for axis in range(3)]
if getenv("BEAM_PADTO", 0): actions += [Opt(op=OptOps.PADTO, axis=axis, arg=amt) for amt in [32] for axis in range(7)]
actions += [Opt(op=OptOps.LOCAL, axis=0, arg=32), Opt(op=OptOps.LOCAL, axis=6, arg=2)]
actions += [Opt(op=OptOps.TC, axis=0, arg=(-1, 0, getenv("TC", 1)))]
# covers resnet kernels (3 global * 3 reduce)
actions += [Opt(op=OptOps.TC, axis=axis, arg=(-1, getenv("TC_OPT", 2), getenv("TC", 1))) for axis in range(9)]
actions += [Opt(op=OptOps.SWAP, axis=axis_0, arg=axis_1) for axis_0 in range(5) for axis_1 in range(axis_0+1, 5)]
actions += [Opt(op=OptOps.THREAD, axis=axis, arg=amt) for amt in [2,3,4,5,8,12,16,24,32,64] for axis in range(3)]
if getenv("NOLOCALS"): actions += [Opt(op=OptOps.NOLOCALS)]

def get_test_global_size(global_size, max_global_size, var_vals):
  test_global_size = [sym_infer(sz, var_vals) for sz in global_size]
  input_size = prod(test_global_size)
  while prod(test_global_size) > max_global_size:
    for j in range(len(global_size)-1,-1,-1):
      if test_global_size[j] > 16:
        test_global_size[j] //= 2
        break
  return test_global_size, input_size / prod(test_global_size)

def _time_program(prg:UOp, var_vals:dict[str, int], rawbufs:list[Buffer], early_stop:float|None=None,
                  allow_test_size:int=True, max_global_size:int|None=65536, clear_l2=False, cnt=3, name="test", dev_timeout=False) -> list[float]:
  timeout = int(early_stop * 1e3) if dev_timeout and early_stop is not None and early_stop < math.inf else None
  factor = 1
  if allow_test_size and max_global_size is not None:
    global_size, factor = get_test_global_size(prg.arg.global_size, max_global_size, var_vals)
    prg = prg.replace(arg=replace(prg.arg, global_size=tuple(global_size)))
  call = prg.call(*[UOp.from_buffer(b) for b in rawbufs])
  tms, timer = [], time_call(call, var_vals, timeout=timeout, clear_l2=clear_l2)
  for _ in range(cnt):
    try: tms.append(next(timer) * factor)
    except AssertionError: return [math.inf] * cnt
    if early_stop is not None and early_stop < min(tms): break
  return tms

class TimeoutException(Exception): pass
def timeout_handler(signum, frame):
  if DEBUG >= 2: print("*** BEAM COMPILE TIMEOUT")
  raise TimeoutException()

def _try_compile(x:tuple[int,Scheduler]) -> tuple[int, tuple[UOp, float]|None]:
  # signal only works in the main thread; when beam runs inside a server worker
  # thread (e.g. ComfyUI) skip the compile timeout instead of crashing
  use_alarm = hasattr(signal, "alarm") and threading.current_thread() is threading.main_thread()
  if use_alarm:
    signal.signal(getattr(signal, 'SIGALRM'), timeout_handler)
    # set timeout
    signal.alarm(getenv("BEAM_TIMEOUT_SEC", 10))
  ret = None
  try:
    st = time.perf_counter()
    ast, dev = x[1].copy().get_optimized_ast(name_override="test"), x[1].ren.target.device
    prg = to_program(ast.substitute({p: p.replace(arg=replace(p.arg, device=dev)) for p in ast.toposort() if p.op is Ops.PARAM}), x[1].ren)
    et = time.perf_counter() - st
    uops = prg.src[1].src
    if len(uops) >= (uops_max:=getenv("BEAM_UOPS_MAX", 3000)) > 0:
      if getenv("BEAM_LOG_SURPASS_MAX"): print(f"too many uops. {len(uops)=}, {uops_max=}")
      raise RuntimeError("too many uops")
    ret = (prg, et)
  except RuntimeError:
    if DEBUG >= 4: traceback.print_exc()
  except Exception as e:
    if getenv("BEAM_STRICT_MODE"): raise e
  finally:
    if use_alarm: signal.alarm(0)
  return x[0], ret

def _ensure_buffer_alloc(bufs:list[Buffer]) -> list[Buffer]: return [buf.ensure_allocated() if buf is not None else buf for buf in bufs]

# *** winner verification (BEAM_VERIFY=1) ***
# tinygrad never checks that a fast candidate computes the RIGHT thing; on this rig the
# tensor-core action miscompiles some shapes and "wins" by being wrong (pure-noise images).
# Cheap defense: run the unoptimized baseline and the winner on identical pseudo-random
# inputs and compare outputs; a miscompile is wildly wrong, float reorder is not.

_VERIFY_HI = (0x3D, 0x3E, 0x3F, 0xBD, 0xBE, 0xBF)   # sign+exponent bytes: finite, signed, |x| in ~[0.03, 4] for f32/f16/bf16

def _fill_bufs_deterministic(bufs:list[Buffer]):
  # the byte pattern has period 64 in k, so build one period and tile it: an np.arange over a
  # 500 MB weight view (the LLM decode kernels take views of the GGUF) would spike host RAM by ~2 GB per fill (2026-09-02)
  # 2026-09-05: the old pattern (every byte <= 0x3F) only produced POSITIVE floats below 1.0, and a tensor-core winner for
  # MiniMax H3's fused swiglu+fc2 kernel passed on it while computing garbage on real (signed, ~N(0,0.5)) activations.
  # Odd bytes are the high byte of every 2-byte float and byte 3 of every 4-byte float: draw them from _VERIFY_HI so
  # the inputs are signed and of order 1; even bytes (mantissa) vary freely.
  for i, b in enumerate(bufs):
    if b is None: continue
    b.ensure_allocated()
    n = b.nbytes
    period = bytes((_VERIFY_HI[(k * 5 + 3 + i * 7) % len(_VERIFY_HI)] if k & 1 else (k * 37 + 11 + i * 31) & 0xFF) for k in range(64))
    Device[b.device].allocator._copyin(b._buf, memoryview(period * (n // 64) + period[:n % 64]))

def _read_out_f32(buf:Buffer):
  import numpy as np
  raw = bytes(buf.as_memoryview())
  fmt = getattr(buf.dtype, "fmt", None)
  if fmt in ("f", "e", "d"): return np.frombuffer(raw, dtype=np.dtype(fmt)).astype(np.float32)
  if "bfloat16" in str(buf.dtype): return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float32)
  if fmt is not None:
    try: return np.frombuffer(raw, dtype=np.dtype(fmt)).astype(np.float32)
    except (TypeError, ValueError): return None
  return None

def _run_once_full(sched:Scheduler, rawbufs:list[Buffer], var_vals:dict[str,int]) -> bool:
  _, proc = _try_compile((0, sched.copy()))
  if proc is None: return False
  tms = _time_program(proc[0], var_vals, rawbufs, cnt=1, allow_test_size=False, max_global_size=None)
  return all(t != math.inf for t in tms)

def _verify_winner(base:Scheduler, cand:Scheduler, rawbufs:list[Buffer], var_vals:dict[str,int]) -> bool:
  import numpy as np
  try:
    _fill_bufs_deterministic(rawbufs)
    if not _run_once_full(base, rawbufs, var_vals):
      if DEBUG >= 1: print("BEAM_VERIFY SKIPPED: baseline would not run; accepting")
      return True  # can't verify -> don't block
    ref = _read_out_f32(rawbufs[0])
    if ref is None:
      if DEBUG >= 1: print(f"BEAM_VERIFY SKIPPED: unreadable ref dtype {rawbufs[0].dtype}; accepting")
      return True
    _fill_bufs_deterministic(rawbufs)
    if not _run_once_full(cand, rawbufs, var_vals): return False
    got = _read_out_f32(rawbufs[0])
    if got is None:
      if DEBUG >= 1: print(f"BEAM_VERIFY SKIPPED: unreadable cand dtype {rawbufs[0].dtype}; accepting")
      return True
    ok = bool(np.allclose(ref, got, rtol=1e-2, atol=1e-3, equal_nan=True))
    if DEBUG >= 1:
      if ok: print(f"BEAM_VERIFY PASS ({ref.size} outputs match) for {cand.applied_opts}")
      else:
        bad = int((~np.isclose(ref, got, rtol=1e-2, atol=1e-3, equal_nan=True)).sum())
        print(f"BEAM_VERIFY REJECTED {cand.applied_opts} ({bad}/{ref.size} outputs mismatch)")
    return ok
  except Exception as e:
    if DEBUG >= 1: print(f"BEAM_VERIFY inconclusive ({type(e).__name__}: {e}); accepting candidate")
    return True

# *** external API ***

# get dictionary of all possible actions
def get_kernel_actions(s:Scheduler, include_0=True, max_up:int|None=None) -> dict[int, Scheduler]:
  acted, max_up, max_lcl = {0:s} if include_0 else {}, getenv("BEAM_UPCAST_MAX", 256) if max_up is None else max_up, getenv("BEAM_LOCAL_MAX", 1024)
  kernel_actions = actions.copy()

  for i,a in enumerate(kernel_actions):
    if a.axis is not None and a.op is not OptOps.TC:
      try: ax = s.real_axis(a.op, a.axis)
      except KernelOptError: continue
      if (ax >= s.shape_len) or (s.full_shape[ax] == a.arg and Opt(a.op, a.axis, 0) in kernel_actions): continue
    s2 = s.copy()
    try:
      s2.apply_opt(a)
      up, lcl, tc_up = 1, 1, prod(tc.dims)//tc.threads if hasattr(s2, 'tensor_core') and (tc:=s2.tensor_core) else 1
      for x,t in zip(s2.full_shape, s2.axis_types):
        if t in (AxisType.UPCAST, AxisType.UNROLL): up *= x
        elif t in (AxisType.WARP, AxisType.LOCAL, AxisType.GROUP_REDUCE): lcl *= x
      if up//tc_up > max_up or lcl > max_lcl:
        if getenv("BEAM_LOG_SURPASS_MAX"): print(f"too many upcast/local. {up//tc_up=}, {max_up=}, {lcl=}, {max_lcl=}")
        continue
      acted[i+1] = s2
    except KernelOptError: pass
  return acted

BEAM_DEBUG = getenv("BEAM_DEBUG")
def beam_search(s:Scheduler, rawbufs:list[Buffer], var_vals:dict[str,int], amt:int, allow_test_size=True, disable_cache=IGNORE_BEAM_CACHE.value):
  key = {"ast": s.ast.key, "amt": amt, "allow_test_size": allow_test_size, "device": s.ren.target.device, "suffix": s.ren.suffix}
  if not disable_cache and CACHELEVEL >= 1 and (val:=diskcache_get("beam_search", key)) is not None:
    ret = s.copy()
    for o in val[len(s.applied_opts):]: ret.apply_opt(o)
    return ret

  beam: list[tuple[Scheduler, float]] = [(s, float("inf"))]
  seen_libs, seen_srcs = set(), set()

  pool = get_worker_pool()

  min_progress = getenv("BEAM_MIN_PROGRESS", 0.01)/1e6
  if BEAM_DEBUG:
    print("BEAM_SEARCH:")
    print(pyrender(s.ast.replace(arg=None)))
  if DEBUG >= 2: print(f"   0.00s:                from   1 ->   1 actions {s.colored_shape()}")

  try:
    rawbufs = _ensure_buffer_alloc(rawbufs)
    exiting, st = False, time.perf_counter()
    dev = Device[s.ren.target.device]
    while not exiting:
      candidates: list[Scheduler] = flatten([get_kernel_actions(si, include_0=False).values() for si,_ in beam])
      timed: list[tuple[Scheduler, float]] = []
      least_compute_ops = math.inf
      # LOCAL PATCH: pool re-fetched per round (a timeout tears it down) and iterated with a bounded wait; candidates
      # whose worker died are simply not timed this round
      pool = get_worker_pool()
      for i, proc in (map(_try_compile, enumerate(candidates)) if pool is None else
                      pool_imap_bounded(pool, _try_compile, list(enumerate(candidates)))):
        if proc is None: continue
        prg, compile_et = proc
        # LOCAL PATCH: workers return uncompiled programs (codegen.in_worker_process). identical source -> identical
        # binary, so dedupe on source first, then compile here through the parent's single docker container.
        if len(prg.src) >= 3 and prg.src[2].op is Ops.SOURCE:
          if (src_txt:=prg.src[2].arg) in seen_srcs: continue
          seen_srcs.add(src_txt)
        try:
          cst = time.perf_counter()
          prg = finalize_program(prg, s.ren)
          compile_et += time.perf_counter() - cst
        except Exception as e:
          if BEAM_DEBUG: print(f"BEAM compile failed for opts: {candidates[i].applied_opts}\n{e}")
          continue
        if (lib:=prg.src[3].arg) in seen_libs: continue
        # filter out kernels that use 1000x more compute than the smallest
        estimates = prg.src[0].arg.estimates
        least_compute_ops = min(this_compute_ops:=sym_infer(estimates.ops if estimates is not None else 0, var_vals), least_compute_ops)
        if least_compute_ops*1000 < this_compute_ops:
          if getenv("BEAM_LOG_SURPASS_MAX"): print(f"too much compute. {this_compute_ops} when least is {least_compute_ops}")
          continue
        seen_libs.add(lib)
        try: tms = _time_program(prg, var_vals, rawbufs, early_stop=beam[0][1]*3 if len(beam) else 1.0,
                                 allow_test_size=allow_test_size, clear_l2=hasattr(dev, 'invalidate_caches'),
                                 dev_timeout=getenv("BEAM_DEV_TIMEOUT", 1))
        except Exception as e:
          if BEAM_DEBUG: print(f"BEAM failed for opts: {candidates[i].applied_opts}\n{e}")
          if isinstance(e, RuntimeError): continue
          raise
        timed.append((candidates[i], min(tms)))
        if BEAM_DEBUG > 1:
          print(f"{time.perf_counter() - st:7.2f}s: {i:5d} {len(prg.src[1].src):5d} uops",
                f"{time_to_str(compile_et, w=12)} compile/{time_to_str(timed[-1][1], w=12)} run",
                f"      {len(timed):4d}/{len(candidates):4d}         {timed[-1][0].colored_shape()}")
        elif DEBUG >= 2:
          print(f"\r{time.perf_counter() - st:7.2f}s: {time_to_str(timed[-1][1], w=12)}",
                f"      {len(timed):4d}/{len(candidates):4d}         {timed[-1][0].colored_shape()}\033[K", end="")

      # done
      opts = sorted(timed, key=lambda x: x[1])
      exiting = len(opts) == 0 or (opts[0][1] < min_progress) or (len(beam) > 0 and ((beam[0][1]-opts[0][1]) < min_progress))
      if not exiting: beam = opts[:amt]
      elif len(opts) > 0 and opts[0][1] < beam[0][1]: beam = opts[:1]
      if DEBUG >= 2:
        print(f"\r{time.perf_counter() - st:7.2f}s:", colored(time_to_str(beam[0][1], w=12), "green" if exiting else None),
              f"from {len(candidates):3d} -> {len(opts):3d} actions\033[K", beam[0][0].colored_shape())
  except KeyboardInterrupt as e:
    terminate_worker_pool()
    raise e

  if getenv("BEAM_VERIFY", 0):
    winner = None
    for cand, _tm in beam:
      if cand.applied_opts == s.applied_opts:
        # nothing beat the unoptimized kernel (or every candidate was dropped): banked without verification, say so
        if DEBUG >= 1: print(f"BEAM_VERIFY NOOP: best candidate is the unoptimized kernel {s.applied_opts}")
        winner = cand
        break
      if _verify_winner(s, cand, rawbufs, var_vals):
        winner = cand
        break
      if DEBUG >= 1: print(f"BEAM_VERIFY: dropping miscompiled winner, trying next candidate")
    if winner is None:
      if DEBUG >= 1: print("BEAM_VERIFY: every candidate failed verification; returning unoptimized kernel")
      winner = s
    if CACHELEVEL >= 1: diskcache_put("beam_search", key, winner.applied_opts)
    # LOCAL PATCH (2026-09-02): the audit line used to be unreachable under BEAM_VERIFY; log the banked winner's time too
    if BEAM_DEBUG: print(f"BEAM_SEARCH: final tm={time_to_str(next((t for c,t in beam if c is winner), float('inf')), w=0)}, applied_opts={winner.applied_opts}")
    return winner
  if CACHELEVEL >= 1: diskcache_put("beam_search", key, beam[0][0].applied_opts)
  if BEAM_DEBUG: print(f"BEAM_SEARCH: final tm={time_to_str(beam[0][1], w=0)}, applied_opts={beam[0][0].applied_opts}")
  return beam[0][0]

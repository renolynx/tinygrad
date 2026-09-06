from collections import defaultdict
from tinygrad.helpers import NO_MEMORY_PLANNER, DEBUG, round_up
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.runtime.support.memory import TLSFAllocator

# local patch #24 (2026-09-02): kernel args can be contiguous sub-buffer VIEWS of a buffer (SHRINK/BITCAST/RESHAPE... chains,
# see UOp.buffer / contiguous_view). The planner's lifetimes, prune_linear's input tracking and the jit's extra_held all go
# through this function, and none of them saw the underlying buffer: a kernel reading a chunk() view of a realized
# intermediate looked input-independent (prune hoisted it into the run-once prologue, where it read freed memory) and the
# base buffer's arena slot was reused while later kernels still read it through the view. Found with FLUX.2 Klein's global
# modulation vectors under TinyJit (LastLayer minimal repro); Krea 2 never produced such args.
_VIEW_OPS = {getattr(Ops, n) for n in ("SHRINK", "BITCAST", "RESHAPE", "PERMUTE", "EXPAND", "PAD", "FLIP", "CONTIGUOUS", "AFTER", "DETACH") if hasattr(Ops, n)}

def _collect_bufs(u:UOp) -> list[UOp]:
  if u.op is Ops.BUFFER: return [u]
  if u.op in {Ops.MSELECT, Ops.MSTACK}: return [b for s in u.src for b in _collect_bufs(s)]
  if u.op in _VIEW_OPS and len(u.src): return _collect_bufs(u.src[0])
  return []

def _can_plan(b:UOp, held_bufs:set[UOp]) -> bool:
  if b in held_bufs: return False
  devs = (b.device,) if isinstance(b.device, str) else b.device
  # CL and WEBGPU do not support views, see explanation in contiguous_view_offset
  return all(not d.startswith(("DISK", "TINYFS", "CL", "WEBGPU")) for d in devs)

LaneKey = tuple[str, int]

def memory_plan_rewrite(linear:UOp, held_bufs:set[UOp]|None=None) -> UOp:
  if NO_MEMORY_PLANNER: return linear
  if held_bufs is None: held_bufs = set()

  # local patch #25 (2026-09-02): a buffer that is READ before any call in this linear WRITES it carries data from outside
  # the captured graph (e.g. the NPY source of a CPU->device copy of a conditioning tensor that is not a jit input). Planning
  # it into the arena replaces it with a slot nothing ever fills, so the replay reads garbage. Such buffers are held.
  # (With held_bufs = "tensor-reachable" (patch #7) these temporaries are not reachable; upstream held everything.)
  from tinygrad.engine.realize import get_call_outs_ins
  def _out_slots(si:UOp) -> set[int]|None:
    outs, _ = get_call_outs_ins(si)
    if outs: return set(outs)
    if si.src[0].op is not Ops.SINK: return None   # unknown call type: no external detection for it
    res: set[int] = set()
    for st in si.src[0].toposort():
      if st.op is Ops.STORE:
        u = st.src[0]
        while u.op not in (Ops.PARAM, Ops.BUFFER, Ops.MSELECT, Ops.MSTACK) and len(u.src): u = u.src[0]
        if u.op is Ops.PARAM and hasattr(u.arg, "slot"): res.add(u.arg.slot)
    return res if res else None
  external: set[UOp] = set()
  written: set[UOp] = set()
  for si in linear.src:
    outs = _out_slots(si)
    if outs is None: continue
    args = [a for a in si.src[1:] if not a.is_bound_var]
    for j, a in enumerate(args):
      for b in _collect_bufs(a):
        if j in outs: written.add(b)
        elif b not in written: external.add(b)
  if external: held_bufs = held_bufs | external

  # compute lifetimes for all plannable internal buffers
  first_appearance:dict[UOp, int] = {}
  last_appearance:dict[UOp, int] = {}
  copy_bufs: set[UOp] = set()
  for i, si in enumerate(linear.src):
    si_bufs = [b for src in si.src[1:] for b in _collect_bufs(src) if _can_plan(b, held_bufs)]
    for b in si_bufs:
      if b not in first_appearance: first_appearance[b] = i
      last_appearance[b] = i
    if si.src[0].op is Ops.COPY: copy_bufs.update(si_bufs)
  if not first_appearance: return linear

  # separate copy and compute buffers into different lanes to avoid introducing dependencies (copy->compute->copy)
  def _key(b:UOp): return (b.device, 1 if b in copy_bufs else 0)
  buf_hold = {b: last_appearance[b] - first_appearance[b] + 1 for b in first_appearance if b in copy_bufs}

  # suballocation: build sorted open/close events, then alloc/free in order
  block_size = 256
  nbytes = {b: round_up(b.max_numel() * b.dtype.itemsize, block_size) for b in first_appearance}
  events = sorted([(first_appearance[b], True, b) for b in first_appearance] +
                  [(last_appearance[b] + 1 + buf_hold.get(b, 0), False, b) for b in first_appearance], key=lambda x: (x[0], x[1]))
  total_memory = sum(nbytes.values()) * 2

  offsets:dict[UOp, int] = {}
  peaks:dict[LaneKey, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=block_size, lv2_cnt=32)))
  for _, is_open, buf in events:
    if is_open: offsets[buf] = peaks[_key(buf)][1].alloc(nbytes[buf])
    else: peaks[_key(buf)][1].free(offsets[buf])
    peaks[_key(buf)] = (max(peaks[_key(buf)][0], offsets[buf] + buf.max_numel() * buf.dtype.itemsize), peaks[_key(buf)][1])
  arena_sizes = {key: round_up(peak, block_size) for key, (peak, _) in peaks.items()}

  # build replace_map: each buffer becomes a SHRINK/BITCAST into a shared per-device-lane arena
  arenas = {key: UOp.new_buffer(key[0], sz, dtypes.int8) for key, sz in arena_sizes.items()}
  replace_map = {buf_uop:arenas[_key(buf_uop)][offset:offset+buf_uop.nbytes()].bitcast(buf_uop.dtype) for buf_uop, offset in offsets.items()}

  if DEBUG >= 1 and (omem:=sum(nbytes.values()) / 1e6) != (nmem:=sum(arena_sizes.values()) / 1e6):
    print(f"memory reduced from {omem:.2f} MB -> {nmem:.2f} MB, {len(first_appearance)} -> {len(arenas)} bufs")

  return linear.substitute(replace_map, name="memory plan", walk=True)

# LOCAL PATCH #19 (2026-09-02): hand-written NVIDIA tensor-core GEMM for the LLM prefill (tinygrad UOps, no CUDA source).
#   out[M, N] (float32) = x[M, K] (float16) @ w[N, K]^T (float16), fp32 accumulate, via mma.sync.m16n8k16 (sm_80+).
# Why: on the 4080 the generic fp16 prefill matmul of LLM_DEQUANT_ONCE (patch #18) is a beam-tuned CUDA-core kernel at
# ~320 GB/s on the 89M-param FFN weights (0.56 ms gate/up, 0.93 ms down). tinygrad's own TC path never matched it
# (float operands after the cast -> only the TF32 core, ~5 TFLOPS). This kernel streams the fp16 weight with 16-byte
# loads straight into mma fragments: 0.30 ms gate/up (600 GB/s), ~0.55 ms down, verified against numpy at every shape.
# Design (one warp per block, no shared memory, the weight is read exactly once):
#   * a warp owns NT n-tiles of 8 weight rows and all MT=M/16 m-tiles; lane = 4*groupID + tid as in the PTX ISA layout.
#   * inside a 32-wide k chunk lane t owns k = 8t..8t+7 of each row it touches: one uint4 (LDG.128, CSE'd out of four
#     .xyzw reads of the same pointer through Ops.CUSTOM) for the weight row and for each activation row. mma step 0
#     consumes halves 0-3, step 1 halves 4-7. The k order inside the chunk is permuted identically for A and B, which the
#     dot product does not notice, so no shuffles are needed to satisfy the fragment layout.
#   * U chunks per loop iteration (more bytes in flight), SPLITK blocks over k (partials summed by the caller) so small-N
#     shapes still fill the 80 SMs. Accumulators live in REG placeholders like kernels/amd.py.
# Correctness on the model: diagnostics/llm_dequant_check.py, ONE PROCESS PER SETTING (getenv is cached per process).
from __future__ import annotations
import functools, os
from tinygrad import Tensor, UOp, Device, Context
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

WMMA_ARG = ((8, 16, 16), "CUDA", 32)   # (N, M, K) of mma.m16n8k16 as tinygrad's cuda_81616 TensorCore, 32 lanes

@functools.cache
def nv_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] not in ("NV", "CUDA"): return False
  with Context(ALLOW_DEVICE_USAGE=1):
    arch = getattr(getattr(Device[device].renderer, "target", None), "arch", "")
  try: return arch.startswith("sm_") and int(arch[3:]) >= 80
  except ValueError: return False

def nv_gemm_config(M:int, N:int, K:int) -> tuple[int, int, int]|None:
  """(U, NT, SPLITK) for out[M,N] = x[M,K] @ w[N,K]^T, or None when the shape does not tile.
  From the 2026-09-02 sweep on the 4080 (diagnostics/nv_tc/v6_shapes.log): U=2 wins everywhere, NT=2 only pays for
  N>=16384 (gate/up), split-K only for tiny N (<=1024: x4) or N<=5120 with a long K (down: x2)."""
  if M % 16 or N % 8 or K % 32 or M > 256 or (M > 32 and M % 32): return None
  NT = 2 if N >= 16384 and N % 16 == 0 else 1
  U = 2 if K % 64 == 0 else 1
  splitk = 4 if N <= 1024 else 2 if N <= 5120 and K >= 16384 else 1
  while splitk > 1 and K % (32 * U * splitk): splitk //= 2
  return U, NT, splitk

def _half(v:UOp) -> UOp: return v.cast(dtypes.uint16).bitcast(dtypes.float16)

def _load16(buf:UOp, *idx) -> tuple[UOp, ...]:
  """8 consecutive halves at buf[*idx] through one 16-byte load; the four component reads of the same pointer are one LDG.128"""
  ptr = buf[idx]
  words = [UOp(Ops.CUSTOM, src=(ptr,), arg=(f"(*(const uint4*)({{0}})).{c}", dtypes.uint32)) for c in "xyzw"]
  return tuple(_half(wd >> (16 * j)) for wd in words for j in range(2))

@functools.cache
def _nv_f16_gemm_kernel(out:UOp, a:UOp, w:UOp, M:int, N:int, K:int, U:int, NT:int, SPLITK:int) -> UOp:
  # rows per block: 32 (two m-tiles). Bigger M is split over blockIdx.x so the blocks of one n-tile run back to back and
  # the second one finds the weight rows in L2 (measured at M=64: one block per 64 rows = 425 GB/s, register pressure
  # and the doubled activation traffic; 32-row blocks keep the M=32 speed per block).
  MB = 32 if M % 32 == 0 else M
  MT = MB // 16
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  mh = UOp.range(M // MB, 0)
  nb = UOp.range(N // (8 * NT), 1)
  sk = UOp.range(SPLITK, 2)
  g, t = lane // 4, lane % 4
  tiles = [nb * NT + j for j in range(NT)]
  kchunks = K // (32 * U * SPLITK)
  kc = UOp.range(kchunks, 3, AxisType.REDUCE)
  accs = [[UOp.placeholder((4,), dtypes.float32, slot=mt * NT + j, addrspace=AddrSpace.REG) for j in range(NT)] for mt in range(MT)]
  accs = [[acc.after(acc.store(acc.const_like(0))) for acc in row] for row in accs]
  cur = [[UOp.stack(*(accs[mt][j].after(kc)[i].load() for i in range(4))) for j in range(NT)] for mt in range(MT)]
  for u in range(U):
    k0 = ((sk * kchunks + kc) * U + u) * 32 + t * 8
    wv = [_load16(w, tile * 8 + g, k0) for tile in tiles]
    av = [(_load16(a, mh * MB + mt * 16 + g, k0), _load16(a, mh * MB + mt * 16 + g + 8, k0)) for mt in range(MT)]
    for s in range(2):
      b = 4 * s
      bfrags = [UOp.stack(*wv[j][b:b+4]) for j in range(NT)]
      for mt in range(MT):
        a0, a1 = av[mt]
        afrag = UOp.stack(a0[b], a0[b+1], a1[b], a1[b+1], a0[b+2], a0[b+3], a1[b+2], a1[b+3])
        for j in range(NT): cur[mt][j] = UOp.wmma(afrag, bfrags[j], cur[mt][j], *WMMA_ARG)
  update = UOp.group(*(accs[mt][j].store(cur[mt][j]) for mt in range(MT) for j in range(NT))).end(kc)
  stores = []
  for mt in range(MT):
    for j in range(NT):
      acc = accs[mt][j].after(update)
      r0, c0 = mh * MB + mt * 16 + g, tiles[j] * 8 + 2 * t
      stores += [out[sk, r0, c0].store(acc[0].load()), out[sk, r0, c0 + 1].store(acc[1].load()),
                 out[sk, r0 + 8, c0].store(acc[2].load()), out[sk, r0 + 8, c0 + 1].store(acc[3].load())]
  name = f"nv_f16_gemm_m{M}_n{N}_k{K}_u{U}_t{NT}_s{SPLITK}"
  return UOp.group(*stores).end(mh, nb, sk, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))

def nv_f16_gemm(x:Tensor, w:Tensor) -> Tensor:
  """x[M, K] half (contiguous), w[N, K] half (contiguous) -> [M, N] float32"""
  M, K = x.shape
  N = w.shape[0]
  assert isinstance(M, int) and isinstance(K, int) and isinstance(N, int) and w.shape[1] == K
  assert x.dtype == dtypes.half and w.dtype == dtypes.half, "nv_f16_gemm needs half inputs"
  cfg = nv_gemm_config(M, N, K)
  assert cfg is not None, f"nv_f16_gemm: shape {(M, N, K)} does not tile"
  U, NT, SPLITK = cfg
  out = Tensor.empty(SPLITK, M, N, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_nv_f16_gemm_kernel, M=M, N=N, K=K, U=U, NT=NT, SPLITK=SPLITK)
  res = Tensor.custom_kernel(out, x, w, fxn=fxn)[0]
  return res[0] if SPLITK == 1 else res.sum(0)


# ******** LOCAL PATCH #23 (2026-09-02): flash attention forward for NV (FlashAttention-2 structure in UOps) ********
# out[bh, n, :] = softmax(q[bh, n, :] . k[bh//GQA, :, :]^T * scale) . v[bh//GQA]   (q, k, v half; out float32)
# One block = 64 query rows (4 warps x 16 rows), looping over 64-key tiles. Per warp and tile: S = Q K^T through
# mma.m16n8k16 (the same 32-wide k permutation as the GEMM: lane t owns d = 8t..8t+7 of each 32-chunk for both Q and K),
# online softmax on the S fragments in registers (row max/sum across the 4 lanes of a row with __shfl_xor), P is
# reinterpreted as the A fragment of the P.V mma without any shuffle (the C layout of m16n8 tiles 2j, 2j+1 is exactly the
# A layout of a 16-wide k chunk), V is staged transposed in shared memory (double-buffered, one barrier per tile) so the
# B fragments are contiguous half pairs. Queries beyond N are clamped for loads and skipped for stores; keys beyond NK
# (and, when causal, beyond the query position q_start + row) are masked to -inf before the softmax.
import math

def _shfl_xor(v:UOp, off:int) -> UOp: return UOp(Ops.CUSTOM, src=(v,), arg=(f"__shfl_xor_sync(0xffffffffu, {{0}}, {off})", v.dtype))
def _row_max(v:UOp) -> UOp:
  v = v.maximum(_shfl_xor(v, 1))
  return v.maximum(_shfl_xor(v, 2))
def _row_sum(v:UOp) -> UOp:
  v = v + _shfl_xor(v, 1)
  return v + _shfl_xor(v, 2)

def _kernel_var(x:UOp) -> UOp:
  return x.substitute({v: UOp.variable(v.expr, v.vmin, v.vmax, dtype=v.dtype, multiple_of=v.arg.multiple_of, param=True)
                       for v in x.toposort() if v.is_variable})

@functools.cache
def _nv_flash_kernel(out:UOp, q:UOp, k:UOp, v:UOp, BH:int, N:int, NK:int, D:int, GQA:int, causal:bool, q_start, scale_log2e:float,
                     DV:int=0, dv0:int=0, cache_mode:bool=False, kvx:UOp|None=None, KVHI:int=0, out_nbh:bool=False) -> UOp:
  # Local patch #34 (2026-09-05, MiniMax H3 length-agnostic capture): kvx is an int32[1] buffer holding kv_lo; keys in
  # [kv_lo, KVHI) are masked out. Read at kernel run time, so one captured graph serves every prompt whose padded text
  # segment is KVHI rows long.
  # LOCAL PATCH #43 (2026-09-06): tile shape from the env for tuning (BM = 16 * WAVES; BN a multiple of 16 with
  # 2*DV*(BN+2) halves of shared memory, so <= 80 at DV=128). Non-default shapes get their own kernel name.
  # Sweep 2026-09-06 at the H3 DiT shape (56 heads, 8640 x 8615, d 128): (4,64) 39 TFLOPS, (4,80) 44, (8,64) 58,
  # (8,80) 73, (16,64) 69, (16,80) 73; Klein's 4115-row case 54 -> 70 TFLOPS. 8 waves x 80 keys is the default now.
  WAVES, BN = int(os.environ.get("NV_FLASH_WAVES", "8")), int(os.environ.get("NV_FLASH_BN", "80"))
  BM = 16 * WAVES
  assert BN % 16 == 0 and 2 * (DV or D) * (BN + 2) * 2 <= 48 * 1024, "flash kernel: BN must be a multiple of 16 and the V tile must fit shared memory"
  DV = DV or D
  assert D % 32 == 0 and DV % 8 == 0 and DV <= 128 and dv0 + DV <= D, "flash kernel: head dim multiple of 32, output slice <= 128"
  if isinstance(q_start, UOp): q_start = _kernel_var(q_start.unbind_all()[0])
  DN, KT = DV // 8, BN // 16                    # output n-tiles per lane, 16-key chunks per tile
  if cache_mode:  # k is the (2, B, HKV, NK, D) cache; v is unused (same buffer)
    kload = lambda bkv, key, d0: _load16(k, 0, bkv // k.shape[2], bkv % k.shape[2], key, d0)
    vload = lambda bkv, key, d0: _load16(k, 1, bkv // k.shape[2], bkv % k.shape[2], key, d0)
  else:
    kload = lambda bkv, key, d0: _load16(k, bkv, key, d0)
    vload = lambda bkv, key, d0: _load16(v, bkv, key, d0)
  kv_lo = kvx[(UOp.const(0, dtypes.int),)].load() if kvx is not None else None
  ntiles_q, ntiles_k = (N + BM - 1) // BM, (NK + BN - 1) // BN
  qb, bh = UOp.range(ntiles_q, 0), UOp.range(BH, 1)
  wave, lane = UOp.range(WAVES, 3, axis_type=AxisType.LOCAL), UOp.range(32, -1, axis_type=AxisType.WARP)
  g, t, tid = lane // 4, lane % 4, wave * 32 + lane
  bkv = bh // GQA
  rows = (qb * BM + wave * 16 + g, qb * BM + wave * 16 + g + 8)
  rows_c = tuple(r.minimum(N - 1) for r in rows)
  vt = UOp.placeholder((2, DV, BN + 2), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  accs = [UOp.placeholder((4,), dtypes.float32, slot=1 + i, addrspace=AddrSpace.REG) for i in range(DN)]
  accs = [a.after(a.store(a.const_like(0))) for a in accs]
  mreg = UOp.placeholder((2,), dtypes.float32, slot=1 + DN, addrspace=AddrSpace.REG)
  mreg = mreg.after(mreg.store(mreg.const_like(-math.inf)))
  lreg = UOp.placeholder((2,), dtypes.float32, slot=2 + DN, addrspace=AddrSpace.REG)
  lreg = lreg.after(lreg.store(lreg.const_like(0)))
  # the S tiles go through REG placeholders too: elements of a WMMA value cannot be indexed directly (renders as pointer math)
  sregs = [UOp.placeholder((4,), dtypes.float32, slot=3 + DN + nt, addrspace=AddrSpace.REG) for nt in range(BN // 8)]
  tile = UOp.range(ntiles_k, 2, AxisType.REDUCE)
  buf = tile % 2
  # 1. this tile of V, transposed into shared memory: every thread moves BN*D/8/128 uint4 (8 halves of one key row)
  vt_t = vt.after(tile)
  vstores = []
  total_v = BN * DV // 8                                   # uint4 loads for this V tile
  for r in range((total_v + WAVES * 32 - 1) // (WAVES * 32)):
    idx = tid + r * (WAVES * 32)
    full = (r + 1) * (WAVES * 32) <= total_v                 # patch #43: a partial last round (BN*DV/8 not a multiple of
    idx_c = idx if full else idx.minimum(total_v - 1)        # the thread count, e.g. BN=80 at D=64) is masked, not dropped
    key, dch = idx_c // (DV // 8), idx_c % (DV // 8)
    vals = vload(bkv, (tile * BN + key).minimum(NK - 1), dv0 + dch * 8)
    vstores += [vt_t[buf, dch * 8 + j, key if full else key.valid(idx < total_v)].store(vals[j]) for j in range(8)]
  vt_ready = vt.after(UOp.barrier(UOp.group(*vstores)))
  # 2. S = Q K^T: 8 key n-tiles of 8, D/16 k-steps each. The mma chains start from stacks of REG loads and end in whole-
  # vector REG stores exactly like the GEMM kernel: other forms (stacks of consts / ALU values) render as pointer math.
  zs = UOp.group(*(sregs[nt].after(tile)[i].store(UOp.const(0.0, dtypes.float32)) for nt in range(BN // 8) for i in range(4)))
  s = [UOp.stack(*(sregs[nt].after(zs)[i].load() for i in range(4))) for nt in range(BN // 8)]
  for c in range(D // 32):
    qa, qb8 = _load16(q, bh, rows_c[0], 32 * c + 8 * t), _load16(q, bh, rows_c[1], 32 * c + 8 * t)
    for nt in range(BN // 8):
      kv = kload(bkv, (tile * BN + 8 * nt + g).minimum(NK - 1), 32 * c + 8 * t)
      for sh in range(2):
        b = 4 * sh
        afrag = UOp.stack(qa[b], qa[b+1], qb8[b], qb8[b+1], qa[b+2], qa[b+3], qb8[b+2], qb8[b+3])
        s[nt] = UOp.wmma(afrag, UOp.stack(*kv[b:b+4]), s[nt], *WMMA_ARG)
  # 3. scale, mask, online softmax (values kept in the log2 domain: s' = s*scale*log2e)
  sst = UOp.group(*(sregs[nt].store(s[nt]) for nt in range(BN // 8)))   # no .after(tile) here: it turns the vector store into pointer math
  neg_inf = UOp.const(-math.inf, dtypes.float32)
  sv = [[None] * 4 for _ in range(BN // 8)]
  for nt in range(BN // 8):
    for i in range(4):
      x = sregs[nt].after(sst)[i].load() * scale_log2e
      key = tile * BN + 8 * nt + 2 * t + (i % 2)
      conds = []
      if NK % BN: conds.append(key < NK)
      if causal: conds.append(key <= rows[i // 2] + q_start)
      if kvx is not None: conds.append((key < kv_lo) | (key >= KVHI))
      if conds:
        cond = conds[0]
        for cd in conds[1:]: cond = cond & cd
        x = cond.where(x, neg_inf)
      sv[nt][i] = x
  m_old = (mreg.after(tile)[0].load(), mreg.after(tile)[1].load())
  l_old = (lreg.after(tile)[0].load(), lreg.after(tile)[1].load())
  m_new, alpha, l_new, p = [], [], [], [[None] * 4 for _ in range(BN // 8)]
  for r in range(2):
    tmax = sv[0][2 * r].maximum(sv[0][2 * r + 1])
    for nt in range(1, BN // 8): tmax = tmax.maximum(sv[nt][2 * r]).maximum(sv[nt][2 * r + 1])
    mn = m_old[r].maximum(_row_max(tmax))
    m_safe = mn.eq(neg_inf).where(UOp.const(0.0, dtypes.float32), mn)   # a fully masked row so far: keep exp2 finite
    tsum = None
    for nt in range(BN // 8):
      for i in (2 * r, 2 * r + 1):
        p[nt][i] = (sv[nt][i] - m_safe).exp2()
        tsum = p[nt][i] if tsum is None else tsum + p[nt][i]
    a = (m_old[r] - m_safe).exp2()
    m_new.append(mn); alpha.append(a); l_new.append(l_old[r] * a + _row_sum(tsum))
  # 4. O = O * alpha + P V (P as half A fragments, V B fragments from the transposed shared tile)
  scaled = UOp.group(*(accs[nd][i].store(accs[nd].after(tile)[i].load() * alpha[i // 2]) for nd in range(DN) for i in range(4)))
  cur = [UOp.stack(*(accs[nd].after(scaled)[i].load() for i in range(4))) for nd in range(DN)]
  for j in range(KT):
    afrag = UOp.stack(*(p[2 * j][i].cast(dtypes.half) for i in range(4)), *(p[2 * j + 1][i].cast(dtypes.half) for i in range(4)))
    for nd in range(DN):
      d = 8 * nd + g
      bfrag = UOp.stack(vt_ready[buf, d, 16 * j + 2 * t].load(), vt_ready[buf, d, 16 * j + 2 * t + 1].load(),
                        vt_ready[buf, d, 16 * j + 8 + 2 * t].load(), vt_ready[buf, d, 16 * j + 9 + 2 * t].load())
      cur[nd] = UOp.wmma(afrag, bfrag, cur[nd], *WMMA_ARG)
  update = UOp.group(*(accs[nd].store(cur[nd]) for nd in range(DN)), *(mreg[r].store(m_new[r]) for r in range(2)),
                     *(lreg[r].store(l_new[r]) for r in range(2))).end(tile)
  # 5. normalise and store the valid rows
  odt = out.dtype
  lf = lreg.after(update)
  inv = (1.0 / lf[0].load(), 1.0 / lf[1].load())
  stores = []
  for nd in range(DN):
    o = accs[nd].after(update)
    c0 = dv0 + 8 * nd + 2 * t
    for r in range(2):
      row = rows[r] if N % BM == 0 else rows[r].valid(rows[r] < N)
      # out_nbh (2026-09-05, MiniMax H3): write [N, BH, D] = the (S, H*D) row-major layout the next linear reads, in
      # the buffer's own dtype, so no relayout/cast pass over a full-width fp32 output follows the kernel
      slot = (lambda c: out[row, bh * D + c]) if out_nbh else (lambda c: out[bh, row, c])   # out_nbh: out is [N, BH*D]
      stores += [slot(c0).store((o[2 * r].load() * inv[r]).cast(odt)), slot(c0 + 1).store((o[2 * r + 1].load() * inv[r]).cast(odt))]
  name = f"nv_flash_bh{BH}_n{N}_nk{NK}_d{D}_v{dv0}_{DV}_g{GQA}{'_causal' if causal else ''}{'_cache' if cache_mode else ''}{f'_x{KVHI}' if kvx is not None else ''}{'_nbh' if out_nbh else ''}{'' if odt == dtypes.float32 else '_o' + odt.name}{'' if (WAVES, BN) == (4, 64) else f'_w{WAVES}b{BN}'}"
  return UOp.group(*stores).end(wave, lane).end(qb, bh).sink(arg=KernelInfo(name=name, opts_to_apply=()))

def nv_flash_attention(q:Tensor, k:Tensor, v:Tensor, causal:bool=False, q_start:int=0, scale:float|None=None,
                       kv_lo:Tensor|None=None, kv_hi:int=0, out_dtype=dtypes.float32, out_nbh:bool=False) -> Tensor:
  """q [BH, N, D], k/v [BH_KV, NK, D] (half, BH % BH_KV == 0) -> [BH, N, D] float32 softmax(q k^T * scale) v.
  kv_lo (int32 [1] tensor on the device) + kv_hi: keys in [kv_lo, kv_hi) are excluded (patch #34).
  out_dtype / out_nbh (2026-09-05): the kernel can write half/bf16 and the [N, BH*D] row-major layout directly."""
  BH, N, D = q.shape
  BHK, NK, DK = k.shape
  assert all(isinstance(x, int) for x in (BH, N, D, BHK, NK)) and DK == D and v.shape == k.shape and BH % BHK == 0
  q, k, v = (x.cast(dtypes.half).contiguous() for x in (q, k, v))
  out = Tensor.empty(*((N, BH * D) if out_nbh else (BH, N, D)), dtype=out_dtype, device=q.device)
  sl = (D ** -0.5 if scale is None else scale) * math.log2(math.e)
  if kv_lo is not None: kv_lo = kv_lo.cast(dtypes.int32).reshape(1).contiguous()
  for dv0 in range(0, D, 128):  # D > 128: one pass per 128-wide output slice (S is recomputed, V/O tiles stay small)
    if kv_lo is None:
      fxn = functools.partial(_nv_flash_kernel, BH=BH, N=N, NK=NK, D=D, GQA=BH // BHK, causal=causal, q_start=q_start,
                              scale_log2e=sl, DV=min(128, D - dv0), dv0=dv0, out_nbh=out_nbh)
      out = Tensor.custom_kernel(out, q, k, v, fxn=fxn)[0]
    else:
      fxn = functools.partial(_nv_flash_kernel, BH=BH, N=N, NK=NK, D=D, GQA=BH // BHK, causal=causal, q_start=q_start,
                              scale_log2e=sl, DV=min(128, D - dv0), dv0=dv0, KVHI=int(kv_hi), out_nbh=out_nbh)
      fxn5 = lambda o, qq, kk, vv, kx, _f=fxn: _f(o, qq, kk, vv, kvx=kx)
      out = Tensor.custom_kernel(out, q, k, v, kv_lo, fxn=fxn5)[0]
  return out

def nv_llm_flash_attention(q:Tensor, cache:Tensor, valid_end) -> Tensor:
  """LLM prefill attention on the KV cache (patch #23): q [B, H, T, D] (T static or symbolic), cache [2, B, HKV, NK, D] half
  (keys/values up to valid_end are the history + this chunk), causal against position valid_end - T + row. Returns [B, H, T, D]."""
  from tinygrad.uop.ops import resolve as _resolve
  B, H, T_real, D = q.shape
  _, _, HKV, NK, DC = cache.shape
  assert DC == D and H % HKV == 0
  q_start, T = valid_end - T_real, T_real
  if isinstance(T_real, UOp):  # symbolic chunk: pad the queries to the static chunk size, garbage rows are sliced off
    T = q.max_shape[2]
    q = q.pad_to((B, H, T, D))
  qh = q.cast(dtypes.half).reshape(B * H, T, D).contiguous()
  out = Tensor.empty(B * H, T, D, dtype=dtypes.float32, device=q.device)
  sl = D ** -0.5 * math.log2(math.e)
  for dv0 in range(0, D, 128):
    fxn = functools.partial(_nv_flash_kernel, BH=B * H, N=T, NK=NK, D=D, GQA=H // HKV, causal=True, q_start=q_start,
                            scale_log2e=sl, DV=min(128, D - dv0), dv0=dv0, cache_mode=True)
    out = Tensor.custom_kernel(out, qh, cache, cache, fxn=fxn)[0]
  out = out.reshape(B, H, T, D)
  return out if not isinstance(T_real, UOp) else out[:, :, :T_real]

# ******** LOCAL PATCH #29 (2026-09-04): real row gather for NV ********
# out[i, :] = src[idx[i], :].  tinygrad has no O(n) gather in the forward path - Tensor.gather, tensor __getitem__ and
# nn.Embedding all go through _one_hot_along_dim, which costs O(n_out * n_in * C). That is fine for an embedding lookup
# and quadratic for a permutation: SeedVR2's DiT permutes its tokens into attention windows and back, and at 2048 px
# that was 198 one-hot matmuls totalling 409 TFLOP - about 150 s of a 272 s DiT - to move 98 MB of rows.
def _nv_gather_kernel(out:UOp, src:UOp, idx:UOp, N:int, C:int, BLOCK:int) -> UOp:
  i = UOp.range(N, 0)
  jb = UOp.range(C // BLOCK, 1)
  jl = UOp.range(BLOCK, 2, axis_type=AxisType.LOCAL)
  j = jb * BLOCK + jl
  row = idx[i].clip(0, src.shape[0] - 1).cast(dtypes.weakint)
  st = out[i, j].store(src[row, j].load())
  return st.end(i, jb, jl).sink(arg=KernelInfo(name=f"nv_gather_{N}_{C}", opts_to_apply=()))

def nv_gather(src:Tensor, idx:Tensor) -> Tensor:
  """src [S, C], idx [N] int -> [N, C] with out[i] = src[idx[i]]. C must be a multiple of 32."""
  S, C = src.shape
  N = idx.shape[0]
  assert all(isinstance(x, int) for x in (S, C, N)) and C % 32 == 0, f"nv_gather: bad shape {src.shape} {idx.shape}"
  BLOCK = next((b for b in (256, 128, 64, 32) if C % b == 0), 32)
  out = Tensor.empty(N, C, dtype=src.dtype, device=src.device)
  fxn = functools.partial(_nv_gather_kernel, N=N, C=C, BLOCK=BLOCK)
  return Tensor.custom_kernel(out, src.contiguous(), idx.cast(dtypes.int).contiguous(), fxn=fxn)[0]

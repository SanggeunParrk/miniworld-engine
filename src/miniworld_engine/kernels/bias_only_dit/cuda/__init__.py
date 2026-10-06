"""CUDA kernels of the bias-only token DiT (B200, sm_100a), inference.

  pv_gate_inf.cu          the attention core, sigmoid(g) * (P v) per head and sample, P = softmax(pair bias) made once per
                          sample() -- a tcgen05 GEMM with the gate in its epilogue (``PvGateCore``)
  bias_only_dit_rows.cu   every row pass between the GEMMs (one warp per row): the conditioning LayerNorm, the input AdaLN
                          with the fp32 residual copy, residual + gate + AdaLN, the last residual written in the output dtype,
                          SwiGLU, and the softmax of the hoisted bias

The core is a cubin built on first use into ``MINIWORLD_ENGINE_JIT_ROOT`` and launched through the sm_100a driver of
``kernels/augmented_attention/cuda/sm100`` (TMA descriptors, current stream, CUDA-graph capturable); the rows are a
torch extension. Neither is built at import.

The fp32 path (TF32 tensor cores: ``pv_gate_tf32.cu``, ``dpb32_sm100.cu``, ``gemm_glu_tf32.cu``, ``bias_only_dit_f32_rows.cu``) is
``tf32.py``.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_dir = Path(__file__).parent
H, DH = 16, 48                                         # the default heads x width
#: (heads, head width) served: 768 attention channels as 16 x 48, 24 x 32 or 12 x 64, and 1024 as 16 x 64
LAYOUTS = ((16, 48), (24, 32), (12, 64), (16, 64))


def _head_defs(nh: int, dh: int) -> tuple[str, ...]:
    """Compile definitions of the sm_100a cubins for a head layout (none for the default, so its cubins keep their keys)."""
    assert (nh, dh) in LAYOUTS, (nh, dh)
    return () if (nh, dh) == (H, DH) else (f"NHEAD={nh}", f"DHEAD={dh}")


@functools.lru_cache(maxsize=None)
def _ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_rows_cuda",
        sources=[str(_dir / "bias_only_dit_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes("80", "90", "100", ptx=("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


def ln_rows(x, out, eps=1e-5):
    """out = LN(x), no affine; x [M, 384]."""
    _ext().ln_rows_cuda(x, out, float(eps))


def adaln_in_rows(single, x, ms, mb, out, T, eps=1e-5, presig=False):
    """x = single (fp32), out = LN(x) * sigmoid(ms[row % T]) + mb[row % T]; width 768. ``presig``: ms holds the sigmoid."""
    _ext().adaln_in_rows_cuda(single, x, ms, mb, out, int(T), float(eps), bool(presig))


def resgate_adaln_rows(x, y, gl, ms, mb, out, T, eps=1e-5, presig=False):
    """x += sigmoid(gl[row % T]) * y in place (fp32); then, when ms is given, out = AdaLN(x) for the next half-block."""
    _ext().resgate_adaln_rows_cuda(x, y, gl, ms, mb, out, int(T), float(eps), bool(presig))


def swiglu_rows(ab, out):
    """out = silu(a) * b for ab = [a | b] (bf16)."""
    _ext().swiglu_rows_cuda(ab, out)


def resgate_out_rows(x, y, gl, out, T, presig=False):
    """out = x + sigmoid(gl[row % T]) * y in out's dtype; width 768."""
    _ext().resgate_out_rows_cuda(x, y, gl, out, int(T), bool(presig))


def softmax_rows(bias, p, mask=None):
    """p = softmax over the last axis of bias [R, L] bf16 (p may be bias); ``mask`` [L] bool drops the False keys."""
    _ext().softmax_rows_cuda(bias, p, mask)


def softmax_t(bias, p, pt, mask=None):
    """softmax_rows that also writes pt[h] = p[h]^T (bias, p, pt [16 L, L] bf16; p may be bias, pt may not)."""
    _ext().softmax_t_cuda(bias, p, pt, mask)


# --------------------------------------------------------------------------------------------------- the core
_SMEM_MAX = 232448


#: pv_gate_inf.cu's layout: as many 16 KB ring slots as fit beside three 16 KB g staging tiles, 512 B of barriers
_SMEM = (_SMEM_MAX - 3 * 16384 - 1024) // 16384 * 16384 + 3 * 16384 + 512


def _pv_smem(vb: int) -> int:
    """pv_gate_inf.cu's shared memory for v tiles of VB keys: the ring of SLOT-byte slots, three g tiles, the barriers."""
    slot = max(vb * 128, 16384)
    return (_SMEM_MAX - 3 * 16384 - 1024) // slot * slot + 3 * 16384 + 512


def _pv_vb(L: int, S: int) -> int:
    """Keys per v tile: 128 for the inference shapes (few samples; tuned there), else 192 or 256 when one divides L (A = 48,
    gated / ungated us: L384 26.0 / 23.6 against 27.3 / 25.7 at 128; L768 56.2 / 52.2, 58.6 / 54.8 at 256, 63.9 / 57.9 at 128)."""
    forced = os.environ.get("MINIWORLD_BIAS_ONLY_DIT_PV_VB")
    if forced:
        return int(forced)
    if S < 16:
        return 128
    return next((n for n in (192, 256) if L % n == 0), 128)


#: samples per work item the core is built for (a cubin each)
_GROUPS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 48)


def pick_group(S: int, L: int, nsm: int, nh: int = H, dh: int = DH) -> int:
    """Samples per work item: the fewest bytes into the busiest SM, rounds x 2 L (128 + 48 SG) (the P tile once, SG samples'
    v tiles); the epilogue hides under the next sample's products, so bytes are the cost."""
    forced = os.environ.get("MINIWORLD_BIAS_ONLY_DIT_SG")
    if forced:
        return max(1, min(int(forced), S))
    best = None
    for sg in (g for g in _GROUPS if g <= S):
        items = nh * (L // 128) * -(-S // sg)
        cost = -(-items // nsm) * 2 * L * (128 + dh * sg)
        if best is None or cost < best[0]:
            best = (cost, sg)
    return best[1]


def core_supported(dtype: torch.dtype, L: int, d: int, h: int, device_index: int) -> bool:
    """bf16, d attention channels as h heads in LAYOUTS (768 = 16 x 48, 24 x 32, 12 x 64; 1024 = 16 x 64), L a multiple of 128
    up to 768 (the P tile and two accumulators in 512 TMEM columns), sm_100. MINIWORLD_BIAS_ONLY_DIT_CORE=0 turns it off.
    fp32: the TF32 core (``tf32.py``; same layouts and lengths), once the fp32 path's kernels load (``tf32_ready``)."""
    if os.environ.get("MINIWORLD_BIAS_ONLY_DIT_CORE", "1") == "0":
        return False
    if dtype is torch.float32:
        if not (d % h == 0 and (h, d // h) in LAYOUTS and L % 128 == 0 and 0 < L <= 768
                and torch.cuda.get_device_capability(device_index) == (10, 0)):
            return False
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import tf32_ready
        return tf32_ready(device_index, h, d // h, False)
    return (dtype is torch.bfloat16 and d % h == 0 and (h, d // h) in LAYOUTS and L % 128 == 0 and 0 < L <= 768
            and torch.cuda.get_device_capability(device_index) == (10, 0))


def _descriptors(*maps):
    """TMA descriptors that do not keep their tensors alive: a bound launch is cached by the buffers' addresses and layouts
    (the descriptor encodes nothing else), so a freed buffer is not pinned by the cache -- the training step allocates its
    activations afresh every call, and holding them leaked a step's worth of buffers per call."""
    for m in maps:
        m.keep = None
    return maps


PTR = object()        # a _Launch argument given per call (a tensor's address)


class _Launch:
    """One kernel launch with its argument block built once (the shared driver rebuilds ctypes arguments and the launch
    configuration on every call, ~25-35 us of host time per launch in the training step): TMA descriptors and scalars are
    fixed at bind time; PTR slots take a tensor's address per call. Launch parameters are copied by the driver at launch,
    so reusing the block is safe under CUDA graph capture too."""

    def __init__(self, kernel, grid, block, *args):
        import ctypes
        from cuda.bindings import driver as cu
        from miniworld_engine.kernels.augmented_attention.cuda.sm100.driver import TensorMap
        self._cu, self.k, self.maps = cu, kernel, [a for a in args if isinstance(a, TensorMap)]
        self.hold, self.dyn, ptrs = [], [], []
        for a in args:
            if isinstance(a, TensorMap):
                ptrs.append(a.addr)
                continue
            h = ctypes.c_uint64(0) if a is PTR else ctypes.c_int32(int(a))
            (self.dyn if a is PTR else self.hold).append(h)
            ptrs.append(ctypes.addressof(h))
        self.arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        self.argp = ctypes.addressof(self.arr)
        self.grid, self.block = grid, block
        self.cfg = None
        if kernel.cluster:
            cfg = cu.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block
            cfg.sharedMemBytes = kernel.smem
            at = cu.CUlaunchAttribute()
            at.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = kernel.cluster, 1, 1
            cfg.attrs = [at]; cfg.numAttrs = 1
            self.cfg = cfg

    def __call__(self, *tensors):
        cu = self._cu
        for h, t in zip(self.dyn, tensors, strict=True):
            h.value = t.data_ptr()
        st = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        if self.cfg is not None:
            self.cfg.hStream = st
            err, = cu.cuLaunchKernelEx(self.cfg, self.k.func, self.argp, 0)
        else:
            g, b = self.grid, self.block
            err, = cu.cuLaunchKernel(self.k.func, g[0], g[1], g[2], b[0], b[1], b[2], self.k.smem, st, self.argp, 0)
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel: {err}")


class _Runs(dict):
    """Bound launches keyed by (addresses, layouts); the caching allocator hands out a few buffer sets, so a bounded cache."""

    def bind(self, key, make):
        run = self.get(key)
        if run is None:
            if len(self) >= 64:
                self.clear()
            run = self[key] = make()
        return run


class PvGateCore:
    """``pv_gate_inf.cu``: out [S L, 768] = sigmoid(g) * (P v) -- or, without g, P v -- per head and sample; P one block's
    [16 L, L] attention weights (bf16, contiguous), v / g / out [S L, 768] bf16 views with any row stride (v | g are the
    column halves of the v|g GEMM output in the step; the training backward passes P^T and do). Bound launches (their TMA
    descriptors carry the pointers) are cached per (buffers, strides, S, L)."""

    def __init__(self, device_index: int, defs: tuple[str, ...] = (), nh: int = H, dh: int | None = None):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        self._sm100 = sm100
        self._tm = sm100._tm
        self.nh, self.dh = nh, dh or 768 // nh
        self.device_index, self.defs = device_index, defs + _head_defs(nh, self.dh)
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def kernel(self, sg: int, gate: bool = True, vb: int = 128):
        """The cubin for SG samples per item, with or without the gate, VB keys per v tile (built on first use)."""
        vdef = () if vb == 128 else (f"VB={vb}",)
        return self._sm100._sm100_kernel("pv_gate_inf", "bo_pv_gate_inf_sm100", self.device_index, src_dir=str(_dir),
                                         defs=(f"SG={sg}", f"GATE={int(gate)}", *vdef, *self.defs), smem=_pv_smem(vb))

    def _bind(self, v, P, out, S, L, g):
        M = v.shape[0]
        tm, es = self._tm, v.element_size()
        vb, nh, dh = _pv_vb(L, S), self.nh, self.dh
        maps = _descriptors(tm(P, [L, nh * L], L * 2, [64, 128]), tm(v, [nh * dh, M], v.stride(0) * es, [dh, vb]),
                            tm(v if g is None else g, [nh * dh, M], (v if g is None else g).stride(0) * es, [dh, 128]),
                            tm(out, [nh * dh, M], out.stride(0) * es, [dh, 128]))
        sg = pick_group(S, L, self.nsm, nh, dh)
        k = self.kernel(sg, g is not None, vb)
        grid = (min(self.nsm, nh * (L // 128) * -(-S // sg)), 1, 1)

        return _Launch(k, grid, (256, 1, 1), *maps, L, S)

    def __call__(self, v, P, out, S, g=None):
        """v [S L, 768] view, P [nh L, L] contiguous, out [S L, 768] view, g [S L, 768] view or None; all bf16."""
        L = v.shape[0] // S
        key = (v.data_ptr(), v.stride(0), None if g is None else (g.data_ptr(), g.stride(0)), P.data_ptr(), out.data_ptr(),
               out.stride(0), S, L)
        self.runs.bind(key, lambda: self._bind(v, P, out, S, L, g))()
        return out


# --------------------------------------------------------------------------------------------------- training backward
def gate_bwd_rows(da, ao, g, do, dg, dd, L):
    """do = da sigmoid(g), dg = da ao (1 - sigmoid(g)) (ao = sigmoid(g) o, the forward's gated output), dd [A, nh, L] =
    sum over each head's W / nh channels of da ao; da, ao, do [M, W] contiguous, g / dg [M, W] views (W = 768 or 1024)."""
    _ext().gate_bwd_rows_cuda(da, ao, g, do, dg, dd, int(L))


def transpose_hll(x, out):
    """out[h] = x[h]^T for x [H, L, L] bf16."""
    _ext().transpose_hll_cuda(x, out)


def _dpb_nj(L: int, nsm: int = 148, ng: int = H, hp: int = 1) -> int:
    """The key tile of dpb_sm100.cu for ng head groups of hp heads (hp NJ TMEM columns <= 512). An item's K loop over the
    samples is bound by its SM's TMA intake (the query tile and the key tile, 128 + NJ rows per sample), so the cost is the
    rounds of items over the SMs times that: ceil(ng (L / 128) (L / NJ) / nsm) (128 + NJ). A = 48, us: 16 x 48 L384 NJ128 23
    (NJ384 44), L512 NJ256 34 (NJ128 39), L640 NJ128 56, L768 NJ256 65 (NJ192 67, NJ128 71); 24 x 32 (head pairs) L384 NJ128 29
    (NJ192 32), L768 NJ192 59 (NJ256 73, NJ128 70)."""
    best = None
    for nj in (128, 192, 256, 320, 384):
        if L % nj or hp * nj > 512:
            continue
        cost = -(-ng * (L // 128) * (L // nj) // nsm) * (128 + nj)
        if best is None or cost < best[0]:
            best = (cost, nj)
    return best[1]


def _dpb_smem(nj: int) -> int:
    o_st = 2 * 8192 + 4 * 3 * 2048
    stage = 128 * 128 + nj * 128
    return o_st + (_SMEM_MAX - o_st - 1024) // stage * stage + 256


class DpbKernel:
    """``dpb_sm100.cu``: dbias [nh L, L] = P o (sum_a do v^T - D) per head, D[h, i] = sum_a dd[a, h, i]; do / v [A L, 768]
    bf16 views, P [nh L, L] bf16, dd [A, nh, L] fp32 (nh x width: one of LAYOUTS)."""

    def __init__(self, device_index: int, nh: int = H, dh: int | None = None):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        self._sm100, self._tm, self.device_index = sm100, sm100._tm, device_index
        self.nh, self.dh = nh, dh or 768 // nh
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def _bind(self, do, v, P, dd, out, A, L):
        nh, dh = self.nh, self.dh
        hp = 2 if dh == 32 else 1                     # heads per work item (dpb_sm100.cu HP): two 32-wide heads per 128-byte row
        nj = _dpb_nj(L, self.nsm, nh // hp, hp)
        k = self._sm100._sm100_kernel("dpb_sm100", "bo_dpb_sm100", self.device_index, src_dir=str(_dir),
                                      defs=(f"NJ={nj}", *_head_defs(nh, dh)), smem=_dpb_smem(nj))
        tm, M = self._tm, do.shape[0]
        maps = _descriptors(tm(do, [nh * dh, M], do.stride(0) * 2, [hp * dh, 128]),
                            tm(v, [nh * dh, M], v.stride(0) * 2, [hp * dh, nj // 2]),
                            tm(P, [L, nh * L], L * 2, [32, 128], swizzle=64), tm(out, [L, nh * L], L * 2, [32, 32], swizzle=64))
        grid = (min(self.nsm, nh // hp * (L // 128) * (L // nj)), 1, 1)

        return _Launch(k, grid, (256, 1, 1), *maps, PTR, L, A)

    def __call__(self, do, v, P, dd, out, A):
        L = do.shape[0] // A
        key = (do.data_ptr(), do.stride(0), v.data_ptr(), v.stride(0), P.data_ptr(), out.data_ptr(), A, L)
        self.runs.bind(key, lambda: self._bind(do, v, P, dd, out, A, L))(dd)
        return out


# ------------------------------------------------------------------------------------------------ expand GEMM + SwiGLU (training)
class GemmSwigluAB:
    """The token DiT's ``gemm_swiglu2_sm100.cu`` built with SAVE_AB: h = silu(a) b and [a | b] = X [Wa; Wb]^T (bf16) from one
    kernel, the SwiGLU in the GEMM epilogue, so the training forward does not read [a | b] back for h (A = 48: L384 82 us
    against cuBLAS 59 + SwiGLU 29, L768 152 against 170). X [M, 768], W [4 x 768, 768], ab [M, 4 x 768], h [M, 2 x 768]
    contiguous; bound launches (their TMA descriptors) are cached per buffer set."""

    SMEM = 4 * 32768 + 32768 + 4 * 16384 + 512

    def __init__(self, device_index: int, K: int = 768):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu
        self._tm, self.K, self.H = sm100._tm, K, 2 * K
        self.k = sm100._sm100_kernel("gemm_swiglu2_sm100", "gemm_swiglu2_sm100", device_index, src_dir=gemm_swiglu._dir,
                                     defs=(f"DIM={K}", "HMUL=2", "ITEM_SCHED", "SAVE_AB"), cluster=2, smem=self.SMEM)
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def _bind(self, x, w, ab, h):
        M, K, H = x.shape[0], self.K, self.H
        tm = lambda t, rows, cols, box: self._tm(t, [cols, rows], t.stride(0) * 2, box)
        maps = _descriptors(tm(x, M, K, [64, 64]), tm(w[:H], H, K, [64, 128]), tm(w[H:], H, K, [64, 128]), tm(h, M, H, [64, 64]),
                            tm(ab[:, :H], M, H, [64, 64]), tm(ab[:, H:], M, H, [64, 64]))
        grid, tiles = (self.nsm & ~1, 1, 1), (M + 127) // 128

        return _Launch(self.k, grid, (512, 1, 1), *maps, 1, tiles)

    def __call__(self, x, w, ab, h):
        M, K, H = x.shape[0], self.K, self.H
        assert x.shape == (M, K) and w.shape == (2 * H, K) and ab.shape == (M, 2 * H) and h.shape == (M, H)
        assert all(t.is_contiguous() and t.dtype is torch.bfloat16 for t in (x, w, ab, h))
        self.runs.bind((x.data_ptr(), w.data_ptr(), ab.data_ptr(), h.data_ptr(), M), lambda: self._bind(x, w, ab, h))()
        return h


# --------------------------------------------------------------------------------------------------- GEMM + residual + AdaLN
def _resln_smem(cl: int = 8) -> int:
    """gemm_resln_sm100.cu's shared memory: the K ring fills what the epilogue slots, the exchange area and the barriers leave."""
    return _SMEM_MAX - 1024 + 256                              # O_BAR + 256


class ResLnGemm:
    """``gemm_resln_sm100.cu``: x += sigmoid(gl) * (A W^T) (fp32, in place), then xa = AdaLN(x) (bf16) -- or, ``final``,
    out = x in bf16 -- with W [768, K]. A cluster of CL CTAs per 128-row tile (8 while the row tiles x 8 fit the SMs, else 4),
    the LayerNorm statistics exchanged once through distributed shared memory. Bound launches are cached per buffer set."""

    def __init__(self, device_index: int, final: bool = False, defs: tuple[str, ...] = (), presig: bool = False):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        self._sm100, self._tm, self.final = sm100, sm100._tm, final
        self.defs = (*defs, *(("PRESIG",) if presig else ()))
        self.device_index = device_index
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def kernel(self, cl: int):
        defs = (f"CL={cl}", *(("FINAL",) if self.final else ()), *self.defs)
        return self._sm100._sm100_kernel("gemm_resln_sm100", "bo_gemm_resln_sm100", self.device_index, src_dir=str(_dir),
                                         defs=defs, cluster=cl, smem=_resln_smem(cl))

    def cluster(self, M: int) -> int:
        return 8 if (M // 128) * 8 <= self.nsm else 4

    def _bind(self, a, w, x, gl, ms, mb, out, T, eps):
        M, K = a.shape
        cl = self.cluster(M)
        nc = 768 // cl
        tm = self._tm
        tab = lambda t: tm(t, [768, t.shape[0]], t.stride(0) * 2, [32, 128], swizzle=64)
        maps = (tm(a, [K, M], K * 2, [64, 128]), tm(w, [K, 768], K * 2, [64, nc]),
                tm(x, [768, M], 768 * 4, [32, 128], swizzle=128, dtype="f32"), tab(gl),
                tab(gl if ms is None else ms), tab(gl if mb is None else mb),
                tm(out, [768, M], 768 * 2, [32, 128], swizzle=64),
                tm(x, [768, M], 768 * 4, [32, 32], swizzle=128, dtype="f32"), tm(out, [768, M], 768 * 2, [32, 32], swizzle=64))
        grid, k = (cl * (M // 128), 1, 1), self.kernel(cl)

        def run():
            k(grid, (256, 1, 1), *maps, int(K), int(T), float(eps))
        run.keep = (maps, a, w, x, gl, ms, mb, out)
        return run

    def __call__(self, a, w, x, gl, ms, mb, out, T, eps=1e-5):
        """a [M, K], w [768, K] bf16; x [M, 768] fp32 (updated); gl / ms / mb table views [T, 768] (ms, mb unused when final);
        out [M, 768] bf16: xa, or with ``final`` the block's output."""
        key = (a.data_ptr(), w.data_ptr(), x.data_ptr(), gl.data_ptr(), None if ms is None else ms.data_ptr(),
               None if mb is None else mb.data_ptr(), out.data_ptr(), a.shape, T)
        run = self.runs.get(key)
        if run is None:
            run = self.runs[key] = self._bind(a, w, x, gl, ms, mb, out, T, eps)
        run()
        return out


# --------------------------------------------------------------------------------------------------- conditioning tables
_COND_NT = 256


class CondTables:
    """``cond_tables_sm100.cu``: every block's conditioning tables in one kernel -- G [T, N] = sig(rstd (c W^T - mu colsum) + b)
    on the first ``n_g1`` columns (AdaLN scale / shift, LN(c) folded in algebraically), sig(c W^T + b) on the rest (the output
    gates); sig = sigmoid on the scale and gate columns. c [T, 384] bf16, W [N, 384] bf16, b and colsum [N] fp32."""

    SMEM = 6 * 16384 + 2 * _COND_NT * 128 + 8 * 3 * 2048 + 2 * 2 * _COND_NT * 4 + 2 * 2 * 128 * 4 + 256

    def __init__(self, device_index: int):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        self._tm = sm100._tm
        self.k = sm100._sm100_kernel("cond_tables_sm100", "bo_cond_tables_sm100", device_index, src_dir=str(_dir),
                                     defs=(f"NT={_COND_NT}",), smem=self.SMEM)
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def _bind(self, c, w, bias, colsum, out, n_g1, eps):
        T, N = c.shape[0], w.shape[0]
        tm = self._tm
        maps = (tm(c, [384, T], c.stride(0) * 2, [64, 128]), tm(w, [384, N], 384 * 2, [64, _COND_NT]),
                tm(out, [N, T], N * 2, [32, 32], swizzle=64))
        items = (T // 128) * (N // _COND_NT)
        grid = (min(self.nsm, items), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, bias, colsum, int(T), int(N), int(n_g1), float(eps))
        run.keep = (maps, c, w, bias, colsum, out)
        return run

    def __call__(self, c, w, bias, colsum, out, n_g1, eps=1e-5):
        key = (c.data_ptr(), c.shape, w.data_ptr(), out.data_ptr())
        run = self.runs.get(key)
        if run is None:
            run = self.runs[key] = self._bind(c, w, bias, colsum, out, n_g1, eps)
        run()
        return out


__all__ = ["CondTables", "DpbKernel", "GemmSwigluAB", "PvGateCore", "ResLnGemm", "gate_bwd_rows", "transpose_hll", "adaln_in_rows", "core_supported", "ln_rows", "pick_group", "resgate_adaln_rows", "resgate_out_rows", "softmax_t",
           "softmax_rows", "swiglu_rows"]

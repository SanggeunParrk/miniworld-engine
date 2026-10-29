"""CUDA kernels of the bias-only token DiT (B200, sm_100a), inference.

  pv_gate_inf.cu          the attention core, sigmoid(g) * (P v) per head and sample, P = softmax(pair bias) made once per
                          sample() -- a tcgen05 GEMM with the gate in its epilogue (``PvGateCore``); L <= 768 (P in tensor memory)
  pv_stream_sm100.cu      the same core past L 768 (to 3072, bf16): P streamed through shared memory, key-outer / sample-inner
                          (``PvStreamCore``; ``PvGateCore`` hands such calls over)
  bias_only_dit_rows.cu   every row pass between the GEMMs (one warp per row): the conditioning LayerNorm, the input AdaLN
                          with the fp32 residual copy, residual + gate + AdaLN, the last residual written in the output dtype,
                          SwiGLU, and the softmax of the hoisted bias

The core is a cubin built on first use into ``settings.engine_jit_root`` and launched through the sm_100a driver of
``kernels/augmented_attention/cuda/sm100`` (TMA descriptors, current stream, CUDA-graph capturable); the rows are a
torch extension. Neither is built at import.

Training: ``dpb_sm100.cu`` and ``dpbx2_sm100.cu`` (the bias gradient; the second on CTA pairs, the default where 256-key tiles divide
L: ``DpbKernel``), the token DiT's ``gemm_swiglu2_sm100`` with SAVE_AB (the expand GEMM + SwiGLU).

The fp32 path (TF32 tensor cores: ``pv_gate_tf32.cu``, ``dpb32_sm100.cu``, ``gemm_glu_tf32.cu``, ``bias_only_dit_f32_rows.cu``) is
``tf32.py``.
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path

import torch

from miniworld_engine.kernels._device import num_sms
from miniworld_engine import settings
from miniworld_engine.kernels._compile import device_constant

_dir = Path(__file__).parent
H, DH = 16, 48  # the default heads x width
#: (heads, head width) served: 768 attention channels as 16 x 48, 24 x 32 or 12 x 64, and 1024 as 16 x 64
LAYOUTS = ((16, 48), (24, 32), (12, 64), (16, 64))


#: Hopper PV core: True = ``pvx`` (all samples of a group in one pass over the keys, 3-stage TMA ring, two consumer
#: warpgroups), False = the earlier ``pv_grouped`` (P tile resident, one consumer warpgroup). Kept for A/B.
_H100_PVX = True
_H100_DPB2 = True  # the 128 x TN bias-gradient kernel (False: the 64 x 64 ``dpb_pipe``)
_H100_PVX_FORCE = (0, 0, 0)  # (consumer warpgroups, samples per CTA, cluster); 0 = the host cost model
#: Longest L the Hopper inference path serves (``pvx`` / ``pvx32`` stream P; the softmax rows take up to 3072 keys).
H100_MAX_L = 3072
#: B200: the longest L whose P tile fits tensor memory beside two accumulators (``pv_gate_inf.cu``, and every fp32 kernel's bound);
#: past it the bf16 core streams P through shared memory (``pv_stream_sm100.cu``) up to ``B200_MAX_L`` (MiniWorld's token rungs).
B200_TMEM_MAX_L = 768
B200_MAX_L = 3072


def b200_max_l(dtype: torch.dtype) -> int:
    """Longest L the sm_100a path serves for ``dtype``: bf16 up to ``B200_MAX_L`` (the streaming core past ``B200_TMEM_MAX_L``,
    settings.bias_only_dit_stream=0 keeps the old bound), fp32 up to ``B200_TMEM_MAX_L``."""
    if dtype is torch.bfloat16 and settings.current().bias_only_dit_stream:
        return B200_MAX_L
    return B200_TMEM_MAX_L


def sm100_kernel(*args, **kw):
    """``augmented_attention.cuda.sm100._sm100_kernel`` for this family's sm_100a cubins, built by the newest nvcc here that knows
    sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``: 13.1 on the B200 box) rather than the torch-matched 12.9."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    return sm100._sm100_kernel(*args, newest=True, **kw)


def h100_enabled(device) -> bool:
    """Native Hopper kernels; per-shape speed qualification is recorded separately."""
    return torch.cuda.get_device_capability(device) == (9, 0)


def _h100_group(samples, length, heads, width, sms):
    """P bytes are reused across the group; at most two resident CTAs are scheduled."""
    residents = 2 if length * 128 + 3 * 8192 + 1024 <= 232448 // 2 else 1
    return min(
        (g for g in _GROUPS if g <= samples),
        key=lambda g: (
            -(-heads * (length // 64) * -(-samples // g) // (sms * residents))
            * length
            * (64 + width * g)
        ),
    )


@functools.lru_cache(maxsize=None)
def h100_ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    k = _dir.parent.parent
    return load_extension(
        name="bias_only_dit_sm90a",
        sources=[str(_dir / "pv_gate_inf.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            *gencodes("90a"),
            "--expt-relaxed-constexpr",
            "-DMINIWORLD_BO_SM90=1",
            f"-I{k / 'transition' / 'cuda' / 'anthropic_v5'}",
            f"-I{k / 'transition' / 'cuda' / 'wide' / 'kernels'}",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def psb_ext():
    """``bo_psb.cu``: the attention core on a pair bias of its own per sample (single-pass mma.sync forward / backward). Built on first use."""
    from miniworld_engine.kernels._nvcc import (
        ensure_cuda_home,
        gencodes,
        host_flags,
        load_extension,
    )

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_psb_sm90a",
        sources=[str(_dir / "bo_psb.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            *gencodes("90a"),
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def psb32_ext():
    """``bo_psb32.cu``: the fp32 (TF32 tensor cores) attention core on a pair bias of its own per sample, recompute form. Built on first use."""
    from miniworld_engine.kernels._nvcc import (
        ensure_cuda_home,
        gencodes,
        host_flags,
        load_extension,
    )

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_psb32_sm90a",
        sources=[str(_dir / "bo_psb32.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


#: Longest L of the fp32 per-sample forward (v of the (head, sample) stays resident in shared memory)
PSB32_MAX_L = 512


def psb32_fwd(bias, stats, mask, vg, og, samples, heads, dh):
    """fp32: og = sigmoid(g) (P v) with P = softmax(bias + key mask) on TF32 tensor cores; ``stats`` [H A L, 2] fp32 get (mc, 1 / sum), the
    backward rebuilds P from them. bias [H A L, L]; vg [A L, 2 H dh]; og [A L, H dh]; mask bool [L] or [A, L] or None."""
    psb32_ext().psb32_fwd(bias, stats, mask, vg, og, int(samples), int(heads), int(dh))


def psb32_bwd(bias, stats, mask, dout, vg, dd, dbias, dvg, samples, heads, dh):
    """fp32: dV = P^T dO into the v columns of ``dvg`` and dbias = P o (dO v^T - D) into ``dbias`` [H A L, L], P rebuilt from the bias,
    the forward's statistics and the key mask; dd [A, H, L]."""
    psb32_ext().psb32_bwd(bias, stats, mask, dout, vg, dd, dbias, dvg, int(samples), int(heads), int(dh))


def psb_fwd(bias, p, mask, vg, og, samples, heads, dh, stats=None):
    """og = sigmoid(g) (P v), P = softmax(bias + key mask). bias [H A L, L] bf16 head-major; vg [A L, 2 H dh] = v | g; og [A L, H dh]; mask
    bool [L] or [A, L] or None. One pass, bf16. Either P is written to ``p`` (``stats`` None), or ``p`` is None and the row statistics
    (mc, 1 / sum) go to ``stats`` [H A L, 2] fp32 -- the backward then rebuilds P from the bias (``psb_bwd(..., stats=...)``)."""
    psb_ext().psb_fwd(bias, p, stats, mask, vg, og, int(samples), int(heads), int(dh))


def psb_bwd(p, dout, vg, dd, dbias, dvg, samples, heads, dh, stats=None, mask=None):
    """dV = P^T dO into the v columns of ``dvg`` and dbias = P o (dO v^T - D) into ``dbias`` [H A L, L]; dd [A, H, L] fp32. One pass.
    ``stats`` None: ``p`` is P. Else ``p`` is the bias and P is rebuilt from the forward's row statistics (and the key mask)."""
    psb_ext().psb_bwd(p, stats, mask, dout, vg, dd, dbias, dvg, int(samples), int(heads), int(dh))


@functools.lru_cache(maxsize=None)
def h100_gemm_ext():
    """``bo_gemm_sm90.cu``: Hopper wgmma GEMMs with a row pass in the epilogue -- the one-launch conditioning tables (dispatched), and
    the squeeze + last residual and out-projection gradient + gate backward fusions (measured slower; probes). Built on first use."""
    from miniworld_engine.kernels._nvcc import (
        ensure_cuda_home,
        gencodes,
        host_flags,
        load_extension,
    )

    ensure_cuda_home()
    k = _dir.parent.parent
    return load_extension(
        name="bias_only_dit_gemm_sm90a",
        sources=[str(_dir / "bo_gemm_sm90.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            *gencodes("90a"),
            "--expt-relaxed-constexpr",
            f"-I{k / 'transition' / 'cuda' / 'anthropic_v5'}",
            f"-I{k / 'transition' / 'cuda' / 'wide' / 'kernels'}",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def _head_defs(nh: int, dh: int) -> tuple[str, ...]:
    """Compile definitions of the sm_100a cubins for a head layout (none for the default, so its cubins keep their keys)."""
    assert (nh, dh) in LAYOUTS, (nh, dh)
    return () if (nh, dh) == (H, DH) else (f"NHEAD={nh}", f"DHEAD={dh}")


@functools.lru_cache(maxsize=None)
def _ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    hopper = h100_enabled(torch.cuda.current_device())
    return load_extension(
        name="bias_only_dit_rows_cuda" + ("_sm90a" if hopper else ""),
        sources=[str(_dir / "bias_only_dit_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *(gencodes("90a") if hopper else gencodes("80", "90", "100", ptx=("100",))),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def ln_rows(x, out, eps=1e-5):
    """out = LN(x), no affine; x [M, 384]."""
    _ext().ln_rows_cuda(x, out, float(eps))


def adaln_in_rows(single, x, ms, mb, out, T, eps=1e-5, presig=False):
    """x = single (fp32; x empty: no copy), out = LN(single) * sigmoid(ms[row % T]) + mb[row % T]; width 768. ``presig``: ms holds
    the sigmoid."""
    _ext().adaln_in_rows_cuda(single, x, ms, mb, out, int(T), float(eps), bool(presig))


def resgate_adaln_rows(x, y, gl, ms, mb, out, T, eps=1e-5, presig=False):
    """x += sigmoid(gl[row % T]) * y in place (fp32); then, when ms is given, out = AdaLN(x) for the next half-block."""
    _ext().resgate_adaln_rows_cuda(
        x, y, gl, ms, mb, out, int(T), float(eps), bool(presig)
    )


def resgate_adaln_rows_in(xin, x, y, gl, ms, mb, out, T, eps=1e-5, presig=False):
    """x = xin + sigmoid(gl[row % T]) * y (fp32; xin bf16 or fp32: the step input when ``adaln_in_rows`` made no fp32 copy), out =
    AdaLN(x)."""
    _ext().resgate_adaln_rows_in_cuda(
        xin, x, y, gl, ms, mb, out, int(T), float(eps), bool(presig)
    )


def swiglu_rows(ab, out):
    """out = silu(a) * b for ab = [a | b] (bf16)."""
    _ext().swiglu_rows_cuda(ab, out)


def resgate_out_rows(x, y, gl, out, T, presig=False):
    """out = x + sigmoid(gl[row % T]) * y in out's dtype; width 768."""
    _ext().resgate_out_rows_cuda(x, y, gl, out, int(T), bool(presig))


def softmax_rows(bias, p, mask=None):
    """p = softmax over the last axis of bias [R, L] bf16 (p may be bias); ``mask`` [L] bool drops the False keys."""
    _ext().softmax_rows_cuda(bias, p, mask)


def softmax_t(bias, p, pt=None, mask=None, samples=1, scale=1.0):
    """softmax_rows that also writes pt[h] = p[h]^T (bias, p, pt [16 L, L] bf16; p may be bias, pt may not). bias may also be fp32
    (the Hopper training logits, not overwritten: p is a separate bf16 buffer). ``pt`` None: P only.

    Per-sample bias (the per-augmentation pair bias, head-major [H, A, L, L] as a [H A L, L] matrix): ``samples`` = A, and a bool
    ``mask`` [A, L] is one key mask per sample ([L]: one for all). ``scale`` multiplies the logits first (ln 2 for logits in log2
    units: P = softmax(scale bias))."""
    _ext().softmax_t_cuda(bias, p, pt, mask, int(samples), float(scale))


# --------------------------------------------------------------------------------------------------- the core
_SMEM_MAX = 232448


#: pv_gate_inf.cu's layout: as many 16 KB ring slots as fit beside three 16 KB g staging tiles, 512 B of barriers
_SMEM = (_SMEM_MAX - 3 * 16384 - 1024) // 16384 * 16384 + 3 * 16384 + 512


def _pv_smem(vb: int) -> int:
    """pv_gate_inf.cu's shared memory for v tiles of VB keys: the ring of SLOT-byte slots, three g tiles, the barriers."""
    slot = max(vb * 128, 16384)
    return (_SMEM_MAX - 3 * 16384 - 1024) // slot * slot + 3 * 16384 + 512


def _pv_vb(L: int, S: int, dh: int = DH) -> int:
    """Keys per v tile: 128 for the inference shapes (few samples; tuned there); 256 at L768 with d_head 48 and S >= 16 (A = 48,
    16 x 48, in-process A/B of VB 192 / 256, us: forward 65.21 / 58.28, dV 55.15 / 49.94 -- fewer v tiles, each costing ~90 ns
    beside its products); else 192 or 256 when one divides L (A = 48, gated / ungated us: L384 26.0 / 23.6 against 27.3 / 25.7 at
    128). settings.bias_only_dit_pv_vb forces one (read per call)."""
    forced = settings.current().bias_only_dit_pv_vb
    if forced:
        return int(forced)
    if S < 16:
        return 128
    if L == 768 and dh == 48:
        return 256
    return next((n for n in (192, 256) if L % n == 0), 128)


#: samples per work item the core is built for (a cubin each)
_GROUPS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 48)


def pick_group(S: int, L: int, nsm: int, nh: int = H, dh: int = DH) -> int:
    """Samples per work item: the fewest bytes into the busiest SM, rounds x 2 L (128 + 48 SG) (the P tile once, SG samples'
    v tiles); the epilogue hides under the next sample's products, so bytes are the cost."""
    forced = settings.current().bias_only_dit_sg
    if forced:
        return max(1, min(int(forced), S))
    best = None
    for sg in (g for g in _GROUPS if g <= S):
        items = nh * (L // 128) * -(-S // sg)
        cost = -(-items // nsm) * 2 * L * (128 + dh * sg)
        if best is None or cost < best[0]:
            best = (cost, sg)
    return best[1]


def core_supported(
    dtype: torch.dtype, L: int, d: int, h: int, device_index: int
) -> bool:
    """bf16, d attention channels as h heads in LAYOUTS (768 = 16 x 48, 24 x 32, 12 x 64; 1024 = 16 x 64), L a multiple of 128
    up to ``b200_max_l`` on sm_100 (``pv_gate_inf.cu`` with the P tile in tensor memory up to 768, ``pv_stream_sm100.cu`` streaming
    P beyond, to 3072), up to ``H100_MAX_L`` on Hopper (``pvx`` / ``pvx32`` stream P through a TMA ring: no length bound of their
    own; the sampler's token rungs reach 2560). settings.bias_only_dit_core=0 turns it off.
    fp32: the TF32 core (``tf32.py``; same layouts, L up to 768), once the fp32 path's kernels load (``tf32_ready``)."""
    if not settings.current().bias_only_dit_core:
        return False
    if h100_enabled(device_index):  # Hopper: ``pvx`` (bf16) / the TF32 wgmma core ``pvx32`` (fp32), head widths 32 / 48 / 64
        return (
            dtype in (torch.bfloat16, torch.float32)
            and d % h == 0
            and (h, d // h) in LAYOUTS
            and L % 128 == 0
            and 0 < L <= H100_MAX_L
        )
    if dtype is torch.float32:
        if not (
            d % h == 0
            and (h, d // h) in LAYOUTS
            and L % 128 == 0
            and 0 < L <= B200_TMEM_MAX_L
            and torch.cuda.get_device_capability(device_index) == (10, 0)
        ):
            return False
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import tf32_ready

        return tf32_ready(device_index, h, d // h, False)
    if not (
        dtype is torch.bfloat16
        and d % h == 0
        and (h, d // h) in LAYOUTS
        and L % 128 == 0
        and 0 < L <= b200_max_l(dtype)
        and torch.cuda.get_device_capability(device_index) == (10, 0)
    ):
        return False
    return L <= B200_TMEM_MAX_L or stream_ready(device_index, h, d // h)


def _descriptors(*maps):
    """TMA descriptors that do not keep their tensors alive: a bound launch is cached by the buffers' addresses and layouts
    (the descriptor encodes nothing else), so a freed buffer is not pinned by the cache -- the training step allocates its
    activations afresh every call, and holding them leaked a step's worth of buffers per call."""
    for m in maps:
        m.keep = None
    return maps


PTR = object()  # a _Launch argument given per call (a tensor's address)


class _Launch:
    """One kernel launch with its argument block built once (the shared driver rebuilds ctypes arguments and the launch
    configuration on every call, ~25-35 us of host time per launch in the training step): TMA descriptors and scalars are
    fixed at bind time; PTR slots take a tensor's address per call. Launch parameters are copied by the driver at launch,
    so reusing the block is safe under CUDA graph capture too."""

    def __init__(self, kernel, grid, block, *args):
        import ctypes
        from cuda.bindings import driver as cu
        from miniworld_engine.kernels.augmented_attention.cuda.sm100.driver import (
            TensorMap,
        )

        self._cu, self.k, self.maps = (
            cu,
            kernel,
            [a for a in args if isinstance(a, TensorMap)],
        )
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
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = (
                kernel.cluster,
                1,
                1,
            )
            cfg.attrs = [at]
            cfg.numAttrs = 1
            self.cfg = cfg

    def __call__(self, *tensors):
        cu = self._cu
        for h, t in zip(self.dyn, tensors, strict=True):
            h.value = t.data_ptr()
        st = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        if self.cfg is not None:
            self.cfg.hStream = st
            (err,) = cu.cuLaunchKernelEx(self.cfg, self.k.func, self.argp, 0)
        else:
            g, b = self.grid, self.block
            (err,) = cu.cuLaunchKernel(
                self.k.func,
                g[0],
                g[1],
                g[2],
                b[0],
                b[1],
                b[2],
                self.k.smem,
                st,
                self.argp,
                0,
            )
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


class PvGateCoreH100TF32:
    """Hopper fp32 attention core: out = sigmoid(g) * (P v) per head and sample on TF32 wgmma (``pvx32``). v arrives
    transposed (TF32 wgmma takes K-major operands only): ``vt`` [nh dh, S L] fp32, P [nh L, L] fp32, g / out [S L, nh dh]."""

    def __init__(self, device_index: int, nh: int = 16, dh: int = 48):
        assert dh in (32, 48, 64) and nh * dh in (768, 1024), (nh, dh)
        self.device_index, self.nh, self.dh = device_index, nh, dh
        self.nsm = num_sms(device_index)

    def __call__(self, vt, P, out, S, g):
        h100_ext().pv_cta_tf32(vt, P, out, S, g, self.nh, self.dh, self.nsm)
        return out


class PvGateCore:
    """``pv_gate_inf.cu``: out [S L, 768] = sigmoid(g) * (P v) -- or, without g, P v -- per head and sample; P one block's
    [16 L, L] attention weights (bf16, contiguous), v / g / out [S L, 768] bf16 views with any row stride (v | g are the
    column halves of the v|g GEMM output in the step; the training backward passes P^T and do). Bound launches (their TMA
    descriptors carry the pointers) are cached per (buffers, strides, S, L). ``pdl``: the bf16 three-kernel inference step's
    build (``-DPDL_INF``: programmatic dependent launch), launched with the PDL attribute (default False: the training backward and
    the 12-launch step). ``p_early`` (with ``pdl``; the runner's three-kernel inference step only): the ``-DP_EARLY`` build, which
    loads P before griddepcontrol.wait -- only for a P no kernel of the PDL chain writes (the runner's hoisted P, made once per
    sample() before the step); never for the training backward's P^T (softmax_t right before it in the same stream).
    settings.bias_only_dit_inf3_p_early=0 (read per call) takes the plain PDL build.
    ``psb``: a pair bias of its own per sample -- P is [nh, S, L, L] (a plane per (head, sample), contiguous; the ``-DPSB`` build, one
    sample per work item), L up to ``B200_TMEM_MAX_L``; no PDL."""

    def __init__(
        self,
        device_index: int,
        defs: tuple[str, ...] = (),
        nh: int = H,
        dh: int | None = None,
        pdl: bool = False,
        p_early: bool = False,
        psb: bool = False,
    ):
        assert pdl or not p_early, (
            "P_EARLY moves P before the PDL wait: a PDL build only"
        )
        assert not (psb and pdl), "the per-sample build has no PDL form"
        self.psb = psb
        from miniworld_engine.kernels.augmented_attention.cuda import sm100

        self._sm100 = sm100
        self._tm = sm100._tm
        self.nh, self.dh = nh, dh or 768 // nh
        self.device_index, self.defs, self.pdl, self.p_early = (
            device_index,
            defs + _head_defs(nh, self.dh),
            pdl,
            p_early,
        )
        self.nsm = num_sms(device_index)
        self.runs = _Runs()
        self._stream = None

    def early(self, L: int = 0) -> bool:
        """This call takes the P_EARLY build: ``p_early`` and not settings.bias_only_dit_inf3_p_early=0, and L within tensor memory
        (the streaming core past it has no P_EARLY form)."""
        return (
            self.p_early
            and settings.current().bias_only_dit_inf3_p_early
            and L <= B200_TMEM_MAX_L
        )

    def kernel(
        self, sg: int, gate: bool = True, vb: int = 128, early: bool | None = None
    ):
        """The cubin for SG samples per item, with or without the gate, VB keys per v tile, the P_EARLY build where ``early``
        (default: ``self.early()``) (built on first use)."""
        vdef = () if vb == 128 else (f"VB={vb}",)
        if self.pdl:
            edef = ("P_EARLY",) if (self.early() if early is None else early) else ()
            return sm100_kernel(
                "pv_gate_inf",
                "bo_pv_gate_inf_sm100",
                self.device_index,
                pdl=True,
                src_dir=str(_dir),
                defs=(
                    f"SG={sg}",
                    f"GATE={int(gate)}",
                    *vdef,
                    *self.defs,
                    "PDL_INF",
                    *edef,
                ),
                smem=_pv_smem(vb),
            )
        return sm100_kernel(
            "pv_gate_inf",
            "bo_pv_gate_inf_sm100",
            self.device_index,
            src_dir=str(_dir),
            defs=(f"SG={sg}", f"GATE={int(gate)}", *vdef, *self.defs, *(("PSB",) if self.psb else ())),
            smem=_pv_smem(vb),
        )

    def _bind(self, v, P, out, S, L, g, vb, early):
        M = v.shape[0]
        tm, es = self._tm, v.element_size()
        nh, dh = self.nh, self.dh
        planes = nh * S if self.psb else nh  # per sample: [nh, S, L, L]
        maps = _descriptors(
            tm(P, [L, planes * L], L * 2, [64, 128]),
            tm(v, [nh * dh, M], v.stride(0) * es, [dh, vb]),
            tm(
                v if g is None else g,
                [nh * dh, M],
                (v if g is None else g).stride(0) * es,
                [dh, 128],
            ),
            tm(out, [nh * dh, M], out.stride(0) * es, [dh, 128]),
        )
        sg = 1 if self.psb else pick_group(S, L, self.nsm, nh, dh)  # per sample: an item is one sample (its own P tile)
        k = self.kernel(sg, g is not None, vb, early)
        grid = (min(self.nsm, nh * (L // 128) * -(-S // sg)), 1, 1)
        if self.pdl:
            from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import _LaunchPDL

            return _LaunchPDL(k, grid, (256, 1, 1), *maps, L, S)
        return _Launch(k, grid, (256, 1, 1), *maps, L, S)

    def stream(self) -> "PvStreamCore":
        """The streaming core for L > ``B200_TMEM_MAX_L`` (same layout and PDL build), made on first use."""
        if self._stream is None:
            self._stream = PvStreamCore(
                self.device_index, nh=self.nh, dh=self.dh, pdl=self.pdl
            )
        return self._stream

    def __call__(self, v, P, out, S, g=None):
        """v [S L, 768] view, P [nh L, L] contiguous ([nh S L, L] with ``psb``), out [S L, 768] view, g [S L, 768] view or None; all
        bf16. L > 768 (the P tile no longer fits tensor memory) runs ``pv_stream_sm100.cu`` (``PvStreamCore``; not with ``psb``)."""
        if self.psb:
            assert not h100_enabled(self.device_index) and v.shape[0] // S <= B200_TMEM_MAX_L, "per-sample P: B200, L <= 768"
            assert P.shape[0] == self.nh * v.shape[0] and P.is_contiguous(), "per-sample P: [nh S L, L] contiguous"
        if (
            not h100_enabled(self.device_index)
            and v.shape[0] // S > B200_TMEM_MAX_L
        ):
            return self.stream()(v, P, out, S, g)
        if h100_enabled(self.device_index):
            if _H100_PVX:
                h100_ext().pv_cta(
                    v, P, out, S, g, self.nh, self.dh, self.nsm, *_H100_PVX_FORCE
                )
                return out
            sg = _h100_group(S, v.shape[0] // S, self.nh, self.dh, self.nsm)
            h100_ext().grouped_pv_pipeline(v, P, out, S, g, self.nh, self.dh, sg)
            return out
        L = v.shape[0] // S
        vb, early = (
            _pv_vb(L, S, self.dh),
            self.early(),
        )  # both read per call (the A/B switches): in the key
        key = (
            v.data_ptr(),
            v.stride(0),
            None if g is None else (g.data_ptr(), g.stride(0)),
            P.data_ptr(),
            out.data_ptr(),
            out.stride(0),
            S,
            L,
            vb,
            early,
        )
        self.runs.bind(key, lambda: self._bind(v, P, out, S, L, g, vb, early))()
        return out


# --------------------------------------------------------------------------------------------------- the core past 768 keys
#: samples per work item ``pv_stream_sm100.cu`` is built for (N = SG x 64 per product, two accumulator sets within 512 columns)
_STREAM_GROUPS = (1, 2, 3, 4)



def _stream_smem(sg: int) -> int:
    """pv_stream_sm100.cu's shared memory: NST stages of (16 KB P chunk + SG 8 KB v tiles), three 16 KB staging tiles, barriers."""
    stb = 16384 + sg * 8192
    return (_SMEM_MAX - 1024 - 3 * 16384) // stb * stb + 3 * 16384 + 512


def pick_group_stream(S: int, L: int, nsm: int, nh: int = H, dh: int = DH) -> int:
    """Samples per work item of the streaming core: the least modelled time over the SMs, rounds of items x max(the SM's TMA intake
    for the item -- the P tile once, SG v tiles -- at ~66 B/clk, the products at N = SG x 64 and ~4096 MAC/clk). Per 16-key step:
    (128 + SG DH) 32 B / 66 against SG 32 clocks. settings.bias_only_dit_sg forces one (clamped to the built groups)."""
    groups = [g for g in _STREAM_GROUPS if g <= max(S, 1)]
    forced = settings.current().bias_only_dit_sg
    if forced:
        f = int(forced)
        return max([g for g in groups if g <= f] or [1])
    best = None
    for sg in groups:
        items = nh * (L // 128) * -(-S // sg)
        per = max((128 + sg * dh) * 32 / 66, sg * 32)
        cost = -(-items // nsm) * per
        if best is None or cost < best[0] - 1e-9:
            best = (cost, sg)
    return best[1]


class PvStreamCore:
    """``pv_stream_sm100.cu``: ``PvGateCore``'s contract (out = sigmoid(g) * (P v), or P v without g; v / g / out [S L, nh dh] bf16
    views with any row stride, P [nh L, L] bf16 contiguous) for L > 768, where the P tile no longer fits tensor memory: P streams
    through shared memory in 64-key chunks as the K-major A operand of ONE product per 16-key step for the whole sample group
    (B = the group's v tiles side by side, N = SG x 64). Bound launches cached per (buffers, strides, S, L). ``pdl``: the -DPDL_INF
    build, launched with the PDL attribute (the bf16 three-kernel inference step, the training backward's programmatic dV)."""

    def __init__(
        self, device_index: int, nh: int = H, dh: int | None = None, pdl: bool = False
    ):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100

        self._sm100, self._tm = sm100, sm100._tm
        self.nh, self.dh = nh, dh or 768 // nh
        assert (self.nh, self.dh) in LAYOUTS, (nh, dh)
        self.device_index, self.pdl = device_index, pdl
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def kernel(self, sg: int, gate: bool = True):
        """The cubin for SG samples per item, with or without the gate (built on first use)."""
        return sm100_kernel(
            "pv_stream_sm100",
            "bo_pv_stream_sm100",
            self.device_index,
            pdl=self.pdl,
            src_dir=str(_dir),
            defs=(
                f"SG={sg}",
                f"GATE={int(gate)}",
                f"NHEAD={self.nh}",
                f"DHEAD={self.dh}",
                *(("PDL_INF",) if self.pdl else ()),
            ),
            smem=_stream_smem(sg),
        )

    def _bind(self, v, P, out, S, L, g):
        M, nh, dh, tm, es = v.shape[0], self.nh, self.dh, self._tm, v.element_size()
        gg = v if g is None else g
        sg = pick_group_stream(S, L, self.nsm, nh, dh)
        rs = v.stride(0) * es
        maps = _descriptors(
            tm(P, [L, nh * L], L * 2, [64, 128]),
            tm(v, [nh * dh, L, S], [rs, L * rs], [dh, 64, sg]),  # the group's v tiles: one 3-D box per stage
            tm(gg, [nh * dh, M], gg.stride(0) * es, [dh, 128]),
            tm(out, [nh * dh, M], out.stride(0) * es, [dh, 128]),
        )
        k = self.kernel(sg, g is not None)
        grid = (min(self.nsm, nh * (L // 128) * -(-S // sg)), 1, 1)
        if self.pdl:
            from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import _LaunchPDL

            return _LaunchPDL(k, grid, (256, 1, 1), *maps, L, S)
        return _Launch(k, grid, (256, 1, 1), *maps, L, S)

    def __call__(self, v, P, out, S, g=None):
        L = v.shape[0] // S
        assert (
            v.dtype is torch.bfloat16
            and P.dtype is torch.bfloat16
            and out.dtype is torch.bfloat16
            and P.is_contiguous()
            and tuple(P.shape) == (self.nh * L, L)
            and L % 128 == 0
        ), (v.dtype, P.dtype, tuple(P.shape), L)
        key = (
            v.data_ptr(),
            v.stride(0),
            None if g is None else (g.data_ptr(), g.stride(0)),
            P.data_ptr(),
            out.data_ptr(),
            out.stride(0),
            S,
            L,
            settings.current().bias_only_dit_sg,
        )
        self.runs.bind(key, lambda: self._bind(v, P, out, S, L, g))()
        return out


_STREAM_FAILED = False
_STREAM_READY: set = set()


@device_constant
def stream_ready(index: int, nh: int, dh: int) -> bool:
    """Build and load representative streaming-core cubins for (nh, dh) once; False (after one warning) when a toolchain or driver
    problem keeps the module path for L > 768. A constant to ``torch.compile``."""
    global _STREAM_FAILED
    if _STREAM_FAILED:
        return False
    key = (index, nh, dh)
    if key in _STREAM_READY:
        return True
    try:
        core = PvStreamCore(index, nh, dh)
        core.kernel(1, True)
        core.kernel(1, False)
    except Exception as exc:  # noqa: BLE001 -- a toolchain or driver problem keeps the module path past 768 keys
        _STREAM_FAILED = True
        warnings.warn(
            f"bias-only DiT long-L core (pv_stream_sm100) unavailable, L > {B200_TMEM_MAX_L} keeps the module path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    _STREAM_READY.add(key)
    return True


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


def _dpbx2_smem() -> int:
    """dpbx2_sm100.cu's shared memory (mirrors the source): the P ring and the dbias staging, as many 32 KB stages as fit, barriers."""
    o_st = 2 * 8192 + 4 * 3 * 2048
    return o_st + (_SMEM_MAX - o_st - 1024) // 32768 * 32768 + 256


_DPBX2_FAILED: set = set()


def dpb_pair_on(L: int, nh: int = H, hp: int = 1, nsm: int = 148) -> bool:
    """The bias gradient on CTA pairs (``dpbx2_sm100.cu``: 256-query x 256-key items, a third less TMA intake per product; L768 A = 48,
    16 x 48: 61.2 -> 47.4 us, in-process A/B) -- the default where 256-key tiles divide L and the pairs' items fill the SM pairs (fewer
    leave SMs idle that dpb_sm100.cu's 128-query items would use): at L 128-768 that is L768. settings.bias_only_dit_dpb_pair=0 / 1
    (read per call) keeps dpb_sm100.cu / takes the pairs wherever 256 divides L."""
    forced = settings.current().bias_only_dit_dpb_pair
    if L % 256 or forced is False:
        return False
    return forced is True or (nh // hp) * (L // 256) * (L // 256) >= nsm // 2


class DpbKernel:
    """``dpb_sm100.cu``: dbias [nh L, L] = P o (sum_a do v^T - D) per head, D[h, i] = sum_a dd[a, h, i]; do / v [A L, 768]
    bf16 views, P [nh L, L] bf16, dd [A, nh, L] fp32 (nh x width: one of LAYOUTS). Where ``dpb_pair_on(L)``: ``dpbx2_sm100.cu``
    (the same result on CTA pairs; a failed build warns once and keeps dpb_sm100.cu; ``last`` says which ran).
    ``psb``: a pair bias of its own per sample -- P and dbias [nh, A, L, L], dbias_{h,a} = P_{h,a} o (do_a v_a^T - D_{h,a}), nothing
    summed over the samples, 0 on the keys ``mask`` (bool [L] or [A, L]) drops (the ``-DPSB`` build of dpb_sm100.cu; no CTA-pair
    form)."""

    def __init__(self, device_index: int, nh: int = H, dh: int | None = None, psb: bool = False):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100

        self._sm100, self._tm, self.device_index = sm100, sm100._tm, device_index
        self.psb = psb
        self.nh, self.dh = nh, dh or 768 // nh
        self.nsm = num_sms(device_index)
        self.runs = _Runs()
        self._pair_k = None

    def _bind(self, do, v, P, dd, out, A, L, kmode=0):
        nh, dh = self.nh, self.dh
        hp = (
            2 if dh == 32 else 1
        )  # heads per work item (dpb_sm100.cu HP): two 32-wide heads per 128-byte row
        groups = (A if self.psb else 1) * (nh // hp)  # per sample: items over (sample, head group)
        nj = _dpb_nj(L, self.nsm, groups, hp)
        k = sm100_kernel(
            "dpb_sm100",
            "bo_dpb_sm100",
            self.device_index,
            src_dir=str(_dir),
            defs=(f"NJ={nj}", *_head_defs(nh, dh), *(("PSB",) if self.psb else ())),
            smem=_dpb_smem(nj),
        )
        tm, M = self._tm, do.shape[0]
        planes = nh * A if self.psb else nh
        maps = _descriptors(
            tm(do, [nh * dh, M], do.stride(0) * 2, [hp * dh, 128]),
            tm(v, [nh * dh, M], v.stride(0) * 2, [hp * dh, nj // 2]),
            tm(P, [L, planes * L], L * 2, [32, 128], swizzle=64),
            tm(out, [L, planes * L], L * 2, [32, 32], swizzle=64),
        )
        grid = (min(self.nsm, groups * (L // 128) * (L // nj)), 1, 1)
        if self.psb:  # + the key mask (a pointer per call) and its mode (0 none, 1 one for all samples, 2 per sample)
            return _Launch(k, grid, (256, 1, 1), *maps, PTR, L, A, PTR, kmode)
        return _Launch(k, grid, (256, 1, 1), *maps, PTR, L, A)

    def pair_kernel(self):
        """The cubin of dpbx2_sm100.cu for this layout (built on first use)."""
        return sm100_kernel(
            "dpbx2_sm100",
            "bo_dpbx2_sm100",
            self.device_index,
            src_dir=str(_dir),
            defs=_head_defs(self.nh, self.dh),
            cluster=2,
            smem=_dpbx2_smem(),
        )

    def _bind_pair(self, do, v, P, dd, out, A, L):
        nh, dh = self.nh, self.dh
        hp = 2 if dh == 32 else 1
        k = self._pair_k
        tm, M = self._tm, do.shape[0]
        mdo, mv = (
            tm(do, [nh * dh, M], do.stride(0) * 2, [hp * dh, 128]),
            tm(v, [nh * dh, M], v.stride(0) * 2, [hp * dh, 128]),
        )
        maps = _descriptors(
            mdo,
            mdo,
            mv,
            mv,
            tm(
                P, [L, nh * L], L * 2, [32, 128], swizzle=64
            ),  # (the TF32 build's narrow-box maps unused)
            tm(out, [L, nh * L], L * 2, [32, 32], swizzle=64),
        )
        items = nh // hp * (L // 256) * (L // 256)
        return _Launch(
            k, (min(self.nsm & ~1, 2 * items), 1, 1), (256, 1, 1), *maps, PTR, L, A
        )

    def __call__(self, do, v, P, dd, out, A, mask=None):
        if h100_enabled(self.device_index):
            if _H100_DPB2 and self.dh in (32, 48, 64) and do.shape[0] // A >= 384:  # L256: 64 CTAs, the 64 x 64 kernel wins
                h100_ext().dpb2(do, v, P, dd.sum(0).contiguous(), out, A, self.nh, self.dh, self.nsm)
            else:
                h100_ext().dpb_pipeline(do, v, P, dd, out, A, self.nh, self.dh)
            return out
        L = do.shape[0] // A
        hp = 2 if self.dh == 32 else 1
        if self.psb:
            assert L <= B200_TMEM_MAX_L and P.shape[0] == self.nh * A * L and out.shape[0] == P.shape[0], "per-sample P / dbias [nh A L, L]"
            kmode = 0 if mask is None else (1 if mask.numel() == L else 2)
            assert mask is None or (mask.dtype is torch.bool and mask.is_contiguous() and mask.numel() in (L, A * L)), "bool mask [L] / [A, L]"
            key = ("psb", do.data_ptr(), do.stride(0), v.data_ptr(), v.stride(0), P.data_ptr(), out.data_ptr(), A, L, kmode)
            self.runs.bind(key, lambda: self._bind(do, v, P, dd, out, A, L, kmode))(dd, dd if mask is None else mask)
            return out
        pair = (
            not self.psb
            and dpb_pair_on(L, self.nh, hp, self.nsm)
            and (self.device_index, self.nh, self.dh) not in _DPBX2_FAILED
        )
        if pair and self._pair_k is None:
            try:
                self._pair_k = self.pair_kernel()
            except Exception as exc:  # noqa: BLE001 -- a toolchain or driver problem keeps dpb_sm100.cu
                _DPBX2_FAILED.add((self.device_index, self.nh, self.dh))
                warnings.warn(
                    f"bias-only DiT CTA-pair bias gradient (dpbx2_sm100) unavailable, keeping dpb_sm100: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                pair = False
        self.last = "pair" if pair else "single"
        key = (
            pair,
            do.data_ptr(),
            do.stride(0),
            v.data_ptr(),
            v.stride(0),
            P.data_ptr(),
            out.data_ptr(),
            A,
            L,
        )
        bind = self._bind_pair if pair else self._bind
        self.runs.bind(key, lambda: bind(do, v, P, dd, out, A, L))(dd)
        return out


class PvDpb:
    """``bo_pvdpb.cu``: pv dV (dvg's dV half = P^T do per head and sample; ``PvGateCore`` without the gate) and the bias gradient
    (``DpbKernel``'s dpb_sm100.cu) in ONE launch -- the first NPV CTAs run pv dV's body, the rest the bias gradient's, on their own
    virtual grids, each sized as its own launch would be. Launched as a programmatic dependent. Where ``dpb_pair_on`` (the bias
    gradient on CTA pairs, dpbx2_sm100.cu: its cta_group::2 MMAs cannot share a function with pv dV's cta_group::1 ones) the call
    returns False and launches nothing, unless ``single``: then the fused launch with dpb_sm100.cu (the bwdf7 node A/B against pv dV +
    dpbx2)."""

    def __init__(
        self,
        device_index: int,
        nh: int = H,
        dh: int | None = None,
        single: bool = False,
    ):
        import hashlib
        from miniworld_engine.kernels.augmented_attention.cuda import sm100

        self._sm100, self._tm, self.device_index, self.single = (
            sm100,
            sm100._tm,
            device_index,
            single,
        )
        self.nh, self.dh = nh, dh or 768 // nh
        self.nsm = num_sms(device_index)
        self.runs = _Runs()
        h = (
            hashlib.sha256()
        )  # the cubin cache hashes bo_pvdpb.cu only: the included bodies by this define
        for f in ("pv_gate_inf.cu", "dpb_sm100.cu"):
            h.update((_dir / f).read_bytes())
        self._src = f"SRC_HASH={h.hexdigest()[:16]}"

    def kernel(self, sg: int, vb: int, nj: int):
        """The cubin for SG samples per pv item, VB keys per v tile, NJ-key bias-gradient tiles."""
        defs = (
            f"SG={sg}",
            *(() if vb == 128 else (f"VB={vb}",)),
            f"NJ={nj}",
            *_head_defs(self.nh, self.dh),
            self._src,
        )
        return sm100_kernel(
            "bo_pvdpb",
            "bo_pvdpb_sm100",
            self.device_index,
            pdl=True,
            src_dir=str(_dir),
            defs=defs,
            smem=max(_pv_smem(vb), _dpb_smem(nj)),
        )

    def _bind(self, do, pt, dv, v, P, dbias, A, L):
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import _LaunchPDL

        nh, dh, tm, M = self.nh, self.dh, self._tm, do.shape[0]
        hp = 2 if dh == 32 else 1
        vb, sg = _pv_vb(L, A, dh), pick_group(A, L, self.nsm, nh, dh)
        nj = _dpb_nj(L, self.nsm, nh // hp, hp)
        maps = _descriptors(
            tm(pt, [L, nh * L], L * 2, [64, 128]),
            tm(do, [nh * dh, M], do.stride(0) * 2, [dh, vb]),
            tm(do, [nh * dh, M], do.stride(0) * 2, [dh, 128]),
            tm(dv, [nh * dh, M], dv.stride(0) * 2, [dh, 128]),
            tm(do, [nh * dh, M], do.stride(0) * 2, [hp * dh, 128]),
            tm(v, [nh * dh, M], v.stride(0) * 2, [hp * dh, nj // 2]),
            tm(P, [L, nh * L], L * 2, [32, 128], swizzle=64),
            tm(dbias, [L, nh * L], L * 2, [32, 32], swizzle=64),
        )
        npv = min(self.nsm, nh * (L // 128) * -(-A // sg))
        ndpb = min(self.nsm, nh // hp * (L // 128) * (L // nj))
        return _LaunchPDL(
            self.kernel(sg, vb, nj),
            (npv + ndpb, 1, 1),
            (256, 1, 1),
            *maps,
            PTR,
            L,
            A,
            npv,
        )

    def __call__(self, do, pt, dv, v, P, dd, dbias, A) -> bool:
        """do [A L, DA] view, pt / P [nh L, L] (P^T, P) contiguous, dv [A L, DA] view (dV out), v [A L, DA] view, dd [A, nh, L] fp32,
        dbias [nh L, L] out; bf16 but dd. False (nothing launched) where the bias gradient belongs on CTA pairs, and past L 768 (pv
        dV then runs the streaming core, ``pv_stream_sm100.cu``, which is not a body of this launch)."""
        L = do.shape[0] // A
        hp = 2 if self.dh == 32 else 1
        if L > B200_TMEM_MAX_L or (dpb_pair_on(L, self.nh, hp, self.nsm) and not self.single):
            return False
        key = (
            do.data_ptr(),
            do.stride(0),
            pt.data_ptr(),
            dv.data_ptr(),
            dv.stride(0),
            v.data_ptr(),
            v.stride(0),
            P.data_ptr(),
            dbias.data_ptr(),
            A,
            L,
        )
        self.runs.bind(key, lambda: self._bind(do, pt, dv, v, P, dbias, A, L))(dd)
        return True


# ------------------------------------------------------------------------------------------------ expand GEMM + SwiGLU (training)
class GemmSwigluAB:
    """The token DiT's ``gemm_swiglu2_sm100.cu`` built with SAVE_AB: h = silu(a) b and [a | b] = X [Wa; Wb]^T (bf16) from one
    kernel, the SwiGLU in the GEMM epilogue, so the training forward does not read [a | b] back for h (A = 48: L384 82 us
    against cuBLAS 59 + SwiGLU 29, L768 152 against 170). X [M, 768], W [4 x 768, 768], ab [M, 4 x 768], h [M, 2 x 768]
    contiguous; bound launches (their TMA descriptors) are cached per buffer set."""

    SMEM = 4 * 32768 + 32768 + 4 * 16384 + 512

    def __init__(self, device_index: int, K: int = 768):
        self.device_index = device_index
        if h100_enabled(device_index):
            self.K, self.H = K, 2 * K
            return
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu

        self._tm, self.K, self.H = sm100._tm, K, 2 * K
        self.k = sm100_kernel(
            "gemm_swiglu2_sm100",
            "gemm_swiglu2_sm100",
            device_index,
            src_dir=gemm_swiglu._dir,
            defs=(f"DIM={K}", "HMUL=2", "ITEM_SCHED", "SAVE_AB"),
            cluster=2,
            smem=self.SMEM,
        )
        self.nsm = num_sms(device_index)
        self.runs = _Runs()

    def _bind(self, x, w, ab, h):
        M, K, H = x.shape[0], self.K, self.H
        tm = lambda t, rows, cols, box: self._tm(t, [cols, rows], t.stride(0) * 2, box)
        maps = _descriptors(
            tm(x, M, K, [64, 64]),
            tm(w[:H], H, K, [64, 128]),
            tm(w[H:], H, K, [64, 128]),
            tm(h, M, H, [64, 64]),
            tm(ab[:, :H], M, H, [64, 64]),
            tm(ab[:, H:], M, H, [64, 64]),
        )
        grid, tiles = (self.nsm & ~1, 1, 1), (M + 127) // 128

        return _Launch(self.k, grid, (512, 1, 1), *maps, 1, tiles)

    def __call__(self, x, w, ab, h):
        if h100_enabled(self.device_index):
            h100_ext().swiglu_pipeline(x, w, ab, h)
            return h
        M, K, H = x.shape[0], self.K, self.H
        assert (
            x.shape == (M, K)
            and w.shape == (2 * H, K)
            and ab.shape == (M, 2 * H)
            and h.shape == (M, H)
        )
        assert all(
            t.is_contiguous() and t.dtype is torch.bfloat16 for t in (x, w, ab, h)
        )
        self.runs.bind(
            (x.data_ptr(), w.data_ptr(), ab.data_ptr(), h.data_ptr(), M),
            lambda: self._bind(x, w, ab, h),
        )()
        return h


# --------------------------------------------------------------------------------------------------- GEMM + residual + AdaLN
def _resln_smem(cl: int = 8) -> int:
    """gemm_resln_sm100.cu's shared memory: the K ring fills what the epilogue slots, the exchange area and the barriers leave."""
    return _SMEM_MAX - 1024 + 256  # O_BAR + 256


class ResLnGemm:
    """``gemm_resln_sm100.cu``: x += sigmoid(gl) * (A W^T) (fp32, in place), then xa = AdaLN(x) (bf16) -- or, ``final``,
    out = x in bf16 -- with W [768, K]. A cluster of CL CTAs per 128-row tile (8 while the row tiles x 8 fit the SMs, else 4),
    the LayerNorm statistics exchanged once through distributed shared memory. Bound launches are cached per buffer set."""

    def __init__(
        self,
        device_index: int,
        final: bool = False,
        defs: tuple[str, ...] = (),
        presig: bool = False,
    ):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100

        self._sm100, self._tm, self.final = sm100, sm100._tm, final
        self.defs = (*defs, *(("PRESIG",) if presig else ()))
        self.device_index = device_index
        self.nsm = num_sms(device_index)
        self.runs = _Runs()

    def kernel(self, cl: int):
        defs = (f"CL={cl}", *(("FINAL",) if self.final else ()), *self.defs)
        return sm100_kernel(
            "gemm_resln_sm100",
            "bo_gemm_resln_sm100",
            self.device_index,
            src_dir=str(_dir),
            defs=defs,
            cluster=cl,
            smem=_resln_smem(cl),
        )

    def cluster(self, M: int) -> int:
        return 8 if (M // 128) * 8 <= self.nsm else 4

    def _bind(self, a, w, x, gl, ms, mb, out, T, eps):
        M, K = a.shape
        cl = self.cluster(M)
        nc = 768 // cl
        tm = self._tm
        tab = lambda t: tm(t, [768, t.shape[0]], t.stride(0) * 2, [32, 128], swizzle=64)
        maps = (
            tm(a, [K, M], K * 2, [64, 128]),
            tm(w, [K, 768], K * 2, [64, nc]),
            tm(x, [768, M], 768 * 4, [32, 128], swizzle=128, dtype="f32"),
            tab(gl),
            tab(gl if ms is None else ms),
            tab(gl if mb is None else mb),
            tm(out, [768, M], 768 * 2, [32, 128], swizzle=64),
            tm(x, [768, M], 768 * 4, [32, 32], swizzle=128, dtype="f32"),
            tm(out, [768, M], 768 * 2, [32, 32], swizzle=64),
        )
        grid, k = (cl * (M // 128), 1, 1), self.kernel(cl)

        def run():
            k(grid, (256, 1, 1), *maps, int(K), int(T), float(eps))

        run.keep = (maps, a, w, x, gl, ms, mb, out)
        return run

    def __call__(self, a, w, x, gl, ms, mb, out, T, eps=1e-5):
        """a [M, K], w [768, K] bf16; x [M, 768] fp32 (updated); gl / ms / mb table views [T, 768] (ms, mb unused when final);
        out [M, 768] bf16: xa, or with ``final`` the block's output."""
        key = (
            a.data_ptr(),
            w.data_ptr(),
            x.data_ptr(),
            gl.data_ptr(),
            None if ms is None else ms.data_ptr(),
            None if mb is None else mb.data_ptr(),
            out.data_ptr(),
            a.shape,
            T,
        )
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

    SMEM = (
        6 * 16384
        + 2 * _COND_NT * 128
        + 8 * 3 * 2048
        + 2 * 2 * _COND_NT * 4
        + 2 * 2 * 128 * 4
        + 256
    )

    def __init__(self, device_index: int):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100

        self._tm = sm100._tm
        self.k = sm100_kernel(
            "cond_tables_sm100",
            "bo_cond_tables_sm100",
            device_index,
            src_dir=str(_dir),
            defs=(f"NT={_COND_NT}",),
            smem=self.SMEM,
        )
        self.nsm = num_sms(device_index)
        self.runs = _Runs()

    def _bind(self, c, w, bias, colsum, out, n_g1, eps):
        T, N = c.shape[0], w.shape[0]
        tm = self._tm
        maps = (
            tm(c, [384, T], c.stride(0) * 2, [64, 128]),
            tm(w, [384, N], 384 * 2, [64, _COND_NT]),
            tm(out, [N, T], N * 2, [32, 32], swizzle=64),
        )
        items = (T // 128) * (N // _COND_NT)
        grid = (min(self.nsm, items), 1, 1)

        def run():
            self.k(
                grid,
                (384, 1, 1),
                *maps,
                bias,
                colsum,
                int(T),
                int(N),
                int(n_g1),
                float(eps),
            )

        run.keep = (maps, c, w, bias, colsum, out)
        return run

    def __call__(self, c, w, bias, colsum, out, n_g1, eps=1e-5):
        key = (c.data_ptr(), c.shape, w.data_ptr(), out.data_ptr())
        run = self.runs.get(key)
        if run is None:
            run = self.runs[key] = self._bind(c, w, bias, colsum, out, n_g1, eps)
        run()
        return out


__all__ = [
    "CondTables",
    "DpbKernel",
    "dpb_pair_on",
    "GemmSwigluAB",
    "PvDpb",
    "PvGateCore",
    "ResLnGemm",
    "gate_bwd_rows",
    "transpose_hll",
    "adaln_in_rows",
    "core_supported",
    "ln_rows",
    "pick_group",
    "resgate_adaln_rows",
    "resgate_out_rows",
    "psb32_bwd",
    "psb32_fwd",
    "psb_bwd",
    "psb_fwd",
    "softmax_t",
    "softmax_rows",
    "swiglu_rows",
]

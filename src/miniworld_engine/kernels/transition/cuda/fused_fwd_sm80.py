"""The fused sm_80 Transition forward at D = 64 / 128, the generalisation of ``fused_sm80``'s forward kernel (``sm80/tr_fwd_sm80.cuh``) over (D, H).

One kernel per call: a warp owns 32 token rows for the whole hidden dimension (LayerNorm into the GEMM1 A fragments, [a | b] per 16 hidden units, SwiGLU in
the C fragments, squeeze in f16 with the residual as the accumulator's initial value), weights streaming from L2 through a cp.async ring.  Built for
(D, H) in ``SHAPES``; any row count (the kernel predicates the tail).  It writes ``xn`` and (mean, rstd) when asked, which is what the wide backward
(``fused_wide_sm80``) consumes.  The weight layouts are packed per call by one gather launch (cached across inference calls by parameter version).
"""

import functools
from pathlib import Path

import torch

from ... import _capture
from ..._nvcc import build_failed, ensure_cuda_home, host_flags, load_extension
from ..._sm120_builds import per_arch

_dir = Path(__file__).parent / "sm80"

#: (D, H) with a kernel build; CH is the hidden chunk of the weight ring.
SHAPES = ((64, 128), (64, 256), (128, 256), (128, 512))
CH = 32


def _build(suffix: str, gencode: str):
    ensure_cuda_home()
    return load_extension(
        name=f"transition_fwd_{suffix}",
        sources=[str(_dir / "transition_fwd_sm80.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            gencode,
            f"-I{_dir}",
            *(["-DMWE_SM120"] if suffix == "sm120" else []),  # the RTX 5090's ring depth (99 KB a block): tr_fwd_sm80.cuh
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


#: the A100 build (``transition_fwd_sm80``) and the GeForce RTX 5090 one (``transition_fwd_sm120``: the same source for sm_120a, a shallower weight ring)
_ext, _ext_sm80, _ext_sm120 = per_arch(_build)


@torch.compiler.assume_constant_result
def loads() -> bool:
    """The extension builds / loads; a failed build / load raises ``_nvcc.KernelBuildError`` (never a fallback).  Constant for dynamo."""
    try:
        _ext()
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed("fused sm80 Transition forward", exc)
        return False  # reached only while planning (build_failed returns there)
    return True


#: Fewest rows for which the one-kernel forward beats the wide chain (LN, dual GEMM, squeeze): below it the kernel's fixed per-tile time (a 256-row tile takes
#: 20 us at D = 64 / H = 256 and 55 us at D = 128 / H = 512, whatever the row count) leaves most SMs idle.  The crossovers measured against the chain with the
#: lean pipelined GEMM kernels (CUDA graph; docs page): (64, 128) 2 k rows, (64, 256) 4.5 k, (128, 256) 10 k, (128, 512) 13 k.
MIN_ROWS = {(64, 128): 2048, (64, 256): 4096, (128, 256): 10240, (128, 512): 12288}


def supports(d: int, h: int, rows: int | None = None) -> bool:
    """A build exists for (d, h), the switch is not off and, when ``rows`` is given, the kernel is the faster path at that row count."""
    if (d, h) not in SHAPES:
        return False
    return rows is None or rows >= MIN_ROWS[(d, h)]


# --------------------------------------------------------------------------------------------------------------- rows per launch
#: Entries of a CTA's work list in the one-kernel forward / backward kernels (``sm80/tr_fwd_sm80.cuh`` / ``tr_bwd_sm80.cuh`` ``MAX_ITEMS``); the
#: schedule keeps two for the remainder's half tile and the terminator, so one launch (one CTA per SM) takes at most 253 tiles per SM.
MAX_ITEMS = 255


_SMS: dict[int, int] = {}


def _num_sms(index: int) -> int:
    n = _SMS.get(index)
    if n is None:
        n = _SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return n


#: The wide chain's GEMM grids put 128-row tiles on gridDim.y (at most 65 535).
CHAIN_MAX_ROWS = 65535 * 128


def max_rows(device: torch.device, dtype: torch.dtype = torch.bfloat16) -> int:
    """Most rows one call of the sm_80 Transition takes on ``device``: (MAX_ITEMS - 2) tiles per SM of the one-kernel kernels (bf16: 256-row
    tiles, ``fused_sm80`` / the bf16 builds / ``fused_bwd_sm80``; the fp32 build: 128-row tiles) and the wide chain's row grid
    (``CHAIN_MAX_ROWS``).  An inference call past it runs as one call per contiguous row block (``transition_sm80``; FoldForge 5I28: the template
    pair stack at 4 x 2048 x 2048 rows, the confidence block at 5 x 1536 x 1536); a training call past it is refused (``modules.Transition``)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    tile = 128 if dtype is torch.float32 else 256
    return min((MAX_ITEMS - 2) * _num_sms(index) * tile, CHAIN_MAX_ROWS)


# ------------------------------------------------------------------------------------------------------------------ weight layouts
def _k_perm(dev, d):
    """Physical GEMM1 k index 16 s + kk -> input column (thread q of a quad holds the 16 B vectors at 32 i + 8 q)."""
    s = torch.arange(d // 16, device=dev).view(-1, 1)
    kk = torch.arange(16, device=dev).view(1, -1)
    return (32 * (s // 2) + 8 * ((kk % 8) // 2) + 4 * (s % 2) + 2 * (kk // 8) + kk % 2).reshape(-1)


def _o_perm(dev, d):
    """Physical GEMM2 output column 8 J + 2 q + e -> output column 32 (J / 4) + 8 q + 2 (J % 4) + e."""
    pcol = torch.arange(d, device=dev)
    jj, q, e = pcol // 8, (pcol % 8) // 2, pcol % 2
    return 32 * (jj // 4) + 8 * q + 2 * (jj % 4) + e


def _layout_fwd(wa, wb, ws, dev, d, h):
    """[nchunk][W1 (0.5 Wa | Wb, k-permuted, [D / 8 granules][2 CH rows][16 B]) | W2 (Ws, output-permuted, [CH / 8][D rows][16 B])] on element codes."""
    kp = _k_perm(dev, d)
    wa_p, wb_p = wa[:, kp], wb[:, kp]
    ws_p = ws[_o_perm(dev, d)]
    nstep, nchunk = CH // 16, h // CH
    w1 = torch.stack([wa_p.view(nchunk, nstep, 16, d), wb_p.view(nchunk, nstep, 16, d)], 2).reshape(nchunk, 2 * CH, d)
    w1 = w1.view(nchunk, 2 * CH, d // 8, 8).transpose(1, 2).reshape(nchunk, -1)
    w2 = ws_p.view(d, nchunk, CH // 8, 8).permute(1, 2, 0, 3).reshape(nchunk, -1)
    return w1, w2


@functools.lru_cache(maxsize=16)
def _tables(dev, d, h):
    """Gather tables of the pack kernel: idx16 (the packed weights [nchunk][W1 | W2]: element | source << 26 (Wa, Wb, Ws) | x0.5 << 28 | f16 << 29) and idx32 (the
    LayerNorm affine as float4 slots: element | beta << 26)."""
    n = h * d
    code = lambda src: torch.arange(n, device=dev, dtype=torch.int64) | (src << 26)  # noqa: E731
    wa, wb = code(0).view(h, d), code(1).view(h, d)
    ws = code(2).view(d, h)
    w1, w2 = _layout_fwd(wa | (1 << 28), wb, ws | (1 << 29), dev, d, h)  # W1 carries 0.5 Wa; W2 runs in f16
    idx16 = torch.cat([w1, w2], 1).reshape(-1).to(torch.int32).contiguous()
    s_, q_ = (
        torch.arange(d // 16, device=dev).view(-1, 1),
        torch.arange(4, device=dev).view(1, -1),
    )
    c0 = (32 * (s_ // 2) + 8 * q_ + 4 * (s_ % 2)).reshape(-1, 1) + torch.arange(4, device=dev).view(1, -1)
    idx32 = torch.cat([c0.reshape(-1), c0.reshape(-1) | (1 << 26)]).to(torch.int32).contiguous()
    return idx16, idx32


_packs: dict = _capture.weight_cache("transition_fused_fwd_sm80.packs")


def pack(gamma, beta, wa, wb, ws, cache: bool):
    """(w, gb): the kernel's weight layout and LayerNorm affine slots (``gamma`` None: the bare FFN, no affine; gb is empty).  The weights may be bf16 or fp32
    (an fp32 master): they are cast to bf16 here, on a miss.  ``cache``: reuse the pack of the same parameter versions, keyed on the parameters themselves and scoped by
    ``_capture`` (an eager entry serves eager calls only; a capture packs once, recorded; inference only)."""
    d, h = ws.shape[0], wa.shape[0]
    ln = gamma is not None

    def build():
        idx16, idx32 = _tables(wa.device, d, h)
        g, b = gamma, beta
        if not ln:
            idx32 = idx32[:0]
            g = b = wa.new_empty((0,), dtype=torch.float32)
        out16 = torch.empty(idx16.numel(), dtype=torch.bfloat16, device=wa.device)
        gb = torch.empty(idx32.numel(), dtype=torch.float32, device=wa.device)
        # fmt: off
        _ext().pack(*(w.to(torch.bfloat16).contiguous() for w in (wa, wb, ws)), g, b, idx16, idx32, out16, gb)
        # fmt: on
        return (
            out16,
            gb,
            (gamma, beta, wa, wb, ws),
        )  # the parameters stay alive with their pack: a recycled data_ptr cannot alias it

    if not cache:
        return build()[:2]
    if not torch.is_grad_enabled():
        # grad off: a CUDA-graph capture reads the pack its eager warm-up made (``capture_reads_eager``): no cast / pack kernel per replay
        params = [gamma, beta, wa, wb, ws] if ln else [wa, wb, ws]
        return _capture.param_forms(("transition_sm80_pack", d, h, ln), params, lambda: build()[:2], capture_reads_eager=True)
    key = tuple((t.data_ptr(), t._version, t.dtype) for t in ((gamma, beta, wa, wb, ws) if ln else (wa, wb, ws))) + (d, h, ln)
    return _capture.lookup(_packs, key, build, limit=8)[:2]


def forward(x, gamma, beta, wa, wb, ws, eps: float, save: bool, cache: bool, ln: bool = True):
    """(out, xn, stats) of the fused forward: x [M, D] bf16, gamma / beta f32 [D], weights bf16 or fp32 (cast in ``pack``); xn / stats are empty unless ``save`` (and ``ln``).
    Not ``ln``: the bare SwiGLU FFN (no LayerNorm, no residual); gamma / beta are ignored."""
    d, h = x.shape[1], wa.shape[0]
    w, gb = pack(gamma if ln else None, beta if ln else None, wa, wb, ws, cache)
    out, xn, stats = _ext().fwd(d, h, ln, x, w, gb, eps, save and ln)
    return out, xn, stats


# ------------------------------------------------------------------------------------------------------------------ fp32 (inference)
#: (D, H) of the fp32 build (``TrCfgT<D, H, true>``: x / out fp32, the wide family's fp32 scheme -- fp16 operands at TF32's rounding with the f32_forms
#: scales, fp32 accumulation -- in the one-kernel forward, 16 rows per warp): the float32 eval rows where the three-launch wide chain is slower than the
#: kits (token_pair d 128 n 2).
F32_SHAPES = ((128, 256),)
#: Fewest rows for which the fused fp32 forward serves (below it the wide chain's few-rows split-K squeeze).
F32_MIN_ROWS = {(128, 256): 8192}


def f32_supports(d: int, h: int, rows: int) -> bool:
    """The fp32 build exists for (d, h) and serves ``rows``."""
    return (d, h) in F32_SHAPES and rows >= F32_MIN_ROWS[(d, h)]


@functools.lru_cache(maxsize=16)
def _tables_f32(dev, d, h):
    """Gather tables of ``pack_raw`` (the fp16 forms copied as they are: no x0.5, no conversion) and the affine slots (as ``_tables``)."""
    n = h * d
    code = lambda src: torch.arange(n, device=dev, dtype=torch.int64) | (src << 26)  # noqa: E731
    w1, w2 = _layout_fwd(code(0).view(h, d), code(1).view(h, d), code(2).view(d, h), dev, d, h)
    idx16 = torch.cat([w1, w2], 1).reshape(-1).to(torch.int32).contiguous()
    return idx16, _tables(dev, d, h)[1]


def pack_f32(gamma, beta, wab16, ws16):
    """(w, gb) of the fp32 build from the forms (``fused_wide_sm80._f32_forms``: wab16 [2H, D], ws16 [D, H] fp16) and the fp32 affine: one launch."""
    d, h = ws16.shape
    idx16, idx32 = _tables_f32(ws16.device, d, h)
    out16 = torch.empty(idx16.numel(), dtype=torch.float16, device=ws16.device)
    gb = torch.empty(idx32.numel(), dtype=torch.float32, device=ws16.device)
    _ext().pack_raw(wab16, ws16, gamma, beta, idx16, idx32, out16, gb)
    return out16, gb


def forward_f32(x, w, gb, sc, eps: float):
    """x + transition(x) of fp32 rows x [M, D] (inference) from ``pack_f32``'s layout and the forms' scales ``sc``."""
    d, h = x.shape[1], w.numel() // (3 * x.shape[1])
    return _ext().fwd_f32(d, h, x, w, gb, sc, eps)

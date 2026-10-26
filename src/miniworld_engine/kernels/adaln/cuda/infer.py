"""H100 (sm_90a) AdaptiveLayerNorm: ``adaln_sm90.cu``.

    y = sigmoid(LN(c) Wscale^T + b) * LN(x) + LN(c) Wbias^T,    LN(c) = (c - mean) rstd * lnw

Two launches (the second under programmatic dependent launch): ``ln_rows`` (LN(cond) and the statistics of x, one warp per row) and ``adaln_gemm``
(one wgmma GEMM over [Wscale | Wbias] with the gate and the LayerNorm of x in its epilogue). bf16 on wgmma k16; fp32 on the TF32 tensor cores
(the A operand is rounded to the TF32 grid to nearest by ``ln_rows``, the weights once per weight pack). Built on first use (``load_extension``),
never at import. Entry point: :func:`adaln_inference`.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque

_dir = Path(__file__).parent
_kernels = _dir.parent.parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="adaln_sm90" + ("_trace" if os.environ.get("CT_TRACE") else ""),
        sources=[str(_dir / "adaln_sm90.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            *gencodes("90a"),
            "--expt-relaxed-constexpr",
            f"-I{_kernels / 'transition' / 'cuda' / 'anthropic_v5'}",
            f"-I{_kernels / 'conditioned_transition' / 'cuda'}",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *(["-DCT_TRACE"] if os.environ.get("CT_TRACE") else []),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


@functools.lru_cache(maxsize=8)
def _nsm(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def num_sms(device: torch.device) -> int:
    return _nsm(device.index if device.index is not None else torch.cuda.current_device())


def pick_ht(m: int, n: int, nsm: int) -> int:
    """Hidden / output columns per tile: 128 (128 x 256 tiles) or 64 (twice the tiles): the one whose last wave of the persistent grid is fuller
    (ties: 128, the larger tile)."""
    best, best_eff = 128, -1.0
    for ht in (128, 64):
        if n % ht:
            continue
        tiles = ((m + 127) // 128) * (n // ht)
        eff = tiles / (-(-tiles // nsm) * nsm)
        if eff > best_eff + 1e-9:
            best, best_eff = ht, eff
    return best


def supported(x: torch.Tensor, cond: torch.Tensor, nx: int, nc: int) -> bool:
    """sm_90 only, bf16, or fp32 under TF32 (``allow_tf32``): the TF32 tensor cores are what runs, and strict fp32 must stay on cuBLAS."""
    if not (x.is_cuda and cond.is_cuda and x.dtype == cond.dtype):
        return False
    if x.dtype == torch.float32:
        if not torch.backends.cuda.matmul.allow_tf32:
            return False
        kq = 32
    elif x.dtype == torch.bfloat16:
        kq = 64
    else:
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(idx) == (9, 0) and nx % 64 == 0 and nc % kq == 0 and nx <= 1024 and nc <= 1024


def rna_tf32(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> the TF32 grid, to nearest (ties away), as fp32."""
    return ((t.contiguous().view(torch.int32) + 0x1000) & ~0x1FFF).view(torch.float32)


def pack_weights(w_scale: torch.Tensor, b_scale: torch.Tensor, w_bias: torch.Tensor, dtype: torch.dtype):
    """([Wscale; Wbias] [2 NX, NC] in ``dtype`` (fp32: on the TF32 grid), the scale bias as fp32 (rounded to ``dtype`` first, as the Triton path's
    casts do)). The GEMM loads the Wscale and Wbias rows of a tile by two TMA boxes: no per-tile packing."""
    wc = torch.cat([w_scale, w_bias]).to(dtype)
    if dtype == torch.float32:
        wc = rna_tf32(wc)
    return wc.contiguous(), b_scale.to(dtype).float().contiguous()


def packed(w_scale, b_scale, w_bias, dtype):
    """``pack_weights`` cached while the parameters are unmodified (``_capture.param_forms``)."""
    return _capture.param_forms(f"adaln_cuda_{dtype}", (w_scale, b_scale, w_bias), lambda: pack_weights(w_scale, b_scale, w_bias, dtype))


def _infer_fake(x, cond, lnw, wc, bs, eps_x, eps_cond, ht, rna):
    """y: x's shape and dtype."""
    return torch.empty_like(x)


@opaque(fake=_infer_fake, name="adaln_cuda_inference")
def _infer(x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, wc: torch.Tensor, bs: torch.Tensor, eps_x: float, eps_cond: float, ht: int,
           rna: bool) -> torch.Tensor:
    """x [M, NX], cond [M, NC] contiguous; ``wc`` / ``bs`` from :func:`pack_weights`."""
    e = ext()
    m, nx = x.shape
    aff = torch.empty_like(cond)
    st = torch.empty(m, 2, device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    e.ln_rows(cond, lnw, x, aff, st, None, eps_cond, eps_x)
    e.adaln_gemm(aff, wc, bs, x, st, y, None, num_sms(x.device), ht, rna, True, False)
    return y


def adaln_inference(x, cond, lnw, w_scale, b_scale, w_bias, eps_x, eps_cond, *, rna=False, ht=None):
    """AdaLN forward without saving anything for a backward, ``x`` ``(..., NX)`` and ``cond`` ``(..., NC)`` of one dtype (bf16 / fp32 under TF32);
    the three projections in any dtype (cast and concatenated once per weight pack). ``rna``: round the fp32 result to the TF32 grid (its consumer
    is a TF32 GEMM)."""
    shape = x.shape
    nx, nc = shape[-1], cond.shape[-1]
    x2 = x.reshape(-1, nx).contiguous()
    c2 = cond.reshape(-1, nc).contiguous()
    ht = ht or pick_ht(x2.shape[0], nx, num_sms(x.device))
    wc, bs = packed(w_scale, b_scale, w_bias, x.dtype)
    y = _infer(x2, c2, lnw.float().contiguous(), wc, bs, float(eps_x), float(eps_cond), ht, rna)
    return y.reshape(shape)


def _fwd_saved_fake(x, cond, lnw, w_scale, b_scale, w_bias, eps_x, eps_cond, ht):
    """(y, LN(cond) lnw, its stats, the stats of x, sb) shapes."""
    m, nx = x.shape
    f = lambda: x.new_empty((m, 2), dtype=torch.float32)  # noqa: E731
    return x.new_empty((m, nx)), cond.new_empty(cond.shape), f(), f(), x.new_empty((m, 2 * nx))


@opaque(fake=_fwd_saved_fake, name="adaln_cuda_train_fwd")
def fwd_saved(x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, w_scale: torch.Tensor, b_scale: torch.Tensor, w_bias: torch.Tensor,
              eps_x: float, eps_cond: float, ht: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The training forward in the layout of ``adaln_rows``' backward: (y, LN(cond) lnw, its (mean, rstd), the (mean, rstd) of x, sb) with the
    scale pre-activation in sb[:, :NX] (the second half is not written: the backward does not read it). Weights are cast and concatenated at every
    call (they change between steps)."""
    e = ext()
    m, nx = x.shape
    wc, bs = pack_weights(w_scale, b_scale, w_bias, x.dtype)
    aff = torch.empty_like(cond)
    stx = torch.empty(m, 2, device=x.device, dtype=torch.float32)
    stc = torch.empty(m, 2, device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    sb = torch.empty(m, 2 * nx, device=x.device, dtype=x.dtype)
    mode = os.environ.get("ADALN_MODE", "h128")   # measured (M = 18432 / 36864 / 49152, bf16): 128-column tiles 68 / 153 / 194 us, 64-column 69 / 160 / 239, A-resident 112 / 183 / 199
    ar = x.dtype == torch.bfloat16 and cond.shape[1] == 384 and nx % 64 == 0 and mode == "ar"   # A-resident row blocks (see adaln_sm90.cu)
    if mode == "h64": ht = 64
    elif mode == "h128": ht = 128
    elif ar: ht = 64
    e.ln_rows(cond, lnw, x, aff, stx, stc, eps_cond, eps_x)
    e.adaln_gemm(aff, wc, bs, x, stx, y, sb, num_sms(x.device), ht, False, True, ar)
    return y, aff, stc, stx, sb





def _colsum_fake(parts):
    """[N]: the column sums of [P, N]."""
    return parts.new_empty((parts.shape[1],))


@opaque(fake=_colsum_fake, name="adaln_cuda_colsum")
def colsum(parts: torch.Tensor) -> torch.Tensor:
    """sum over the rows of ``parts`` [P, N] fp32 in a fixed order (one launch)."""
    return ext().colsum(parts)


def _train_bwd_fake(dy2, x2, c2, cn, stc, stx, sb, wc, ws, bs, wb, has_bs):
    """(dx, dcond, dwc, dWs, dbs, dWb) like ``adaln_h100_bwd``."""
    d, dc = ws.shape
    return (
        x2.new_empty(x2.shape),
        c2.new_empty(c2.shape),
        wc.new_empty((dc,), dtype=torch.float32),
        ws.new_empty((d, dc), dtype=torch.float32),
        ws.new_empty((d if has_bs else 0,), dtype=torch.float32),
        ws.new_empty((d, dc), dtype=torch.float32),
    )


@opaque(fake=_train_bwd_fake, name="adaln_cuda_train_bwd")
def train_bwd(
    dy2: torch.Tensor, x2: torch.Tensor, c2: torch.Tensor, cn: torch.Tensor, stc: torch.Tensor, stx: torch.Tensor, sb: torch.Tensor,
    wc: torch.Tensor, ws: torch.Tensor, bs: torch.Tensor | None, wb: torch.Tensor, has_bs: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward of ``adaln_h100_bwd`` (the row kernels of ``adaln_rows.cu`` around cuBLAS) with the column sums of its block partials taken by one
    fixed-order kernel of ours instead of two PyTorch reductions (20 us each at M = 18432)."""
    from . import _ext as rows_ext
    from . import _wcat_bcat

    d = x2.shape[-1]
    e = rows_ext()
    wcat, _ = _wcat_bcat(x2, ws, bs, wb)
    dsb, dx, pbs = e.adaln_bwd(dy2, x2, sb, stx)
    dcn = torch.mm(dsb, wcat)
    dwcat = torch.mm(dsb.t(), cn, out_dtype=torch.float32) if dsb.dtype is torch.bfloat16 else torch.mm(dsb.t(), cn)
    dcond, pbc = e.ln_cond_bwd(dcn, c2, stc, wc.contiguous())
    dbs = colsum(pbs) if has_bs else pbs.new_empty(0)
    return dx, dcond, colsum(pbc), dwcat[:d].clone(), dbs, dwcat[d:].clone()


class _AdaLNTrain(torch.autograd.Function):
    """``adaln_rows``' autograd function with the wgmma forward (``fwd_saved``: ``ln_rows`` + one GEMM with the gate and the LayerNorm in its epilogue)
    in place of ``ln_cond_fwd`` -> cuBLAS -> ``adaln_fwd``; the saved tensors and the backward are the same."""

    @staticmethod
    def forward(ctx, x, cond, wc, ws, bs, wb, eps_x, eps_c):  # noqa: D102
        d, dc = x.shape[-1], cond.shape[-1]
        x2, c2 = x.reshape(-1, d).contiguous(), cond.reshape(-1, dc).contiguous()
        y, cn, stc, stx, sb = fwd_saved(x2, c2, wc, ws, bs if bs is not None else ws.new_zeros(d), wb, eps_x, eps_c, pick_ht(x2.shape[0], d, num_sms(x2.device)))
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(x2, c2, cn, stc, stx, sb, wc, ws, wb, *([] if bs is None else [bs]))
            ctx.shape, ctx.cshape, ctx.has_bs = x.shape, cond.shape, bs is not None
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        x2, c2, cn, stc, stx, sb, wc, ws, wb, *rest = ctx.saved_tensors
        bs = rest[0] if ctx.has_bs else None
        dy2 = dy.reshape(-1, x2.shape[-1]).contiguous()
        dx, dcond, dwc, dws, dbs, dwb = train_bwd(dy2, x2, c2, cn, stc, stx, sb, wc, ws, bs, wb, ctx.has_bs)
        return dx.view(ctx.shape), dcond.view(ctx.cshape), dwc, dws, (dbs if ctx.has_bs else None), dwb, None, None


def adaln_train(x, cond, lnw, w_scale, b_scale, w_bias, eps_x=1e-5, eps_cond=1e-5):
    """Training AdaLN (forward + backward) on the sm_90a wgmma forward; bf16 only (the fp32 forward stays on the row kernels + cuBLAS of ``adaln()``)."""
    return _AdaLNTrain.apply(x, cond, lnw, w_scale, b_scale, w_bias, eps_x, eps_cond)

__all__ = ["adaln_inference", "adaln_train", "colsum", "ext", "fwd_saved", "num_sms", "pack_weights", "pick_ht", "supported", "train_bwd"]

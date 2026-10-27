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
    return (x.new_empty((m, nx)), cond.new_empty(cond.shape), f(), f(), x.new_empty((m, 2 * nx)),
            x.new_empty((2 * nx, cond.shape[1]) if x.dtype is torch.bfloat16 else (0,)))


@opaque(fake=_fwd_saved_fake, name="adaln_cuda_train_fwd")
def fwd_saved(x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, w_scale: torch.Tensor, b_scale: torch.Tensor, w_bias: torch.Tensor,
              eps_x: float, eps_cond: float, ht: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The training forward in the layout of ``adaln_rows``' backward: (y, LN(cond) lnw, its (mean, rstd), the (mean, rstd) of x, sb) with the
    scale pre-activation in sb[:, :NX] (the second half is not written: the backward does not read it), and the packed [Ws; Wb] (bf16; an empty tensor
    for fp32) that the backward reuses. The weights are cast and concatenated at every call (they change between steps): bf16 by the row kernels' one-launch pack."""
    e = ext()
    m, nx = x.shape
    if x.dtype is torch.bfloat16:
        from . import _ext as rows_ext

        wc = torch.empty(2 * nx, cond.shape[1], device=x.device, dtype=x.dtype)
        bcat = torch.empty(2 * nx, device=x.device, dtype=x.dtype)
        rows_ext().pack(w_scale.contiguous(), w_bias.contiguous(), b_scale.contiguous(), wc, bcat)
        bs = bcat[:nx].float()
    else:
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
    return y, aff, stc, stx, sb, (wc if x.dtype is torch.bfloat16 else wc.new_empty(0))





def _colsum_fake(parts):
    """[N]: the column sums of [P, N]."""
    return parts.new_empty((parts.shape[1],))


@opaque(fake=_colsum_fake, name="adaln_cuda_colsum")
def colsum(parts: torch.Tensor) -> torch.Tensor:
    """sum over the rows of ``parts`` [P, N] fp32 in a fixed order (one launch)."""
    return ext().colsum(parts)


def train_supported(x: torch.Tensor) -> bool:
    """The wgmma training forward: bf16 (fp32 training is served by the row kernels + cuBLAS)."""
    return x.dtype == torch.bfloat16


def _colsum2_fake(a, b):
    """The column sums of a [Pa, Na] and b [Pb, Nb]."""
    return a.new_empty((a.shape[1],)), b.new_empty((b.shape[1],))


@opaque(fake=_colsum2_fake, name="adaln_cuda_colsum2")
def colsum2(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``colsum`` of two partial-sum buffers in one launch."""
    oa, ob = ext().colsum2(a, b)
    return oa, ob

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


def pack_weights(w_scale: torch.Tensor, b_scale: torch.Tensor, w_bias: torch.Tensor, ht: int, dtype: torch.dtype):
    """([Wscale | Wbias] per tile of ``ht`` output columns, [2 NX, NC] in ``dtype`` (fp32: on the TF32 grid), the scale bias as fp32)."""
    nx, nc = w_scale.shape
    ws, wb = w_scale.to(dtype), w_bias.to(dtype)
    wp = torch.stack([ws.reshape(nx // ht, ht, nc), wb.reshape(nx // ht, ht, nc)], 1).reshape(2 * nx, nc)
    if dtype == torch.float32:
        wp = rna_tf32(wp)
    return wp.contiguous(), b_scale.to(dtype).float().contiguous()


def packed(w_scale, b_scale, w_bias, ht, dtype):
    """``pack_weights`` cached while the parameters are unmodified (``_capture.param_forms``)."""
    return _capture.param_forms(
        f"adaln_cuda_{dtype}_{ht}", (w_scale, b_scale, w_bias), lambda: pack_weights(w_scale, b_scale, w_bias, ht, dtype)
    )


def _infer_fake(x, cond, lnw, wp, bs, eps_x, eps_cond, ht, rna):
    return torch.empty_like(x)


@opaque(fake=_infer_fake, name="adaln_cuda_inference")
def _infer(x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, wp: torch.Tensor, bs: torch.Tensor, eps_x: float, eps_cond: float, ht: int,
           rna: bool) -> torch.Tensor:
    """x [M, NX], cond [M, NC] contiguous; ``wp`` / ``bs`` from :func:`pack_weights`."""
    e = ext()
    m, nx = x.shape
    aff = torch.empty_like(cond)
    stats = torch.empty(2, m, device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    e.ln_rows(cond, lnw, x, aff, stats[0], stats[1], None, None, eps_cond, eps_x)
    e.adaln_gemm(aff, wp, bs, x, stats[0], stats[1], y, None, num_sms(x.device), ht, rna, True)
    return y


def adaln_inference(x, cond, lnw, w_scale, b_scale, w_bias, eps_x, eps_cond, *, rna=False, ht=None):
    """AdaLN forward without saving anything for a backward, ``x`` ``(..., NX)`` and ``cond`` ``(..., NC)`` of one dtype (bf16 / fp32 under TF32);
    the three projections in any dtype (cast and packed once per weight pack). ``rna``: round the fp32 result to the TF32 grid (its consumer is a
    TF32 GEMM)."""
    shape = x.shape
    nx, nc = shape[-1], cond.shape[-1]
    x2 = x.reshape(-1, nx).contiguous()
    c2 = cond.reshape(-1, nc).contiguous()
    ht = ht or pick_ht(x2.shape[0], nx, num_sms(x.device))
    wp, bs = packed(w_scale, b_scale, w_bias, ht, x.dtype)
    y = _infer(x2, c2, lnw.float().contiguous(), wp, bs, float(eps_x), float(eps_cond), ht, rna)
    return y.reshape(shape)


__all__ = ["adaln_inference", "ext", "num_sms", "pack_weights", "pick_ht", "supported"]

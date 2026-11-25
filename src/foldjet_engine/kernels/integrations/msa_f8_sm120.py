"""The RTX 5090 (sm_120a) extension of the opt-in e4m3 OuterProductMean / MSA pair-weighted-averaging routes (``csrc/sm120/msa_f8_sm120.cu``):
``transpose_quant`` (bf16 [B, R, N] -> e4m3 [B, N, Rp], static scale, amax guard), ``w_quant`` (e4m3 of 256 w), ``ln_vt8`` (LayerNorm + value projection
straight into the K-major e4m3 vT of the PWA contraction). Loaded only by ``opm_fp8_sm120`` / ``pwa_fp8_sm120`` when their switch is on."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

E4M3 = torch.float8_e4m3fn
_dir = Path(__file__).parent / "csrc" / "sm120"


@functools.lru_cache(maxsize=1)
def ext():
    from foldjet_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension
    from foldjet_engine.kernels._sm120_builds import SM120

    ensure_cuda_home()
    return load_extension(
        name="msa_f8_sm120",
        sources=[str(_dir / "msa_f8_sm120.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", SM120, "-DMWE_SM120"],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def transpose_quant(x: torch.Tensor, q: torch.Tensor, amax: torch.Tensor, rows_pad: int | None = None) -> torch.Tensor:
    """x bf16 [B, R, N] (or [R, N]) -> e4m3 [B, N, Rp] (or [N, Rp]) = x^T q (saturating), Rp = R rounded up to 16 (zero rows past R); max |x q| into amax."""
    x3 = x if x.dim() == 3 else x.unsqueeze(0)
    b, r, n = x3.shape
    rp = rows_pad if rows_pad is not None else -(-r // 16) * 16
    y = torch.empty((b, n, rp), device=x.device, dtype=E4M3)
    ext().transpose_quant(x3, y, q, amax)
    return y if x.dim() == 3 else y[0]


def w_quant(w: torch.Tensor, scale: float = 256.0) -> torch.Tensor:
    out = torch.empty(w.shape, device=w.device, dtype=E4M3)
    ext().w_quant(w, out, float(scale))
    return out


def vt8_y(y, wv, vinv, amax, depth: int, length: int) -> torch.Tensor:
    """The BIG width: y [S N, 256] bf16 (LayerNorm output), Wv [256, 256] -> e4m3 vT [8, S 32, N] = (y Wv^T)^T * vinv; max |v vinv| into amax."""
    vt = torch.empty((8, depth * 32, length), device=y.device, dtype=E4M3)
    ext().vt8_y(y, wv, vinv, amax, vt, int(depth), int(length))
    return vt


def ln_vt8(m, lnw, lnb, wv, eps, vinv, amax, nt: int = 64, blocks_per_sm: int = 2) -> torch.Tensor:
    """m [S, N, D] bf16 -> e4m3 vT [8, S C, N] = (LN(m) Wv^T)^T * vinv (C = Wv rows / 8); max |v vinv| into amax."""
    s, n, _ = m.shape
    c = wv.shape[0] // 8
    vt = torch.empty((8, s * c, n), device=m.device, dtype=E4M3)
    ext().ln_vt8(m, lnw, lnb, wv, float(eps), vinv, amax, vt, int(nt), int(blocks_per_sm))
    return vt

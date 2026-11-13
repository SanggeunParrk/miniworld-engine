"""Chai-1 OuterProductMean (``groups = 8``, d_hidden 64, product LayerNorm) on A100 (sm_80): ``sm80/chai_sm80.cu``, the passes of the H100 ``chai.cu``
on mma.sync / cp.async, with the same entry points and contracts as ``cuda.chai_proj`` / ``chai_epilogue`` / ``chai_bwd`` / ``chai_projbwd`` (the
driver ``outer_product_mean/chai.py`` picks the build by the card)."""

from __future__ import annotations

import functools
from pathlib import Path

_dir = Path(__file__).parent / "sm80"


@functools.lru_cache(maxsize=None)
def ext():
    """``chai_sm80.cu``, sm_80. Built on first use (or prebuilt: ``kernels/integrations`` plans it for the grouped OPM rows)."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
    from miniworld_engine.kernels import _msa_card as _card

    ensure_cuda_home()
    return load_extension(
        name=_card.name("opm_chai_sm80"),
        sources=[str(_dir / "chai_sm80.cu")],
        extra_include_paths=[str(_dir)],
        extra_cuda_cflags=_card.cflags([*host_flags(), "-O3", "-std=c++17", *gencodes("80")]),
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def chai_epilogue(o, wfold, bias, residual=None, save=False, eps=0.1):
    return ext().chai_epilogue(o, wfold, bias, residual, bool(save), float(eps))


def chai_proj(x, mask, gamma, beta, w, eps=1e-5, save=False):
    return ext().chai_proj(x, mask, gamma, beta, w, float(eps), bool(save))


def chai_bwd(dy, y, stats, n):
    return ext().chai_bwd(dy, y, stats, int(n))


def chai_projbwd(x, mask, da, db, gamma, beta, stats, wlr):
    return ext().chai_projbwd(x, mask, da, db, gamma, beta, stats, wlr)


def chai_products(a, b, mask, s, n):
    """O [8, N K, N K] = a_g^T b_g per group (a, b [8, S N, 8]): with a mask the CUTLASS product that skips the tokens masked in every MSA row
    (``ogemm_sm80``; their blocks are exactly 0), one launch per group, from L 384; else the cuBLAS batched GEMM."""
    import torch

    if mask is None or n < 384:  # short lengths: the eight launches cost more than the skipped blocks save (measured L128 1.4x)
        cfg = _products_cfg(s, n) if _sm120() else -1  # the RTX 5090: the CUTLASS batched GEMM where it beats cuBLAS (-1: cuBLAS)
        if cfg >= 0:
            return _gemm_sm120().mm(a.view(8, s, n * 8), b.view(8, s, n * 8), True, False, cfg)
        return torch.bmm(a.view(8, s, n * 8).transpose(1, 2), b.view(8, s, n * 8))
    from . import ogemm_sm80

    live = mask.view(s, n).to(torch.uint8).amax(0).contiguous()
    o = a.new_empty((8, n * 8, n * 8))
    for g in range(8):
        ogemm_sm80.outer(a[g].view(s, n * 8), b[g].view(s, n * 8), live, 8, o[g])
    return o


def chai_bwd_products(a, b, do, s, n):
    """(dA, dB) [8, S, N K] = (b_g do_g^T, a_g do_g) per group (a, b [8, S N, 8], do [8, N K, N K]): cuBLAS batched GEMMs on the A100, the
    CUTLASS batched GEMM on the RTX 5090."""
    import torch

    a3, b3 = a.view(8, s, n * 8), b.view(8, s, n * 8)
    cfg = _bwd_cfg(s, n) if _sm120() else -1
    if cfg >= 0:  # the RTX 5090: the CUTLASS batched GEMM where it beats cuBLAS
        g = _gemm_sm120()
        do = do.contiguous()
        return g.mm(b3, do, False, True, cfg), g.mm(a3, do, False, False, cfg)
    return torch.bmm(b3, do.transpose(1, 2)), torch.bmm(a3, do)


def _sm120() -> bool:
    from miniworld_engine.kernels import _msa_card

    return _msa_card.sm120()


def _gemm_sm120():
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import gemm_sm120

    return gemm_sm120


def _products_cfg(s: int, n: int) -> int:
    """The RTX 5090's tile of O = a_g^T b_g (``gemm_sm120.CFGS``; M = N = 8 N, K = S) where the masked product does not serve, measured at S 384 /
    1024: cuBLAS (-1) at S >= 1024 to N 256 (its 64 x 64 kernel leads there), 64 x 128 x 64 at N 128, 128 x 128 below N 384, 256 x 128 past it."""
    if s >= 1024 and n <= 256:
        return -1
    return 4 if n <= 128 else 0 if n < 384 else 2


def _bwd_cfg(s: int, n: int) -> int:
    """The RTX 5090's tile of dA = b_g do_g^T / dB = a_g do_g (M = S, N = K = 8 N): cuBLAS (-1) at S >= 1024 to N 256, 64 x 128 x 64 to N 384,
    128 x 128 x 64 past it."""
    if s >= 1024 and n <= 256:
        return -1
    return 4 if n <= 384 else 3

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

    ensure_cuda_home()
    return load_extension(
        name="opm_chai_sm80",
        sources=[str(_dir / "chai_sm80.cu")],
        extra_include_paths=[str(_dir)],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("80")],
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
        return torch.bmm(a.view(8, s, n * 8).transpose(1, 2), b.view(8, s, n * 8))
    from . import ogemm_sm80

    live = mask.view(s, n).to(torch.uint8).amax(0).contiguous()
    o = a.new_empty((8, n * 8, n * 8))
    for g in range(8):
        ogemm_sm80.outer(a[g].view(s, n * 8), b[g].view(s, n * 8), live, 8, o[g])
    return o

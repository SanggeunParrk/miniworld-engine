"""CUDA pass of the composed OuterProductMean (``gather.cu``), sm_90a. Built on first use (``load_extension``), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="opm_gather_sm90a",
        sources=[str(_dir / "gather.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def gather(prod, groups, k, weight=None, bias=None, eps=0.1):
    """prod [B G, N K, N K] bf16 -> [B N N, G K K] bf16 rows (g, k, l) [, LayerNorm * weight + bias over them]."""
    return ext().opm_gather(prod, int(groups), int(k), weight, bias, float(eps))


def split_groups(x, b, s, n, groups, k):
    """x [B S N, G K] bf16 (the projection rows) -> [B G, S, N K] bf16: per group, the rows' K channels in row order."""
    return ext().opm_split_groups(x, int(b), int(s), int(n), int(groups), int(k))


def bias_finish(out, c1, c2, count, c):
    """In place: out += (c1 + c2) / max(count, 1) + (count > 0) c (out bf16 [..., F]; c1 / c2 fp32 like it; count fp32 per row; c fp32 [F])."""
    ext().opm_bias_finish(out, c1, c2, count, c)

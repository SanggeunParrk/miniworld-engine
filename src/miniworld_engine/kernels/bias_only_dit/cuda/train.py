"""CUDA row kernels of the bias-only token DiT TRAINING block (``bias_only_dit_train_rows.cu``), forward and backward.

Built on first use (``load_extension``), never at import. The per-column gradient sums leave as a [partial_rows(M), 768]
buffer per sum (one row per persistent block; the kernels return how many they wrote), summed by ``colsum_cuda``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_train_rows_cuda",
        sources=[str(_dir / "bias_only_dit_train_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes("80", "90", "100", ptx=("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


def partials(M: int, n: int, device) -> torch.Tensor:
    """[n, partial_rows(M), 768]: n per-column sums, each written one row per block."""
    return torch.empty(n, ext().partial_rows(M), 768, device=device)


__all__ = ["ext", "partials"]

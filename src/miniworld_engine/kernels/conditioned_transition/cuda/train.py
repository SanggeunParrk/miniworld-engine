"""CUDA row kernels of the fused token DiT TRAINING block (``token_dit_train_rows.cu``), forward and backward.

Built on first use (``load_extension``), never at import. Entry points take the operands in the layouts
``integrations/token_dit_train.py`` keeps them: residual stream fp32 [M, 768], every GEMM operand in the path's dtype (bf16
or fp32), per-column bias / weight gradient sums as per-block partial rows that the host sums.
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
        name="token_dit_train_rows_cuda",
        sources=[str(_dir / "token_dit_train_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes("80", "90", "100", ptx=("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


RPB = 8          # rows per block of the D-wide kernels (token_dit_train_rows.cu)


def partials(M: int, width: int, device) -> torch.Tensor:
    """[blocks, width] buffer for the per-block column sums of a kernel over M rows (every row of it is written)."""
    return torch.empty((M + RPB - 1) // RPB, width, device=device)


__all__ = ["RPB", "ext", "partials"]

"""Own A100 all-five weight-gradient GEMM and FP32 reduction."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="swa_wgrad_grouped5_sm80",
        sources=[str(Path(__file__).parent / "sm80" / "wgrad_grouped5.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


def grouped_weight_gradients5(
    dp: torch.Tensor,
    x: torch.Tensor,
    dab: torch.Tensor,
    y: torch.Tensor,
    datt: torch.Tensor,
    gated: torch.Tensor,
    dffn: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """All five unrounded FP32 gradients, each with fresh independent storage."""
    return tuple(_ext().compute(dp, x, dab, y, datt, gated, dffn, hidden))

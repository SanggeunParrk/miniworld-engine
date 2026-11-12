"""Own A100 grouped weight gradients with C++ workspace/output allocation."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags
from miniworld_engine.kernels._sm120 import sm80_load_extension as load_extension  # sm_120a: same sources (kernels/_sm120.py)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="swa_wgrad_grouped8_alloc_sm80",
        sources=[str(Path(__file__).parent / "sm80" / "wgrad_grouped8_alloc.cu")],
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


def grouped_weight_gradients8_alloc(
    dp: torch.Tensor,
    x: torch.Tensor,
    dab: torch.Tensor,
    y: torch.Tensor,
    datt: torch.Tensor,
    gated: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Unrounded FP32 gradients, with four fresh independent output tensors."""
    return tuple(_ext().compute(dp, x, dab, y, datt, gated))

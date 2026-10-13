"""Own A100 eight-warp grouped QKV/G, up and output weight gradients."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="swa_wgrad_grouped8_sm80",
        sources=[str(Path(__file__).parent / "sm80" / "wgrad_grouped8.cu")],
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


def grouped_weight_gradients8(
    dp: torch.Tensor,
    x: torch.Tensor,
    dab: torch.Tensor,
    y: torch.Tensor,
    datt: torch.Tensor,
    gated: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Unrounded FP32 gradients; all four outputs have independent fresh storage."""
    partial = torch.empty(
        (2 * 32 * 512 * 128 + 32 * 128 * 128,), device=dp.device, dtype=torch.float32
    )
    qkv = torch.empty((384, 128), device=dp.device, dtype=torch.float32)
    gate = torch.empty((128, 128), device=dp.device, dtype=torch.float32)
    up = torch.empty((512, 128), device=dp.device, dtype=torch.float32)
    output = torch.empty((128, 128), device=dp.device, dtype=torch.float32)
    _ext().run(dp, x, dab, y, datt, gated, partial, qkv, gate, up, output)
    return qkv, gate, up, output

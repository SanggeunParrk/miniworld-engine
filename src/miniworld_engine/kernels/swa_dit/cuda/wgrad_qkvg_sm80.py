"""Own merged QKV/gate gradient GEMM and fresh reduce-scatter on A100."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="swa_wgrad_qkvg_sm80",
        sources=[str(Path(__file__).parent / "sm80" / "wgrad_qkvg.cu")],
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


def qkvg_weight_gradient(
    dp: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unrounded FP32 gradients in separate fresh allocations."""
    partial = torch.empty((16, 512, 128), device=dp.device, dtype=torch.float32)
    qkv = torch.empty((384, 128), device=dp.device, dtype=torch.float32)
    gate = torch.empty((128, 128), device=dp.device, dtype=torch.float32)
    _ext().run(dp, x, partial, qkv, gate)
    return qkv, gate

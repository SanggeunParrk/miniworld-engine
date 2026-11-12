"""Own A100 CUDA finalization of the five block weight gradients."""

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
        name="swa_grad_finalize_sm80",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


def finalize(grads: list[torch.Tensor], bf16_mask: int) -> list[torch.Tensor]:
    """Fresh outputs: BF16 nearest rounding or bit-preserving FP32 copies."""
    return _ext().finalize(grads, bf16_mask)

"""Isolated own-CUDA exact OPM mask count candidate, no default dispatch."""

import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="opm_mask_count_sm80",
        sources=[str(Path(__file__).parent / "sm80/mask_count.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def result(mask, clamp=True, cfg=0):
    S, L = mask.shape
    packed = torch.empty(((S + 31) // 32, L), device=mask.device, dtype=torch.int32)
    valid = torch.empty((L,), device=mask.device, dtype=torch.uint8)
    norm = torch.empty((L, L), device=mask.device, dtype=torch.float32)
    _ext().run(mask, packed, valid, norm, clamp, cfg)
    return norm

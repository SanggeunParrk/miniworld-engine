"""Isolated own-CUDA exact OPM mask count candidate, no default dispatch."""

import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="opm_mask_count_ballot_sm80",
        sources=[str(Path(__file__).parent / "sm80/mask_count_ballot.cu")],
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
    return _ext().result(mask, clamp, cfg)

"""Isolated own-CUDA PWA pair-weight gradient contraction; no production dispatch."""

import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name="pwa_dw_fold_sm80",
        sources=[str(directory / "dw_fold.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def result(a, b, replicas=8, cfg=0):
    return _ext().result(a, b, replicas, cfg)

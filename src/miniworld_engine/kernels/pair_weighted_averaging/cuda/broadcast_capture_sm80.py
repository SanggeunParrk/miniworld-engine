"""Experimental C++ allocation, preserving old native CUDA outside capture."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    ext = load_extension(
        name="pwa_broadcast_capture_sm80",
        sources=[
            str(directory / "broadcast_capture_old.cu"),
            str(directory / "broadcast_capture_pipeline.cu"),
        ],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )
    ext.prepare()
    return ext


def result(w, v, transpose=False, mode=1):
    return _ext().result(w, v, transpose, mode)

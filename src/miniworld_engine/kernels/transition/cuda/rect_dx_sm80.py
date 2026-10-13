"""Isolated own-CUDA full-width GEMM candidate; not a production dispatch."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name="transition_rect_dx_sm80",
        sources=[str(directory / "rect_dx_gemm.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_80,code=sm_80",
            f"-I{directory}",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def run(a, b, out, stages=4, grid=0):
    _ext().run(a, b, out, stages, grid)

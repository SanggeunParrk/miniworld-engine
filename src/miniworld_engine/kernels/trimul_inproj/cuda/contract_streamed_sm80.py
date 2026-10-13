"""Candidate own-CUDA plane GEMM; callers retain their measured default until validation passes."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name="trimul_contract_streamed_sm80",
        sources=[str(directory / "contract_streamed.cu")],
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


def run(a, b, out, h, *, mode=-1, config=0, grid=0):
    """Write square BF16 planes: mode -1 mixed NT/TN, 0 NT, 1 TN, 2 NN, 3 TT."""
    _ext().run(a, b, out, h, mode, config, grid)

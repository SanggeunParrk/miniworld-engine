"""Experimental explicit nearest-rounded TF32 attention; old extension preserved."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80_tf32_qk_rne"
    return load_extension(
        name="augmented_attention_tf32_qk_rne_sm80",
        sources=[str(directory / "ops_tf32.cu")],
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

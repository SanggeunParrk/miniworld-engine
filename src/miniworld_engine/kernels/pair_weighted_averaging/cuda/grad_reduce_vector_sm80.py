"""E115: deterministic CUDA partial reduction and dtype finalization, opt-in."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent / "sm80"
    return load_extension(
        name="pwa_grad_reduce_vector_sm80",
        sources=[str(d / "grad_reduce_vector.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def result(glue, dgv, pm, ps, gamma, beta, wb, dtype_bits):
    return _ext().result(glue, dgv, pm, ps, gamma, beta, wb, 64, 256, 128, dtype_bits)

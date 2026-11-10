"""Own-CUDA batched FP32 master parameter casts; BF16 inputs retain their tensors."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent
    return load_extension(
        name="parameter_cast_sm80",
        sources=[str(directory / "parameter_cast_sm80.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def result(parameters):
    return tuple(_ext().result(list(parameters)))


def result_wab(wa, wb, ws):
    """(wab [2H, D] bf16 = [wa; wb], ws bf16) from fp32 masters in one launch."""
    return tuple(_ext().result_wab(wa, wb, ws))

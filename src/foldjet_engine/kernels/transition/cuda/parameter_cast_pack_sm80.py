"""Own CUDA RNE parameter casts and distinct packed projection copies."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent
    return load_extension(
        name="parameter_cast_pack_sm80",
        sources=[str(directory / "parameter_cast_pack_sm80.cu")],
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


def result(parameters, pairs):
    outputs = list(_ext().result(list(parameters), [list(pair) for pair in pairs]))
    return tuple(outputs[: len(parameters)]), tuple(outputs[len(parameters) :])

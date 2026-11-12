"""Separate own-CUDA typed OPM gradient finalizer candidate."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from miniworld_engine.kernels import _msa_card as _card


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name=_card.name("opm_finalize_typed_sm80"),
        sources=[str(Path(__file__).parent / "sm80/finalize_typed.cu")],
        extra_cuda_cflags=_card.cflags([
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ]),
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def result(sources, parameters):
    return _ext().result(sources, parameters)

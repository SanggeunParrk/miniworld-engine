"""A100 (sm_80) builds of the bidirectional bias-only TriangleAttention's kernels (``kernels/integrations/bidir_triattn_h100``, which serves
both cards): the width-generic glue (``h100_sources/triattn_generic_sm90.cu``: LayerNorm forward / backward, the gate) and the gate's
backward on channel planes (``h100_sources/triattn_bidir_sm90.cu``), each source compiled unchanged for sm_80 (plain CUDA-core kernels:
the sm_90-only helpers of ``sm90.cuh`` it includes are never called). Built on first use, never at import."""

from __future__ import annotations

import functools
from pathlib import Path

from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

from . import sm80_arch as _arch

_SRC = Path(__file__).parent / "h100_sources"
_FLAGS = ("-std=c++17", "-O3", "--use_fast_math", "-DNDEBUG")


@functools.lru_cache(maxsize=None)
def glue():
    ensure_cuda_home()
    return load_extension(
        name=_arch.name("triattn_generic_sm80"),
        sources=[str(_SRC / "triattn_generic_sm90.cu")],
        extra_cuda_cflags=[*host_flags(), *_FLAGS, *_gencodes()],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def ext():
    ensure_cuda_home()
    return load_extension(
        name=_arch.name("triattn_bidir_sm80"),
        sources=[str(_SRC / "triattn_bidir_sm90.cu")],
        extra_cuda_cflags=[*host_flags(), *_FLAGS, *_gencodes()],
        verbose=False,
    )


def _gencodes() -> list[str]:
    """sm_80, or sm_120a on the RTX 5090 (``sm80_arch``)."""
    return gencodes("120a") if _arch.sm120_build() else gencodes("80")

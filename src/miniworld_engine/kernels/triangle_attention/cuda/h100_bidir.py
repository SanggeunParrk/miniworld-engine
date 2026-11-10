"""H100 (sm_90a) glue of the bidirectional bias-only TriangleAttention (``kernels/integrations/bidir_triattn_h100``):
``h100_sources/triattn_bidir_sm90.cu`` (the gate's backward on channel planes). The path's LayerNorm / gate forward are the width-generic
kernels of ``h100_sm90_generic._glue``."""

from __future__ import annotations

import functools
from pathlib import Path

from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_SRC = Path(__file__).parent / "h100_sources"


@functools.lru_cache(maxsize=None)
def ext():
    ensure_cuda_home()
    return load_extension(
        name="triattn_h100_bidir",
        sources=[str(_SRC / "triattn_bidir_sm90.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), "--use_fast_math", "-DNDEBUG"],
        verbose=False,
    )

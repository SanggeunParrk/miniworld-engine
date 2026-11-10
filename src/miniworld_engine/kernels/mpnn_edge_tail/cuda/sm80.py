"""Loader of the sm_80 MPNN edge extension (``sm80/ops.cu``: weight packing, the edge-tail chain kernels, the edge-MLP, LayerNorm and dropout kernels).

Built on first use, never at import (or taken from the prebuilt / JIT cache): ``ext()`` raises when the toolchain cannot build it (the callers in
``integrations/mpnn_edge_sm80.py`` turn that into a refusal of the ``cuda`` backend).
"""

from __future__ import annotations

import functools
from pathlib import Path

_dir = Path(__file__).parent / "sm80"


@functools.lru_cache(maxsize=1)
def ext():
    from miniworld_engine.kernels._nvcc import (
        ensure_cuda_home,
        gencodes,
        host_flags,
        load_extension,
    )

    ensure_cuda_home()
    flags = [
        *host_flags(),
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        *gencodes("80"),
    ]
    return load_extension(
        name="mpnn_edge_sm80",
        sources=[str(_dir / "ops.cu")],
        extra_include_paths=[str(_dir)],
        extra_cuda_cflags=flags,
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )

"""H100 TriangleAttention parameter-pack / dropout-mask extension (``h100_sources/pack.cu``).

The attention forward / backward of the H100 TriangleAttention are in ``h100_sm90`` / ``h100_sm90_generic`` / ``h100_dh``.
What remains here is the one build ``integrations/pwa_train`` uses: the Philox keep-mask of the MSA row dropout (``drop_keep``).
"""

from __future__ import annotations

import functools
from pathlib import Path

from miniworld_engine.kernels._nvcc import (
    cutlass_include,
    ensure_cuda_home,
    gencodes,
    host_flags,
    load_extension,
)

_SRC = Path(__file__).with_name("h100_sources")
_PARENT = _SRC.parent


@functools.lru_cache(maxsize=None)
def extension(part: str):
    if part != "pack":
        raise ValueError(part)
    ensure_cuda_home()
    # CUTLASS 4.2 headers are needed only to build (prebuilt / cached artifacts load without; _nvcc raises CutlassMissing)
    cutlass_inc = cutlass_include()
    return load_extension(
        name="triattn_h100_b200style_pack_v1",
        sources=[str(_SRC / "pack.cu")],
        extra_include_paths=[str(_PARENT), str(cutlass_inc)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            *host_flags(),
            *gencodes("90a"),
            "-O3",
            "-std=c++17",
            "-DMW_TRIATTN_KV_TILE=64",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
        ],
    )

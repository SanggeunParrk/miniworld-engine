"""H100 (sm_90a) glue of the bidirectional bias-only TriangleAttention (``kernels/integrations/bidir_triattn_h100``):
``h100_sources/triattn_bidir_sm90.cu`` (the gate's backward on channel planes, plain and fused with the du GEMM; the softmax-logits
pass; the softmax gradient's last step; the fused gate + out-projection tail; the planes GEMM through a bit-equal cublasLt algorithm;
the weight cast). The path's LayerNorm / gate forward are the width-generic kernels of ``h100_sm90_generic._glue``."""

from __future__ import annotations

import functools
from pathlib import Path

from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_SRC = Path(__file__).parent / "h100_sources"


@functools.lru_cache(maxsize=None)
def ext():
    from miniworld_engine.kernels._codegraph import CUDA_HOME_STANDIN

    cuda_home = ensure_cuda_home() or CUDA_HOME_STANDIN  # the toolkit's headers are needed only to build
    return load_extension(
        name="triattn_h100_bidir2",
        sources=[str(_SRC / "triattn_bidir_sm90.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), "--use_fast_math", "-DNDEBUG"],
        extra_include_paths=[str(Path(cuda_home) / "targets" / "x86_64-linux" / "include")],
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )

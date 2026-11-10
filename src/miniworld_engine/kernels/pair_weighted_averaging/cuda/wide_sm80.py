"""A100 (sm_80) layout / gate kernels of the BIG-width MSA pair-weighted averaging (d_msa 256, 8 heads x 32; ``integrations/pwa_wide_sm80.py``):
``sm80/wide_ops.cu`` (see its header). One shape-independent extension, built on first use (``load_extension``), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"


@functools.lru_cache(maxsize=1)
def ext():
    ensure_cuda_home()
    return load_extension(
        name="pwa_wide_sm80",
        sources=[str(_dir / "wide_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("80")],
        extra_cflags=["-std=c++17", "-O3"],
        extra_ldflags=["-lcublas"],   # head_bmm: the per-head contraction on the natural layout
        verbose=False,
    )

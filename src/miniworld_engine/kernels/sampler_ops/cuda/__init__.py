"""CUDA kernels of the sampler glue (``sampler_ops.cu``: CUDA cores and programmatic dependent launch, no architecture-specific MMA), built
for the device: sm_90a on H100, sm_100a on B200. Built on first use (``load_extension``), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

_dir = Path(__file__).parent


def ext():
    import torch

    return _ext("90a" if torch.cuda.get_device_capability() == (9, 0) else "100a")


@functools.lru_cache(maxsize=None)
def _ext(arch: str):
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name=f"sampler_ops_sm{arch}",
        sources=[str(_dir / "sampler_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes(arch)],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )

"""The A100 (sm_80) build of the sampler glue kernels: ``sampler_ops_sm80.cu`` compiles ``sampler_ops.cu`` unchanged with its programmatic dependent
launch turned into plain launches (sm_80 has no ``griddepcontrol``). Built on first use (``load_extension``), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="sampler_ops_sm80",
        sources=[str(_dir / "sampler_ops_sm80.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("80"), f"-I{_dir}"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def device_ext():
    """The build for the current device: this one on an A100, else the H100 (sm_90a) build."""
    if torch.cuda.get_device_capability() == (8, 0):
        return ext()
    from miniworld_engine.kernels.sampler_ops.cuda import ext as h100_ext

    return h100_ext()

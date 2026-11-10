"""CUDA kernels of the sampler glue (``sampler_ops.cu``): the sm_90a build (H100, kernels chained by programmatic dependent launch) and the sm_80 build
(A100, ``-DSAMPLER_NO_PDL``: the same kernels launched plainly). Built on first use (``load_extension``), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent


def ext():
    """The build for the current device: sm_80 on an A100, else the sm_90a build."""
    return _ext_sm80() if torch.cuda.get_device_capability() == (8, 0) else _ext_sm90a()


@functools.lru_cache(maxsize=None)
def _ext_sm90a():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="sampler_ops_sm90a",
        sources=[str(_dir / "sampler_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def _ext_sm80():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="sampler_ops_sm80",
        sources=[str(_dir / "sampler_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-DSAMPLER_NO_PDL", *gencodes("80")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def prebuild(target, rows):
    """The build of the sampler glue family (kernels/_prebuild.py): one shape-independent extension per arch (``sampler_ops_sm90a`` / ``sampler_ops_sm80``,
    every size is a runtime launch parameter), built once when some requirements row (of ``AugmentNoise`` / ``AlignEuler`` / ``PoolAtoms``, the modules that
    launch it) needs the family."""
    if rows:
        yield _ext_sm80 if target.arch == "sm80" else _ext_sm90a

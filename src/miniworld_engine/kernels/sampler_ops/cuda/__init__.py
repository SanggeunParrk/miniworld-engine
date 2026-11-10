"""CUDA kernels of the sampler glue (``sampler_ops.cu``), sm_90a. Built on first use (``load_extension``), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="sampler_ops_sm90a",
        sources=[str(_dir / "sampler_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def prebuild(target, rows):
    """The build of the sampler glue family (kernels/_prebuild.py): one shape-independent extension (``sampler_ops_sm90a``, every size is a runtime launch
    parameter), built once when some requirements row (of ``AugmentNoise`` / ``AlignEuler`` / ``PoolAtoms``, the modules that launch it) needs the family."""
    if rows and target.arch == "sm80":  # the A100 build (sm80.py)
        from .sm80 import ext as sm80_ext

        yield sm80_ext
    elif rows:
        yield ext

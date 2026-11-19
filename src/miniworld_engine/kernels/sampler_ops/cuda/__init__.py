"""CUDA kernels of the sampler glue (``sampler_ops.cu``), sm_90a; the same source unchanged for sm_100a (B200: CUDA cores and programmatic
dependent launch, nothing Hopper-specific). Built on first use (``load_extension``), never at import."""

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


@functools.lru_cache(maxsize=None)
def ext_sm100():
    """The B200 (sm_100a) build: ``sampler_ops_sm100.cu`` = ``sampler_ops.cu`` (the H100 source, unchanged, PDL kept) + the B200 single-launch
    variants (2026-11-17: thread-block-cluster augment / align kernels, the pooling with every atom row in flight, the sort-free pooling backward)."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="sampler_ops_sm100a",
        sources=[str(_dir / "sampler_ops_sm100.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("100a"), f"-I{_dir}"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def prebuild(target, rows):
    """The build of the sampler glue family (kernels/_prebuild.py): one shape-independent extension (``sampler_ops_sm90a``, every size is a runtime launch
    parameter), built once when some requirements row (of ``AugmentNoise`` / ``AlignEuler`` / ``PoolAtoms``, the modules that launch it) needs the family."""
    if rows and target.arch in ("sm80", "sm120"):  # the A100 build (sm80.py; sm_120a on the RTX 5090: _sm120.py)
        from .sm80 import ext as sm80_ext

        yield sm80_ext
    elif rows and target.arch == "sm100":  # the B200 build of the H100 source
        yield ext_sm100
    elif rows:
        yield ext

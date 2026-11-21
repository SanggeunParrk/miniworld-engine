"""The B200 graph RNG prologue writer (``_graph_rng_sm100.cu``): the ONE ``aten::fill_.Scalar`` (CUDA) override that serves the default
generator's graph-owned seed / offset scalars of every graph holding a registered Philox draw with one programmatic launch.

The families that draw inside a graph -- TriMul (``trimul_inproj/cuda/b200_sources/wide_aux.cu``, the forms launch) and
TriangleAttention (``triangle_attention/cuda/b200_sources/pack_sm100.cu``: the unchanged H100 ``pack.cu`` built for sm_100a, its fill_
override and its capture registration routed here) -- register their captured scalars here through ``api()``: a C table of function
pointers each extension takes once after it is loaded (``connect``). One registry and one writer serve both families: the override
``pack.cu`` registers on its own is a forwarder into this writer's ``fill``, so whichever registration is last the same code runs.
Before (2026-11-17 part 6) each family registered its own override and the last registration won: the other's graphs fell back to
ATen's two fill launches.
"""

from __future__ import annotations

import functools
from pathlib import Path

#: 2: one programmatic launch (default); 1: one ordinary launch; 0: ATen's two fills (A / B, tests)
MODE = 2


@functools.lru_cache(maxsize=1)
def ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    e = load_extension(
        name="graph_rng_sm100_v1",
        sources=[str(Path(__file__).with_name("_graph_rng_sm100.cu"))],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("100a"), "-DNDEBUG"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )
    e.set_mode(MODE)
    return e


def set_mode(mode: int) -> None:
    """The writer's mode for every family (``MODE``); a no-op when unchanged."""
    global MODE
    if int(mode) != MODE:
        MODE = int(mode)
        ext().set_mode(MODE)


def connect(bind) -> None:
    """Hand the writer's function table (its address) to a family extension: ``bind`` is that extension's binding taking it."""
    bind(ext().api())

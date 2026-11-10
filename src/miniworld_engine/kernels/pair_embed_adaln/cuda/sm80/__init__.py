"""sm_80 (A100) build and launch of the pair-embedding + adaptive-LayerNorm kernels: ``pair_embed_adaln_sm80.cu`` compiles ``sm90/pair_embed_adaln.cu`` unchanged with
the sm_80 forms of its sm_90-only helpers (pair_embed_geom's ``pe_sm80.cuh``: the table gradient's warpgroup product as mma.sync). The launchers are the H100 ones
(``sm90/__init__.py``: the same arguments, scratch sizes and implementation choices) bound to this build. Built on first use, never at import."""

from __future__ import annotations

import functools
import sys
import types
from pathlib import Path

import torch

from miniworld_engine.kernels.pair_embed_adaln.cuda import sm90 as _h100

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(name="pair_embed_adaln_sm80", sources=[str(_dir / "pair_embed_adaln_sm80.cu")],
                          extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("80"), f"-I{_dir}"], extra_cflags=["-std=c++17"],
                          verbose=False)


def _on_this_build(fn):
    """The H100 launcher ``fn`` with ``ext`` resolved to this build."""
    return types.FunctionType(fn.__code__, {**vars(_h100), "ext": ext}, fn.__name__, fn.__defaults__, fn.__closure__)


forward = _on_this_build(_h100.forward)
backward = _on_this_build(_h100.backward)


def device_build():
    """The launchers of the current device's build: this one on an A100, else the H100 (sm_90a) one."""
    return sys.modules[__name__] if torch.cuda.get_device_capability() == (8, 0) else _h100

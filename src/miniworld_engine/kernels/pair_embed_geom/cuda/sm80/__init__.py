"""sm_80 (A100) build and launch of the backbone pair-embedding kernels: ``pair_embed_geom_sm80.cu`` compiles ``sm90/pair_embed_geom.cu`` unchanged with
the sm_80 forms of its sm_90-only helpers (``pe_sm80.cuh``: the table gradient's warpgroup product as mma.sync). The launchers are the H100 ones
(``sm90/__init__.py``: the same arguments, scratch sizes and implementation choices) bound to this build. Built on first use, never at import."""

from __future__ import annotations

import functools
import sys
import types
from pathlib import Path

import torch

from miniworld_engine.kernels.pair_embed_geom.cuda import sm90 as _h100

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(name="pair_embed_geom_sm80", sources=[str(_dir / "pair_embed_geom_sm80.cu")],
                          extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("80"), f"-I{_dir}"], extra_cflags=["-std=c++17"],
                          verbose=False)


#: the launchers' module globals: the H100 launcher module's, with ``ext`` this build (and this card's choices, set below)
_G = {**vars(_h100), "ext": ext}


def _on_this_build(fn):
    """The H100 launcher ``fn`` reading ``_G``."""
    return types.FunctionType(fn.__code__, _G, fn.__name__, fn.__defaults__, fn.__closure__)


forward = _on_this_build(_h100.forward)
forward_ca = _on_this_build(_h100.forward_ca)
ln_backward = _on_this_build(_h100.ln_backward)
#: the table gradient per mode on this card (measured L128-L384, L2 flushed): the encoder table (mode 0, 317 rows) on the warpgroup kernel
#: (impl 1, here mma.sync on its swizzled operands), the decoder's (mode 1, 157 rows) on the mma.sync kernel over KS concurrent pair chunks
#: (impl 3, pair_embed_geom_sm80.cu: the H100 grid of 132 CTAs left the A100's SMs too few warps)
BWD_IMPL = {0: 1, 1: 3}
KS = 2   # pair_embed_geom_sm80.cu


def backward(g, codes, mode):
    """Table gradient [N_ROWS[mode], 256] bf16 from d pair = g [B, n, n, 256] bf16 (the H100 launcher's arguments and scratch)."""
    nt = _h100.N_ROWS[mode]
    parts = KS * _h100.NBLK if BWD_IMPL[mode] == 3 else _h100.NBLK
    tpart = torch.empty(parts * nt * 256, device=g.device, dtype=torch.float32)
    dtbl = torch.empty(nt, 256, device=g.device, dtype=torch.bfloat16)
    ext().backward(g, codes, tpart, dtbl, int(mode), BWD_IMPL[mode])
    return dtbl


def device_build():
    """The launchers of the current device's build: this one on an A100, else the H100 (sm_90a) one."""
    return sys.modules[__name__] if torch.cuda.get_device_capability() == (8, 0) else _h100

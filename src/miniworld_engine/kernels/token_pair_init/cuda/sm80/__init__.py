"""sm_80 (A100) build and launch of the token-pair initialisation kernels: ``token_pair_init_sm80.cu`` compiles ``sm90/token_pair_init.cu`` unchanged
except its forward kernel (the class-row table by cp.async instead of the sm_90 bulk copy). The launchers are the H100 ones (``sm90/__init__.py``: the
same arguments, scratch sizes and rows per CTA) bound to this build. Built on first use, never at import."""

from __future__ import annotations

import functools
import sys
import types
from pathlib import Path

import torch

from miniworld_engine.kernels.token_pair_init.cuda import sm90 as _h100

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(name="token_pair_init_sm80", sources=[str(_dir / "token_pair_init_sm80.cu")],
                          extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("80"), f"-I{_dir}"], extra_cflags=["-std=c++17"],
                          verbose=False)


#: the launchers' module globals: the H100 launcher module's, with ``ext`` this build (and this card's choices, set below)
_G = {**vars(_h100), "ext": ext}


def _on_this_build(fn):
    """The H100 launcher ``fn`` reading ``_G``."""
    return types.FunctionType(fn.__code__, _G, fn.__name__, fn.__defaults__, fn.__closure__)


forward = _on_this_build(_h100.forward)
backward = _on_this_build(_h100.backward)
prep = _on_this_build(_h100.prep)


def device_build():
    """The launchers of the current device's build: this one on an A100, else the H100 (sm_90a) one."""
    return sys.modules[__name__] if torch.cuda.get_device_capability() == (8, 0) else _h100

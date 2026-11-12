"""The two cards the sm_80 Transition sources are built for: the A100 (``sm_80``) and the GeForce RTX 5090 (``sm_120a``, card key ``5090_sm120``).

The RTX 5090 runs the A100's mma.sync / ldmatrix / cp.async kernels unchanged, compiled for sm_120a into extensions of their own (the ``_sm120``
names); the A100 builds keep their names and flags, so their artifacts are untouched.  What differs is the shared memory: 99 KB a block (101376 B)
against the A100's 163 KB, so the kernels whose tile configs need more (the one-kernel fused forward / backward of ``fused_sm80``,
``fused_fwd_sm80``, ``fused_bwd_sm80``) are not served on the 5090: the wide chain (``fused_wide_sm80``) serves every call there.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch

from ..._compile import device_constant

SM80 = "-gencode=arch=compute_80,code=sm_80"
SM120 = "-gencode=arch=compute_120a,code=sm_120a"

#: Max dynamic shared memory per block of the RTX 5090 (bytes): the tile configs a launch there may pick.
SM120_SMEM = 101376


@device_constant
@functools.lru_cache(maxsize=8)
def _cc(index: int) -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(index))


def _index(device: torch.device | None) -> int:
    return torch.cuda.current_device() if device is None or device.index is None else device.index


def is_sm120(device: torch.device | None = None) -> bool:
    """The card of ``device`` (default: the current one) is a GeForce RTX 5090 (capability 12.0)."""
    if device is not None and device.type != "cuda":
        return False
    return _cc(_index(device)) == (12, 0)


def served(device: torch.device | None = None) -> bool:
    """The card is one the sm_80 sources have a build for: the A100 (8.0) or the RTX 5090 (12.0)."""
    if device is not None and device.type != "cuda":
        return False
    return _cc(_index(device)) in ((8, 0), (12, 0))


def per_arch(build: Callable[[str, str], object]):
    """``build(suffix, gencode)`` -> ``(ext, ext_sm80, ext_sm120)``: the two cached builds (``suffix`` ``"sm80"`` / ``"sm120"``) and the
    loader that picks the current card's (the RTX 5090's on capability 12.0, else the A100's)."""
    ext_sm80 = functools.lru_cache(maxsize=1)(lambda: build("sm80", SM80))
    ext_sm120 = functools.lru_cache(maxsize=1)(lambda: build("sm120", SM120))

    def ext():
        return ext_sm120() if is_sm120() else ext_sm80()

    return ext, ext_sm80, ext_sm120

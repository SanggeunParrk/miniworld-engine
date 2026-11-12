"""GeForce RTX 5090 (sm_120a, card ``5090_sm120``) builds of the engine's sm_80 sources: one implementation per op.

The A100 kernels (``mma.sync`` / ``ldmatrix`` / ``cp.async``) run unchanged on sm_120a, which has the same warp-level tensor-core
instructions (it has no ``wgmma`` / ``tcgen05``). A builder of an sm_80 extension imports :func:`sm80_load_extension` as its
``load_extension``: on an sm_120 device (or with ``MINIWORLD_ENGINE_BUILD_ARCH=sm120``, for a GPU-less prebuild) it compiles the
same sources for ``sm_120a`` under the extension name ``<name>_sm120`` (``-DMWE_SM120`` defined, for the few size guards the 99 KB
shared-memory limit of the card needs); everywhere else it is ``_nvcc.load_extension`` itself, so the sm_80 build and its key are
unchanged.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import torch

from miniworld_engine.kernels import _nvcc

ENV = "MINIWORLD_ENGINE_BUILD_ARCH"
#: RTX 5090: the opt-in dynamic shared memory per block (101376 B = 99 KB) and the SM count
SMEM_MAX = 101376
SMS = 170
_SM80 = "-gencode=arch=compute_80,code=sm_80"
_SM120 = "-gencode=arch=compute_120a,code=sm_120a"


@functools.cache
def _capability(index: int) -> tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(index)
    return major, minor


@torch.compiler.assume_constant_result
def is_sm120_index(index: int) -> bool:
    """Compute capability 12.0 (a per-device constant: cached, and a constant to ``torch.compile``)."""
    return _capability(index) == (12, 0)


def is_sm120(device: torch.device | None = None) -> bool:
    """A CUDA device of capability 12.0 (``None``: the current device)."""
    if device is not None and device.type != "cuda":
        return False
    index = None if device is None else device.index
    return is_sm120_index(torch.cuda.current_device() if index is None else index)


def building_sm120() -> bool:
    """Whether sm_80 builders compile for sm_120a now: ``MINIWORLD_ENGINE_BUILD_ARCH`` (``sm120`` / ``sm80``) if set, else the current device."""
    forced = os.environ.get(ENV, "").lower().removeprefix("sm").removeprefix("_").rstrip("a")
    if forced:
        return forced == "120"
    return torch.cuda.is_available() and is_sm120()


def _retarget(flags: list[Any]) -> list[Any]:
    from miniworld_engine.kernels._codegraph import GENCODE_MARKER, PTX_MARKER

    out: list[Any] = []
    for f in flags:
        s = str(f)
        if s == _SM80:
            out.append(_SM120)
        elif s == "-arch=sm_80":
            out.append("-arch=sm_120a")
        elif s == f"{GENCODE_MARKER}80":
            out.append(f"{GENCODE_MARKER}120a")
        elif s.startswith(PTX_MARKER):
            continue
        else:
            out.append(f)
    return [*out, "-DMWE_SM120"]


def sm80_load_extension(name: str, sources: list[str], **kwargs: Any) -> Any:
    """``_nvcc.load_extension`` of an sm_80 extension, built for sm_120a as ``<name>_sm120`` when :func:`building_sm120`."""
    if building_sm120():
        kwargs = dict(kwargs)
        kwargs["extra_cuda_cflags"] = _retarget(list(kwargs.get("extra_cuda_cflags") or []))
        name = f"{name}_sm120"
    return _nvcc.load_extension(name, sources, **kwargs)


def sm80_cubin_flags(flags: list[str]) -> tuple[str, list[str]]:
    """(stem suffix, flags) of an sm_80 ``_nvcc.cubin`` build, retargeted like :func:`sm80_load_extension`."""
    if building_sm120():
        return "_sm120", [str(f) for f in _retarget(flags)]
    return "", flags

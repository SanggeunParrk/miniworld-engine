"""Which card the A100 (sm_80) TriangleAttention sources are built for: the A100 itself, or the GeForce RTX 5090 (sm_120a), which runs the
same ``mma.sync`` / ``ldmatrix`` / ``cp.async`` kernels (no wgmma) with 99 KB of shared memory per block and 170 SMs.

One source per op: the 5090 build compiles the sm_80 sources for ``sm_120a`` under its own extension name (``*_sm120``), with
``TRIATTN_SM120`` defined so the few geometry knobs that must fit 99 KB of shared memory change behind that guard only (the sm_80 build
is byte-identical)."""

from __future__ import annotations

import os

import torch

#: max dynamic shared memory per block on the RTX 5090 (cudaDevAttrMaxSharedMemoryPerBlockOptin, measured)
SM120_SMEM = 101376


def _device_is_sm120() -> bool | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(torch.cuda.current_device()) == (12, 0)


def sm120_build() -> bool:
    """The build targets the 5090: ``MINIWORLD_ENGINE_BUILD_ARCH`` says so (the shared switch of every sm_80 builder, ``kernels/_sm120.py``; a
    GPU-less prebuild of ``_prebuild --arch sm120`` sets it), else the current device is one, or (no GPU) ``TORCH_CUDA_ARCH_LIST`` names 12.0 only."""
    forced = os.environ.get("MINIWORLD_ENGINE_BUILD_ARCH", "").lower().removeprefix("sm").removeprefix("_").rstrip("a")
    if forced:
        return forced == "120"
    dev = _device_is_sm120()
    if dev is not None:
        return dev
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    return "12.0" in arch and "8.0" not in arch


def is_card(index: int) -> bool:
    """The device runs the sm_80 TriangleAttention path: an A100 (8.0) or an RTX 5090 (12.0)."""
    return torch.cuda.get_device_capability(index) in ((8, 0), (12, 0))


def is_sm120(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (12, 0)


def name(base: str) -> str:
    """The extension name for the build target (``triattn_sm80_rows`` -> ``triattn_sm120_rows`` on the 5090)."""
    if not sm120_build():
        return base
    return base.replace("sm80", "sm120") if "sm80" in base else base + "_sm120"


def gencode() -> list[str]:
    """The ``-gencode`` (and arch define) flags of the build target."""
    if sm120_build():
        return ["-gencode=arch=compute_120a,code=sm_120a", "-DTRIATTN_SM120=1"]
    return ["-gencode=arch=compute_80,code=sm_80"]


def on_sm120(t: torch.Tensor) -> bool:
    """``t`` is a CUDA tensor on an RTX 5090."""
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (12, 0)

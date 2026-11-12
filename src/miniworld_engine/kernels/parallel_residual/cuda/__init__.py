"""The hand-written CUDA kernels of the fused parallel residual (``modules/residual.py``, ``parallel_residual.cu``): ONE source built as
``parallel_residual_sm90`` (H100, sm_90a) and ``parallel_residual_sm80`` (A100, sm_80); the dtype and the branch count are template
instances of one extension, the sizes launch arguments (no shape-dependent build). Launched on torch's current stream (CUDA-graph
capturable); built ahead of time by the prebuild (:func:`prebuild`), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent
_ARCH = {"sm90": "90a", "sm80": "80"}


@functools.lru_cache(maxsize=None)
def ext(arch: str):
    """The extension of ``arch`` (``"sm90"`` / ``"sm80"``)."""
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name=f"parallel_residual_{arch}",
        sources=[str(_dir / "parallel_residual.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes(_ARCH[arch])],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def _device_ext(device: torch.device):
    return ext("sm80" if torch.cuda.get_device_capability(device) == (8, 0) else "sm90")


def forward(x: torch.Tensor, branches: list[torch.Tensor]) -> torch.Tensor:
    """``x + sum_i (y_i - x)`` for 1 .. 4 contiguous tensors of ``x``'s shape, dtype and device (the module checks)."""
    return _device_ext(x.device).forward(x, branches)


def backward(g: torch.Tensor, n: int) -> torch.Tensor:
    """``(1 - n) g`` (fp32 product, one rounding to ``g``'s dtype): the gradient of ``x``."""
    return _device_ext(g.device).backward(g, int(n))


def prebuild(target, rows):
    """The build of the ``parallel_residual`` family (kernels/_prebuild.py): the target card's extension, once, when a requirements row
    (of ``ParallelResidual``) needs the family."""
    if rows and target.arch in _ARCH:
        yield functools.partial(ext, target.arch)

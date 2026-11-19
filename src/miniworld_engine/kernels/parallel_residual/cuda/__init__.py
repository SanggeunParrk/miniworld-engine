"""The hand-written CUDA kernels of the fused parallel residual (``modules/residual.py``, ``parallel_residual.cu``): ONE source built as
``parallel_residual_sm90`` (H100, sm_90a), ``parallel_residual_sm80`` (A100, sm_80) and ``parallel_residual_sm120`` (RTX 5090, sm_120a); the dtype and the branch count are template
instances of one extension, the sizes launch arguments (no shape-dependent build). Launched on torch's current stream (CUDA-graph
capturable); built ahead of time by the prebuild (:func:`prebuild`), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque

_dir = Path(__file__).parent
_ARCH = {"sm90": "90a", "sm80": "80", "sm120": "120a"}  # sm120: the GeForce RTX 5090 (sm_120a), the same source
# A100: plain loads and stores (PR_HINT=1): the streaming-hinted ones (ld.global.cs / st.global.cs, the H100 default) run 5-8 % under the Triton kernel there (1.60 vs 1.71 TB/s of 1.94)
_FLAGS = {"sm80": ("-DPR_HINT=1",)}


@functools.lru_cache(maxsize=None)
def ext(arch: str):
    """The extension of ``arch`` (``"sm90"`` / ``"sm80"`` / ``"sm120"``)."""
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name=f"parallel_residual_{arch}",
        sources=[str(_dir / "parallel_residual.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes(_ARCH[arch]), *_FLAGS.get(arch, ())],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def _device_ext(device: torch.device):
    return ext({(8, 0): "sm80", (12, 0): "sm120"}.get(torch.cuda.get_device_capability(device), "sm90"))


def _forward_fake(x: torch.Tensor, branches: list[torch.Tensor]) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _backward_fake(g: torch.Tensor, n: int, copies: int = 0) -> list[torch.Tensor]:
    return [torch.empty_like(g, memory_format=torch.contiguous_format) for _ in range(1 + copies)]


# opaque to torch.compile (kernels/_compile.py): a PairformerBlock(parallel=True) calls the forward inside its (compiled) block
@opaque(fake=_forward_fake, name="parallel_residual_cuda_forward")
def forward(x: torch.Tensor, branches: list[torch.Tensor]) -> torch.Tensor:
    """``x + sum_i (y_i - x)`` for 1 .. 5 contiguous tensors of ``x``'s shape, dtype and device (the module checks)."""
    return _device_ext(x.device).forward(x, branches)


@opaque(fake=_backward_fake, name="parallel_residual_cuda_backward")
def backward(g: torch.Tensor, n: int, copies: int = 0) -> list[torch.Tensor]:
    """``[(1 - n) g, g, ..., g]`` in one pass: the gradient of ``x`` (fp32 product, one rounding to ``g``'s dtype) and ``copies`` fresh tensors
    equal to ``g`` (the gradients of the branches that are autograd leaves: ``modules/residual.py``)."""
    return _device_ext(g.device).backward(g, int(n), int(copies))


def prebuild(target, rows):
    """The build of the ``parallel_residual`` family (kernels/_prebuild.py): the target card's extension, once, when a requirements row
    (of ``ParallelResidual``) needs the family."""
    if rows and target.arch in _ARCH:
        yield functools.partial(ext, target.arch)

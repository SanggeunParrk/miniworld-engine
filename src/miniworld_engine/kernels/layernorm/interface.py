"""The LayerNorm family's public entry point.

``layernorm_kernel`` is the dispatching entry (it picks a backward path per GPU via
``dispatch.py``); ``triton_layernorm`` is the plain autograd Triton entry that callers who want
that specific path -- ``transition``'s split fallback, the trimul front -- ask for by name.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.layernorm.compile_native import (
    layernorm_dispatch_compile,
)
from miniworld_engine.kernels.layernorm.reference import layernorm_pytorch
from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm

__all__ = ["layernorm_kernel", "triton_layernorm"]

def layernorm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Standalone LayerNorm kernel with automatic backward reduction dispatch."""
    if not x.is_cuda:
        return layernorm_pytorch(x, weight, bias, eps)
    if weight is None or bias is None:
        return triton_layernorm(x, weight, bias, eps)

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    if _use_pipe(x):
        from miniworld_engine.kernels.layernorm.cuda.pipe import layernorm_pipe

        return layernorm_pipe(x, weight, bias, eps)
    return layernorm_dispatch_compile(x, weight, bias, eps)


def _use_pipe(x: torch.Tensor) -> bool:
    """The pipelined hand-CUDA kernels (``cuda/pipe``) on H100 for rows of 144-2048 bytes, unless the portable rung is forced."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.layernorm.cuda.pipe import supported

    return (
        supported(x)
        and torch.cuda.get_device_capability(x.device) == (9, 0)
        and not settings.current().force_portable_kernels
        and not torch.are_deterministic_algorithms_enabled()  # the weight gradient adds atomically (order not fixed)
    )

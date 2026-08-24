"""The LayerNorm family's public entry point.

``layernorm_kernel`` is the dispatching entry (it picks a backward path per GPU via
``dispatch.py``); ``triton_layernorm`` is the plain autograd Triton entry that callers who want
that specific path -- ``transition``'s split fallback, the trimul front -- ask for by name.
"""

from __future__ import annotations

import torch

from .compile_native import layernorm_dispatch_compile
from .reference import layernorm_pytorch
from .triton.main import triton_layernorm

__all__ = ["layernorm_kernel", "triton_layernorm"]

def layernorm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Standalone LayerNorm kernel with automatic backward reduction dispatch."""
    if weight is None or bias is None:
        return layernorm_pytorch(x, weight, bias, eps)
    if not x.is_cuda:
        return layernorm_pytorch(x, weight, bias, eps)

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    return layernorm_dispatch_compile(x, weight, bias, eps)

"""Public entrypoint for the new standalone LayerNorm kernel."""

from __future__ import annotations

import torch

from .compile_native import layernorm_dispatch_compile
from .reference import layernorm_pytorch

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

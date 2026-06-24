"""Public entrypoint for the new standalone LayerNorm kernel."""

from __future__ import annotations

import torch

from .reference import layernorm_pytorch
from .triton.partial import triton_layernorm_partial


def layernorm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Standalone LayerNorm kernel with partial-buffer backward reduction."""
    if weight is None or bias is None:
        return layernorm_pytorch(x, weight, bias, eps)
    if not x.is_cuda:
        return layernorm_pytorch(x, weight, bias, eps)
    return triton_layernorm_partial(x, weight.contiguous(), bias.contiguous(), eps)

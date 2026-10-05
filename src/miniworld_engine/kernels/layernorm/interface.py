"""The LayerNorm family's public entry point.

``layernorm_kernel`` is the dispatching entry: on an A100 the hand-CUDA rows of ``cuda/sm80.py`` (inference and training, every width, with or without weight / bias),
elsewhere (and for what that gate refuses, or with ``MINIWORLD_NORMS_SM80=0``) the Triton forward with a backward path picked per GPU via ``dispatch.py``;
``triton_layernorm`` is the plain autograd Triton entry that callers who want
that specific path -- ``transition``'s split fallback, the trimul front -- ask for by name.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.layernorm.compile_native import (
    layernorm_dispatch_compile,
)
from miniworld_engine.kernels.layernorm.cuda import sm80 as _sm80
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
    if _sm80.supports(x, weight, bias):
        return _sm80.layernorm(x, weight, bias, eps)
    if weight is None or bias is None:
        return triton_layernorm(x, weight, bias, eps)

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    return layernorm_dispatch_compile(x, weight, bias, eps)

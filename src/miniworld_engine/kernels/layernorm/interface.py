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
    from miniworld_engine.kernels.layernorm.cuda import pipe

    if pipe.serves(x):  # H100, rows of 144-2048 bytes: the pipelined hand-CUDA kernels (``cuda/pipe``)
        return pipe.layernorm_pipe(x, weight, bias, eps)
    return layernorm_dispatch_compile(x, weight, bias, eps)


def layer_norm(x, weight, bias, eps, row_scale=None):
    """LayerNorm over the last axis with an optional per-row scale, on the kernel backend the running card takes (H100 / B200: CUDA; a card without a
    record: PyTorch; any other card: the portable kernel). The entry point for a caller that needs the op alone, without building a module."""
    from miniworld_engine.kernels._ops_routed import layer_norm as _routed

    return _routed(x, weight, bias, eps, row_scale)

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
    import torch.nn.functional as F

    from miniworld_engine.kernels._device import no_triton
    from miniworld_engine.modules.dispatch import portable_blocked

    if no_triton(x):
        # H100 / B200: the general CUDA LayerNorm (any width the kernels take, bf16 / fp16 / fp32, forward and backward); the row scale is
        # applied after it (one more rounding than the Triton kernel's in-kernel scale), a call it does not take runs the PyTorch statements
        from miniworld_engine.kernels.layernorm.cuda.general import layernorm_cuda, supported

        out = layernorm_cuda(x, weight, bias, eps) if supported(x) else F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        return out if row_scale is None else out * row_scale.reshape(*x.shape[:-1], 1).to(out.dtype)
    if x.is_cuda and portable_blocked("layernorm", x.device, x, *(t for t in (weight, bias) if t is not None)):
        out = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        return out if row_scale is None else out * row_scale.reshape(*x.shape[:-1], 1).to(out.dtype)
    # The caller picks the backward: the hand-CUDA warp-per-row one (B200-measured, masked / opted-in bf16 N 128..512) or the Triton one.
    from miniworld_engine.kernels.layernorm.cuda.masked_bwd import (
        layernorm_cuda_bwd_autograd,
        use_cuda_bwd,
    )

    if use_cuda_bwd(x, row_scale):
        return layernorm_cuda_bwd_autograd(x, weight, bias, eps, row_scale)
    from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm

    return triton_layernorm(x, weight, bias, eps, row_scale)

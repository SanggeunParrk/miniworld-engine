"""General RMSNorm on CUDA: ``x / sqrt(mean(x^2) + eps) * weight`` over the last dimension at any width (1 .. 65 536 bf16 / fp16,
32 768 fp32), any row count, bf16 / fp16 / fp32 activations, fp32 statistics, optional weight (cast to fp32), forward and backward
(dx, and dw without atomics: a replay is bit-identical). It is the general LayerNorm kernel (``kernels/layernorm/cuda/general``) built
with the mean and the bias dropped, so it shares that extension, its launch configuration and its measurements.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.layernorm.cuda.general import _affine, _ext, config, supported


def _fwd_fake(x, w, eps, save):
    """y like x; rstd fp32 [M] when save, else empty."""
    return torch.empty_like(x), torch.empty((x.shape[0] if save else 0,), dtype=torch.float32, device=x.device)


@opaque(fake=_fwd_fake, name="rmsnorm_general_fwd")
def _fwd(x: torch.Tensor, w: torch.Tensor, eps: float, save: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm forward of contiguous x [M, N]; w fp32 [N] or empty. (y, rstd)."""
    tpr, ch, vec = config(x.shape[1], x.element_size())
    y, _, rstd = _ext().fwd(x, w, x.new_empty((0,), dtype=torch.float32), float(eps), bool(save), True, tpr, ch, vec)
    return y, rstd


def _bwd_fake(dy, x, w, rstd, wgrad):
    """dx like x; dw fp32 [N] when wgrad, else empty."""
    return torch.empty_like(x), torch.empty((x.shape[1] if wgrad else 0,), dtype=torch.float32, device=x.device)


@opaque(fake=_bwd_fake, name="rmsnorm_general_bwd")
def _bwd(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor, wgrad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm backward from the saved rstd. (dx, dw); dw only when wgrad."""
    tpr, ch, vec = config(x.shape[1], x.element_size())
    empty = x.new_empty((0,), dtype=torch.float32)
    dx, dw, _ = _ext().bwd(dy, x, w, empty, rstd, bool(wgrad), True, tpr, ch, vec, 0)
    return dx, dw


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, weight, eps):
        w = _affine(weight, x2)
        y, rstd = _fwd(x2, w, eps, True)
        ctx.save_for_backward(x2, w, rstd)
        ctx.weight_dtype = None if weight is None else weight.dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        wgrad = ctx.weight_dtype is not None and ctx.needs_input_grad[1]
        dx, dw = _bwd(dy.contiguous(), x2, w, rstd, bool(wgrad))
        return dx, (dw.to(ctx.weight_dtype) if wgrad else None), None


def rmsnorm_cuda(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension; ``y`` in ``x``'s dtype, fp32 statistics; ``weight`` optional, any float dtype (its fp32
    gradient handed back in its dtype)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    if not (torch.is_grad_enabled() and (x.requires_grad or (weight is not None and weight.requires_grad))):
        y, _ = _fwd(x2, _affine(weight, x2), float(eps), False)
        return y.reshape(shape)
    return _RMSNorm.apply(x2, weight, float(eps)).reshape(shape)


__all__ = ["rmsnorm_cuda", "supported"]

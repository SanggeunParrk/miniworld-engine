"""Autograd-wrapped (trainable) LayerNormLinear — the portable Triton path.

Forward: the stats-saving Triton forward (``layernorm_linear_triton_fwd_stats``).
Backward (compose):
  dx_normed = dY @ W                         (cuBLAS)
  dW        = dYᵀ @ x_normed                 (cuBLAS wgrad; x_normed recomputed)
  dx,dγ,dβ  = LayerNormBackward(dx_normed,…)  (the repo's Triton ``layer_norm_bwd_dx_fused``)
  db        = Σ_m dY                          (reduction)
"""

from __future__ import annotations

import torch
import triton

# `layer_norm_bwd_dx_fused` is level=both in kernels/registry/registry.csv -> both_key. The key is L (the
# token/atom count), never the row count M: the saved x here is already the flattened (M, K)
# matrix, so L has to arrive from the caller (see `LayerNormLinearTritonFn.forward`'s `length` input).
from miniworld_engine.autotune.shape_key import both_key, pack
from miniworld_engine.kernels._compile import opaque

# torch/triton-only — safe to import eagerly; this IS the LN-part backward.
from miniworld_engine.kernels.layernorm.triton.main import layer_norm_bwd_dx_fused
from miniworld_engine.kernels.layernorm_linear.triton.recompute import (
    _recompute_xnormed,
)


def _ln_backward_fake(dx_normed, x, gamma, mean, rstd, shape_key=None):
    """(dx like dx_normed, dgamma (K,), dbeta (K,)); dγ/dβ are fp32 — atomic-accumulated over M."""
    return (
        torch.empty_like(dx_normed),
        x.new_empty((x.shape[-1],), dtype=torch.float32),
        x.new_empty((x.shape[-1],), dtype=torch.float32),
    )


@opaque(fake=_ln_backward_fake, name="layernorm_linear_ln_bwd")
def _ln_backward(dx_normed: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor,
                 mean: torch.Tensor, rstd: torch.Tensor, shape_key: int | None = None,
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dx, dgamma, dbeta for the LayerNorm part, reusing the repo's fused Triton kernel.

    Feeds ``dy = dx_normed`` (grad w.r.t. the LN output) so the kernel's dx/dw/db become the
    LNL grads dx / dγ / dβ. dw, db come back fp32 (atomic-accumulated over M).

    ``shape_key`` is ``both_key(L)`` from the Function's `length` input. None -> smallest bucket,
    an explicit "L not supplied" label (bench / driver entry only)."""
    M, K = x.shape
    dx = torch.empty_like(dx_normed)
    dgamma = torch.zeros(K, dtype=torch.float32, device=x.device)
    dbeta = torch.zeros(K, dtype=torch.float32, device=x.device)
    xc = x.to(dx_normed.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
    layer_norm_bwd_dx_fused[grid](
        dx, dx_normed, dgamma, dbeta,
        xc, gamma, mean, rstd, rstd,
        dgamma.stride(0), dbeta.stride(0), xc.stride(0), xc.stride(1),
        M, K,
        # BLOCK_N is a tuned tile now (see layernorm/triton/main.py); this is only the cache label.
        shape_key=both_key(0, N=K) if shape_key is None else pack(shape_key, N=K),
        HAS_ROWSCALE=False,
    )
    return dx, dgamma, dbeta


def _compose_backward(dY, x, mean, rstd, gamma, beta, W, has_bias, *,
                      shape_key: int | None = None):
    """Shared LayerNormLinear backward: two GEMMs + the Triton LN-backward.

    dx_normed = dY @ W and dW = dYᵀ @ x_normed via cuBLAS; x_normed recomputed from saved
    mean/rstd; dx/dγ/dβ from the reused Triton LN backward; db = Σ_m dY.

    ``shape_key`` is ``both_key(L)`` from the Function's `length` input; it labels both Triton
    launches here (`_recompute_xnormed` and `_ln_backward`)."""
    dY = dY.contiguous()
    dx_normed = torch.matmul(dY, W)
    x_normed = _recompute_xnormed(x.to(dY.dtype), gamma, beta, mean, rstd, shape_key=shape_key)
    dW = torch.matmul(dY.t(), x_normed)
    dx, dgamma, dbeta = _ln_backward(dx_normed, x, gamma, mean, rstd, shape_key=shape_key)
    db = dY.sum(0).to(W.dtype) if has_bias else None
    return dx.to(x.dtype), dgamma.to(gamma.dtype), dbeta.to(beta.dtype), dW, db


class LayerNormLinearTritonFn(torch.autograd.Function):
    """Portable trainable LayerNormLinear: Triton forward + portable backward (cuBLAS GEMMs +
    Triton LN-bwd).

    ``length`` (L, the pre-flatten token/atom count) is a POSITIONAL input because
    ``autograd.Function.apply`` takes no keywords; it carries no gradient, so ``backward``
    returns a trailing ``None`` for it."""

    @staticmethod
    def forward(ctx, x, ln_weight, ln_bias, weight, bias, eps, length):
        from miniworld_engine.kernels.layernorm_linear.triton.fused import (
            layernorm_linear_triton_fwd_stats,
        )

        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        Y, mean, rstd = layernorm_linear_triton_fwd_stats(
            x2, ln_weight, ln_bias, weight, bias, eps)
        ctx.save_for_backward(x2, mean, rstd, ln_weight, ln_bias, weight)
        ctx.input_shape = x.shape
        ctx.has_bias = bias is not None
        # Rows, not `length`: see BOTH_ROWS. x is (M, K) here, so M is readable directly.
        ctx.shape_key = both_key(x.reshape(-1, x.shape[-1]).shape[0])
        return Y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dY):
        x, mean, rstd, gamma, beta, W = ctx.saved_tensors
        dx, dg, db_ln, dW, db = _compose_backward(
            dY.reshape(-1, dY.shape[-1]), x, mean, rstd, gamma, beta, W,
            ctx.has_bias,
            shape_key=ctx.shape_key,
        )
        return dx.reshape(ctx.input_shape), dg, db_ln, dW, db, None, None


def layernorm_linear_triton_fn(x, ln_weight, ln_bias, weight, bias=None, eps: float = 1e-5,
                               length: int | None = None):
    """Trainable LayerNormLinear (Triton fwd + cuBLAS/Triton bwd).

    ``length`` is L -- the TOKEN/ATOM count of the activation before it was flattened (a trimul
    pair view has M = L*L) -- used only as the backward's autotune-cache label."""
    if not x.is_cuda or x.dtype == torch.float64 or x.shape[-1] > 1024 or x.numel() == 0:
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        xn = torch.nn.functional.layer_norm(
            x.to(acc), (x.shape[-1],), ln_weight.to(acc), ln_bias.to(acc), eps)
        return torch.nn.functional.linear(xn.to(x.dtype), weight, bias)
    if x.dtype == torch.float32:
        from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
        # Keep a pre-flatten leading dimension for the norm's cache-key contract.
        xn = layernorm_kernel(x.unsqueeze(0) if x.ndim == 2 else x,
                              ln_weight, ln_bias, eps).reshape(x.shape)
        return torch.nn.functional.linear(xn, weight, bias)
    return LayerNormLinearTritonFn.apply(x, ln_weight, ln_bias, weight, bias, eps, length)

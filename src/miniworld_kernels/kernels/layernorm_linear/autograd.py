"""Autograd-wrapped (trainable) LayerNormLinear. v1: SM90/Hopper only.

The forward uses the stats-saving M1 path (``save_stats=True`` → returns ``(Y, mean, rstd)``),
and the backward COMPOSES existing pieces rather than writing new kernels:

  dx_normed = dY @ W                         (quack SM90 GEMM)
  dW        = dYᵀ @ x_normed                 (quack SM90 GEMM; x_normed recomputed)
  dx,dγ,dβ  = LayerNormBackward(dx_normed,…)  (the repo's Triton ``layer_norm_bwd_dx_fused``)
  db        = Σ_m dY                          (reduction)

See the derivation in the plan / README. Heavy backends (quack, cute) are imported lazily so
this module imports on non-Hopper boxes; the GEMMs themselves require SM90 (v1 scope — the
portable fallback is Phase 2).
"""

from __future__ import annotations

import torch
import triton

# torch/triton-only (no quack) — safe to import eagerly; this IS the LN-part backward.
from ..layernorm.triton.main import get_seq_group, layer_norm_bwd_dx_fused


def _ln_backward(dx_normed: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor,
                 mean: torch.Tensor, rstd: torch.Tensor):
    """dx, dgamma, dbeta for the LayerNorm part, reusing the repo's fused Triton kernel.

    Feeds ``dy = dx_normed`` (grad w.r.t. the LN output) so the kernel's dx/dw/db become the
    LNL grads dx / dγ / dβ. dw, db come back fp32 (atomic-accumulated over M)."""
    M, K = x.shape
    dx = torch.empty_like(dx_normed)
    dgamma = torch.zeros(K, dtype=torch.float32, device=x.device)
    dbeta = torch.zeros(K, dtype=torch.float32, device=x.device)
    xc = x.to(dx_normed.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa: E731
    layer_norm_bwd_dx_fused[grid](
        dx, dx_normed, dgamma, dbeta,
        xc, gamma, mean, rstd,
        dgamma.stride(0), dbeta.stride(0), xc.stride(0), xc.stride(1),
        M, K,
        BLOCK_N=triton.next_power_of_2(K),
        GROUP_M=get_seq_group(M),
    )
    return dx, dgamma, dbeta


class LayerNormLinearFn(torch.autograd.Function):
    """`Y = LayerNorm(x) @ Wᵀ + b`, differentiable. Saves (x, mean, rstd) for the backward."""

    @staticmethod
    def forward(ctx, x, ln_weight, ln_bias, weight, bias, eps):
        from .cute import layernorm_linear as _fwd  # lazy (pulls quack)

        Y, mean, rstd = _fwd(x, ln_weight, ln_bias, weight, bias, eps, save_stats=True)
        ctx.save_for_backward(x, mean, rstd, ln_weight, ln_bias, weight)
        ctx.eps = eps
        ctx.has_bias = bias is not None
        return Y

    @staticmethod
    def backward(ctx, dY):
        from quack.gemm_interface import gemm  # lazy (SM90)

        x, mean, rstd, gamma, beta, W = ctx.saved_tensors
        dY = dY.contiguous()

        # GEMM #1: dx_normed = dY @ W  — (M,N)@(N,K) -> (M,K)  (W is already (N,K)=(K_gemm,N_gemm))
        dx_normed = gemm(dY, W)

        # recompute x_normed = ((x-μ)·rstd)·γ + β for GEMM #2 (cheap; avoids saving it — TE-style)
        xf = x.float()
        xhat = (xf - mean.unsqueeze(1)) * rstd.unsqueeze(1)
        x_normed = (xhat * gamma.float() + beta.float()).to(x.dtype)

        # GEMM #2: dW = dYᵀ @ x_normed  — (N,M)@(M,K) -> (N,K)
        dW = gemm(dY.t().contiguous(), x_normed)

        # LN part: dx, dγ, dβ (reuses the Triton LN backward with dy = dx_normed)
        dx, dgamma, dbeta = _ln_backward(dx_normed, x, gamma, mean, rstd)

        db = dY.sum(0).to(W.dtype) if ctx.has_bias else None
        return dx.to(x.dtype), dgamma.to(gamma.dtype), dbeta.to(beta.dtype), dW, db, None


def layernorm_linear_fn(x, ln_weight, ln_bias, weight, bias=None, eps: float = 1e-5):
    """Trainable LayerNormLinear (autograd). v1 SM90/Hopper only. Returns Y; grads flow to
    x, ln_weight, ln_bias, weight, bias via ``LayerNormLinearFn``."""
    return LayerNormLinearFn.apply(x, ln_weight, ln_bias, weight, bias, eps)

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
import triton.language as tl

# torch/triton-only (no quack) — safe to import eagerly; this IS the LN-part backward.
from ..layernorm.triton.main import get_seq_group, layer_norm_bwd_dx_fused


@triton.jit
def _xnormed_kernel(x_ptr, g_ptr, b_ptr, mean_ptr, rstd_ptr, y_ptr, M, K, sx0, sx1,
                    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    rm = rows < M
    cm = cols < K
    x = tl.load(x_ptr + rows[:, None] * sx0 + cols[None, :] * sx1,
                mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=rm, other=0.0)[:, None]
    rstd = tl.load(rstd_ptr + rows, mask=rm, other=0.0)[:, None]
    g = tl.load(g_ptr + cols, mask=cm, other=0.0).to(tl.float32)[None, :]
    b = tl.load(b_ptr + cols, mask=cm, other=0.0).to(tl.float32)[None, :]
    y = (x - mean) * rstd * g + b
    tl.store(y_ptr + rows[:, None] * sx0 + cols[None, :] * sx1,
             y.to(y_ptr.dtype.element_ty), mask=rm[:, None] & cm[None, :])


def _recompute_xnormed(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                       mean: torch.Tensor, rstd: torch.Tensor):
    """x_normed = (x-mean)*rstd*γ + β, one fused bf16 pass using the SAVED mean/rstd (no stats
    recompute). Replaces the multi-pass fp32 torch recompute that was 42% of the backward.
    Fixed launch params (no autotune) for robustness across shapes."""
    M, K = x.shape
    y = torch.empty_like(x)
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_M = 8
    grid = (triton.cdiv(M, BLOCK_M),)
    _xnormed_kernel[grid](
        x, gamma, beta, mean, rstd, y, M, K, x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, num_warps=8,
    )
    return y


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

        # recompute x_normed (TE-style, not saved) from the SAVED mean/rstd — one fused bf16
        # pass instead of the multi-pass fp32 torch recompute that was 42% of the backward.
        x_normed = _recompute_xnormed(x.to(dY.dtype), gamma, beta, mean, rstd)

        # GEMM #2: dW = dYᵀ @ x_normed  — (N,M)@(M,K) -> (N,K). This is the "wgrad" aspect
        # (contraction over the huge M, tiny N×K output) where quack has no split-K and stalls
        # at ~1.2ms flat / ~27% peak; cuBLAS split-K is 3-18x faster here, so use torch.matmul.
        # (dx's GEMM #1 stays on quack — there quack ties/beats cuBLAS.)
        dW = torch.matmul(dY.t(), x_normed)

        # LN part: dx, dγ, dβ (reuses the Triton LN backward with dy = dx_normed)
        dx, dgamma, dbeta = _ln_backward(dx_normed, x, gamma, mean, rstd)

        db = dY.sum(0).to(W.dtype) if ctx.has_bias else None
        return dx.to(x.dtype), dgamma.to(gamma.dtype), dbeta.to(beta.dtype), dW, db, None


def layernorm_linear_fn(x, ln_weight, ln_bias, weight, bias=None, eps: float = 1e-5):
    """Trainable LayerNormLinear (autograd). v1 SM90/Hopper only. Returns Y; grads flow to
    x, ln_weight, ln_bias, weight, bias via ``LayerNormLinearFn``."""
    return LayerNormLinearFn.apply(x, ln_weight, ln_bias, weight, bias, eps)

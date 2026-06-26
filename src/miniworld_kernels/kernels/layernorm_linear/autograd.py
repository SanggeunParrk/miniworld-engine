"""Autograd-wrapped (trainable) LayerNormLinear. v1: SM90/Hopper only.

The forward uses the stats-saving M1 path (``save_stats=True`` → returns ``(Y, mean, rstd)``).

Backward (cute path, K≤256): the FUSED 1+4 kernel (``cute/dgrad_lnbwd.py``) does dY@W + the
LN-norm-backward in one epilogue → dx, with NO dx_normed (M,K) round-trip. dW/dγ/dβ/db come from
a single wgrad GEMM T = dYᵀ@x̂ (see ``_compose_backward_fused``). K>256 falls back to the
unfused compose below.

Backward (unfused compose — portable path + cute K>256):
  dx_normed = dY @ W                         (quack SM90 GEMM / torch.matmul)
  dW        = dYᵀ @ x_normed                 (cuBLAS wgrad; x_normed recomputed)
  dx,dγ,dβ  = LayerNormBackward(dx_normed,…)  (the repo's Triton ``layer_norm_bwd_dx_fused``)
  db        = Σ_m dY                          (reduction)

Heavy backends (quack, cute) are imported lazily so this module imports on non-Hopper boxes;
the GEMMs/fused kernel require SM90 (the fully-portable training path is LayerNormLinearTritonFn).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# torch/triton-only (no quack) — safe to import eagerly; this IS the LN-part backward.
from ..layernorm.triton.main import get_seq_group, layer_norm_bwd_dx_fused


@triton.jit
def _xnormed_kernel(x_ptr, g_ptr, b_ptr, mean_ptr, rstd_ptr, y_ptr, M, K, sx0, sx1, sy0, sy1,
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
    tl.store(y_ptr + rows[:, None] * sy0 + cols[None, :] * sy1,
             y.to(y_ptr.dtype.element_ty), mask=rm[:, None] & cm[None, :])


def _recompute_xnormed(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                       mean: torch.Tensor, rstd: torch.Tensor):
    """x_normed = (x-mean)*rstd*γ + β, one fused bf16 pass using the SAVED mean/rstd (no stats
    recompute). Reads x at its own strides (strided/transposed view OK — no pre-copy) and writes
    a CONTIGUOUS (M,K) output."""
    M, K = x.shape
    y = torch.empty(M, K, device=x.device, dtype=x.dtype)   # contiguous out
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_M = 8
    grid = (triton.cdiv(M, BLOCK_M),)
    _xnormed_kernel[grid](
        x, gamma, beta, mean, rstd, y, M, K, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, num_warps=8,
    )
    return y


@triton.jit
def _xhat_kernel(x_ptr, mean_ptr, rstd_ptr, y_ptr, M, K, sx0, sx1, sy0, sy1,
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
    y = (x - mean) * rstd
    tl.store(y_ptr + rows[:, None] * sy0 + cols[None, :] * sy1,
             y.to(y_ptr.dtype.element_ty), mask=rm[:, None] & cm[None, :])


def _recompute_xhat(x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor):
    """x̂ = (x-mean)·rstd (no affine), one fused bf16 pass using the SAVED mean/rstd.
    Reads x at its own strides (so a transposed/strided view is fine — NO pre-copy) and
    writes a CONTIGUOUS (M,K) x̂. This lets the caller feed a strided x (e.g. a bmm
    output viewed channel-major) without a .contiguous() transpose copy."""
    M, K = x.shape
    y = torch.empty(M, K, device=x.device, dtype=x.dtype)   # contiguous out
    grid = (triton.cdiv(M, 8),)
    _xhat_kernel[grid](x, mean, rstd, y, M, K, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                       BLOCK_M=8, BLOCK_K=triton.next_power_of_2(K), num_warps=8)
    return y


def _compose_backward_fused(dY, x, mean, rstd, gamma, beta, W, has_bias):
    """Fused-1+4 cute backward (SM90, K≤256). dx comes from ``dgrad_lnbwd_cute`` (fused dY@W GEMM +
    LN-norm-backward epilogue — no dx_normed (M,K) round-trip). dW/dγ/dβ/db are derived from a
    single wgrad GEMM T = dYᵀ@x̂ (x̂ recomputed from saved stats — no x_normed materialization):

        db = Σ_m dY ;  dW = γ⊙T + outer(db,β) ;  dγ = (W⊙T).sum(0) ;  dβ = db @ W

    All exact (no division); verified cos=1.0 vs autograd (dgrad_lnbwd_verify.py)."""
    from .cute.dgrad_lnbwd import dgrad_lnbwd_cute  # lazy (SM90)
    dY = dY.contiguous()
    xc = x.to(dY.dtype)
    xhat = _recompute_xhat(xc, mean, rstd)               # (M,K) bf16 = (x-μ)·rstd
    dx = dgrad_lnbwd_cute(dY, W, xhat, gamma, rstd)      # fused 1+4 → dx (M,K)
    T = torch.matmul(dY.t(), xhat)                       # (N,K) wgrad on x̂ (cuBLAS)
    db = dY.sum(0)                                       # (N,)
    Tf, gf, bf, dbf, Wf = T.float(), gamma.float(), beta.float(), db.float(), W.float()
    dW = (gf[None, :] * Tf + dbf[:, None] * bf[None, :]).to(W.dtype)
    dgamma = (Wf * Tf).sum(0).to(gamma.dtype)
    dbeta = (dbf @ Wf).to(beta.dtype)
    db_out = db.to(W.dtype) if has_bias else None
    return dx.to(x.dtype), dgamma, dbeta, dW, db_out


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


def _compose_backward(dY, x, mean, rstd, gamma, beta, W, has_bias, *, dx_via_quack):
    """Shared LayerNormLinear backward: two GEMMs + the Triton LN-backward.

    dx_normed = dY @ W   (dx_via_quack: quack SM90 gemm, ties/beats cuBLAS; else torch.matmul
                          for the portable/triton path); dW = dYᵀ @ x_normed via cuBLAS
    (wgrad aspect — quack has no split-K there); x_normed recomputed from saved mean/rstd;
    dx/dγ/dβ from the reused Triton LN backward; db = Σ_m dY."""
    dY = dY.contiguous()
    if dx_via_quack:
        from quack.gemm_interface import gemm  # lazy (SM90)
        dx_normed = gemm(dY, W)
    else:
        dx_normed = torch.matmul(dY, W)
    x_normed = _recompute_xnormed(x.to(dY.dtype), gamma, beta, mean, rstd)
    dW = torch.matmul(dY.t(), x_normed)
    dx, dgamma, dbeta = _ln_backward(dx_normed, x, gamma, mean, rstd)
    db = dY.sum(0).to(W.dtype) if has_bias else None
    return dx.to(x.dtype), dgamma.to(gamma.dtype), dbeta.to(beta.dtype), dW, db


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
        x, mean, rstd, gamma, beta, W = ctx.saved_tensors
        K = x.shape[-1]
        if K <= 128:
            # Fused 1+4 (no dx_normed round-trip) — wins/ties at K=128 across M (A/B: 1.29x@16384,
            # ~1.0x at larger M). At K=256 the full-N epi subtile starves the mainloop (D+C both in
            # smem) → loses at large M; needs gmem x̂-load (M2 mX pattern) before it's wired. K>128
            # uses the unfused compose below.  See dgrad_lnbwd.py / dgrad_lnbwd_bench.py.
            dx, dg, db_ln, dW, db = _compose_backward_fused(
                dY, x, mean, rstd, gamma, beta, W, ctx.has_bias
            )
        else:
            dx, dg, db_ln, dW, db = _compose_backward(
                dY, x, mean, rstd, gamma, beta, W, ctx.has_bias, dx_via_quack=True
            )
        return dx, dg, db_ln, dW, db, None


def layernorm_linear_fn(x, ln_weight, ln_bias, weight, bias=None, eps: float = 1e-5):
    """Trainable LayerNormLinear (autograd, cute/SM90). Returns Y; grads flow to
    x, ln_weight, ln_bias, weight, bias via ``LayerNormLinearFn`` (forward = stats-saving M1)."""
    return LayerNormLinearFn.apply(x, ln_weight, ln_bias, weight, bias, eps)


class LayerNormLinearTritonFn(torch.autograd.Function):
    """Portable trainable LayerNormLinear: Triton forward + portable backward (cuBLAS GEMMs +
    Triton LN-bwd, no quack). The non-Hopper training path (and the bench's `triton` fwd+bwd)."""

    @staticmethod
    def forward(ctx, x, ln_weight, ln_bias, weight, bias, eps):
        from .interface import layernorm_linear_triton  # triton fused forward (portable)

        Y = layernorm_linear_triton(x, ln_weight, ln_bias, weight, bias, eps)
        xf = x.reshape(-1, x.shape[-1]).float()
        mean = xf.mean(-1)
        rstd = torch.rsqrt(xf.var(-1, unbiased=False) + eps)
        ctx.save_for_backward(x, mean, rstd, ln_weight, ln_bias, weight)
        ctx.has_bias = bias is not None
        return Y

    @staticmethod
    def backward(ctx, dY):
        x, mean, rstd, gamma, beta, W = ctx.saved_tensors
        dx, dg, db_ln, dW, db = _compose_backward(
            dY, x, mean, rstd, gamma, beta, W, ctx.has_bias, dx_via_quack=False
        )
        return dx, dg, db_ln, dW, db, None


def layernorm_linear_triton_fn(x, ln_weight, ln_bias, weight, bias=None, eps: float = 1e-5):
    """Trainable LayerNormLinear, portable (Triton fwd + cuBLAS/Triton bwd, no quack)."""
    return LayerNormLinearTritonFn.apply(x, ln_weight, ln_bias, weight, bias, eps)

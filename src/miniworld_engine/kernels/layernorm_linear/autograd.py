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

from .triton.recompute import _recompute_xhat, _recompute_xnormed

import torch
import triton

# torch/triton-only (no quack) — safe to import eagerly; this IS the LN-part backward.

from ..layernorm.triton.main import layer_norm_bwd_dx_fused

# `layer_norm_bwd_dx_fused` is level=both in kernels/registry.csv -> both_key. The key is L (the
# token/atom count), never the row count M: the saved x here is already the flattened (M, K)
# matrix, so L has to arrive from the caller (see `LayerNormLinearFn.forward`'s `length` input).
from miniworld_engine.autotune.shape_key import both_key














def _compose_backward_fused(dY, x, mean, rstd, gamma, beta, W, has_bias, *,
                            shape_key: int | None = None):
    """Fused-1+4 cute backward (SM90, K≤256). dx comes from ``dgrad_lnbwd_cute`` (fused dY@W GEMM +
    LN-norm-backward epilogue — no dx_normed (M,K) round-trip). dW/dγ/dβ/db are derived from a
    single wgrad GEMM T = dYᵀ@x̂ (x̂ recomputed from saved stats — no x_normed materialization):

        db = Σ_m dY ;  dW = γ⊙T + outer(db,β) ;  dγ = (W⊙T).sum(0) ;  dβ = db @ W

    All exact (no division); verified cos=1.0 vs autograd (dgrad_lnbwd_verify.py).

    ``shape_key`` is ``both_key(L)`` from the Function's `length` input; it labels the one Triton
    launch here (`_recompute_xhat`)."""
    from .cute.dgrad_lnbwd import dgrad_lnbwd_cute  # lazy (SM90)
    dY = dY.contiguous()
    xc = x.to(dY.dtype)
    xhat = _recompute_xhat(xc, mean, rstd, shape_key=shape_key)   # (M,K) bf16 = (x-μ)·rstd
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
                 mean: torch.Tensor, rstd: torch.Tensor, *, shape_key: int | None = None):
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
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # noqa: E731
    layer_norm_bwd_dx_fused[grid](
        dx, dx_normed, dgamma, dbeta,
        xc, gamma, mean, rstd, rstd,
        dgamma.stride(0), dbeta.stride(0), xc.stride(0), xc.stride(1),
        M, K,
        # BLOCK_N is a tuned tile now (see layernorm/triton/main.py); this is only the cache label.
        shape_key=both_key(0) if shape_key is None else shape_key,
        HAS_ROWSCALE=False,
    )
    return dx, dgamma, dbeta


def _compose_backward(dY, x, mean, rstd, gamma, beta, W, has_bias, *, dx_via_quack,
                      shape_key: int | None = None):
    """Shared LayerNormLinear backward: two GEMMs + the Triton LN-backward.

    dx_normed = dY @ W   (dx_via_quack: quack SM90 gemm, ties/beats cuBLAS; else torch.matmul
                          for the portable/triton path); dW = dYᵀ @ x_normed via cuBLAS
    (wgrad aspect — quack has no split-K there); x_normed recomputed from saved mean/rstd;
    dx/dγ/dβ from the reused Triton LN backward; db = Σ_m dY.

    ``shape_key`` is ``both_key(L)`` from the Function's `length` input; it labels both Triton
    launches here (`_recompute_xnormed` and `_ln_backward`)."""
    dY = dY.contiguous()
    if dx_via_quack:
        from quack.gemm_interface import gemm  # lazy (SM90)
        dx_normed = gemm(dY, W)
    else:
        dx_normed = torch.matmul(dY, W)
    x_normed = _recompute_xnormed(x.to(dY.dtype), gamma, beta, mean, rstd, shape_key=shape_key)
    dW = torch.matmul(dY.t(), x_normed)
    dx, dgamma, dbeta = _ln_backward(dx_normed, x, gamma, mean, rstd, shape_key=shape_key)
    db = dY.sum(0).to(W.dtype) if has_bias else None
    return dx.to(x.dtype), dgamma.to(gamma.dtype), dbeta.to(beta.dtype), dW, db


class LayerNormLinearFn(torch.autograd.Function):
    """`Y = LayerNorm(x) @ Wᵀ + b`, differentiable. Saves (x, mean, rstd) for the backward.

    ``length`` (L, the pre-flatten token/atom count) is a POSITIONAL input because
    ``autograd.Function.apply`` takes no keywords; it carries no gradient, so ``backward``
    returns a trailing ``None`` for it. It is bucketed once here and reused by the backward via
    ``ctx.shape_key`` -- the saved x is (M, K) and L cannot be recovered from M."""

    @staticmethod
    def forward(ctx, x, ln_weight, ln_bias, weight, bias, eps, length):
        from .cute import layernorm_linear as _fwd  # lazy (pulls quack)

        Y, mean, rstd = _fwd(x, ln_weight, ln_bias, weight, bias, eps, save_stats=True)
        ctx.save_for_backward(x, mean, rstd, ln_weight, ln_bias, weight)
        ctx.eps = eps
        ctx.has_bias = bias is not None
        # Rows, not `length`: see BOTH_ROWS. x is (M, K) here, so M is readable directly.
        ctx.shape_key = both_key(x.reshape(-1, x.shape[-1]).shape[0])
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
                dY, x, mean, rstd, gamma, beta, W, ctx.has_bias, shape_key=ctx.shape_key
            )
        else:
            dx, dg, db_ln, dW, db = _compose_backward(
                dY, x, mean, rstd, gamma, beta, W, ctx.has_bias, dx_via_quack=True,
                shape_key=ctx.shape_key,
            )
        return dx, dg, db_ln, dW, db, None, None


def layernorm_linear_fn(x, ln_weight, ln_bias, weight, bias=None, eps: float = 1e-5,
                        length: int | None = None):
    """Trainable LayerNormLinear (autograd, cute/SM90). Returns Y; grads flow to
    x, ln_weight, ln_bias, weight, bias via ``LayerNormLinearFn`` (forward = stats-saving M1).

    ``length`` is L -- the TOKEN/ATOM count of the activation before it was flattened into the
    (M, K) ``x`` this entry point takes (a trimul pair view has M = L*L). Pure autotune-cache
    label; omitting it labels the backward's Triton launches with the smallest bucket."""
    return LayerNormLinearFn.apply(x, ln_weight, ln_bias, weight, bias, eps, length)


class LayerNormLinearTritonFn(torch.autograd.Function):
    """Portable trainable LayerNormLinear: Triton forward + portable backward (cuBLAS GEMMs +
    Triton LN-bwd, no quack). The non-Hopper training path (and the bench's `triton` fwd+bwd).

    ``length`` is the same positional, gradient-free L input as on ``LayerNormLinearFn``."""

    @staticmethod
    def forward(ctx, x, ln_weight, ln_bias, weight, bias, eps, length):
        from .interface import layernorm_linear_triton  # triton fused forward (portable)

        Y = layernorm_linear_triton(x, ln_weight, ln_bias, weight, bias, eps)
        xf = x.reshape(-1, x.shape[-1]).float()
        mean = xf.mean(-1)
        rstd = torch.rsqrt(xf.var(-1, unbiased=False) + eps)
        ctx.save_for_backward(x, mean, rstd, ln_weight, ln_bias, weight)
        ctx.has_bias = bias is not None
        # Rows, not `length`: see BOTH_ROWS. x is (M, K) here, so M is readable directly.
        ctx.shape_key = both_key(x.reshape(-1, x.shape[-1]).shape[0])
        return Y

    @staticmethod
    def backward(ctx, dY):
        x, mean, rstd, gamma, beta, W = ctx.saved_tensors
        dx, dg, db_ln, dW, db = _compose_backward(
            dY, x, mean, rstd, gamma, beta, W, ctx.has_bias, dx_via_quack=False,
            shape_key=ctx.shape_key,
        )
        return dx, dg, db_ln, dW, db, None, None


def layernorm_linear_triton_fn(x, ln_weight, ln_bias, weight, bias=None, eps: float = 1e-5,
                               length: int | None = None):
    """Trainable LayerNormLinear, portable (Triton fwd + cuBLAS/Triton bwd, no quack).

    ``length`` is L (see ``layernorm_linear_fn``) -- the pre-flatten token/atom count, used only
    as the backward's autotune-cache label."""
    return LayerNormLinearTritonFn.apply(x, ln_weight, ln_bias, weight, bias, eps, length)

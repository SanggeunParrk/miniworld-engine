"""Fused LayerNormLinear backward kernels (Triton).

The composed backward does, separately:
  1. dx_normed = dY @ W          (GEMM)
  2. x_normed  = recompute       (elementwise)
  3. dW        = dYᵀ @ x_normed  (GEMM, wgrad)
  4. dx,dγ,dβ  = LN-backward(dx_normed, ...)

This module fuses **1+4** and **2+3** to cut HBM round-trips:
  - ``dgrad_lnbwd`` (1+4): one kernel does dY@W and the LN-backward in its epilogue, so the
    (M,K) dx_normed is never written/read back. Needs the full K-row in one block (small/mid K).
  - ``xnorm_wgrad`` (2+3): one kernel computes x_normed = (x-mean)·rstd·γ+β on-the-fly while
    accumulating dW = dYᵀ@x_normed, so x_normed is never materialized. Small (N,K) output tiles
    give many CTAs (the wgrad aspect quack starves on).

MEASURED VERDICT (H100 bf16, vs cuBLAS-GEMM + separate Triton LN-bwd/recompute) — both LOSE:
  - 2+3 (xnorm_wgrad): HOPELESS — 3.8-16x slower. The wgrad aspect needs split-K, which cuBLAS
    has and Triton/quack/cute-DSL do not (cuequiv also routes wgrad through cuBLAS). No fusion
    beats cuBLAS here; abandoned.
  - 1+4 (dgrad_lnbwd): wins only at d=128 tiny-M (20.8 vs 26.4µs); 1.7-2.5x slower at large d/M.
    The loss is the Triton dgrad GEMM (dY@W) being slower than cuBLAS — the fusion saving (no
    dx_normed round-trip) can't offset it. The CONCEPT could still win on a fast GEMM: forking
    quack's GemmSm90 (dgrad ties cuBLAS there) with an LN-backward epilogue — M2-level effort,
    untried. This Triton version is a documented negative result; not wired into the backward.

Conclusion: the shipping backward stays cuBLAS GEMMs + separate Triton LN-bwd (see autograd.py).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_kernels.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of


# ============================== 1+4 : dgrad + LN-backward ==============================
_layernorm_linear_fused_bwd_dgrad_lnbwd_prune = make_cache_prune(
    "layernorm_linear_fused_bwd_dgrad_lnbwd", dtype_of=tensor_dtype_of("dY_ptr"),
    bucket_of=key_bucket_of("N", "K"),
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bm in (16, 32, 64)
        for bn in (32, 64, 128)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["N", "K"],
    prune_configs_by={"early_config_prune": _layernorm_linear_fused_bwd_dgrad_lnbwd_prune},
)
@triton.jit
def _dgrad_lnbwd_kernel(
    dY_ptr, W_ptr, x_ptr, g_ptr, mean_ptr, rstd_ptr,
    dx_ptr, dgamma_ptr, dbeta_ptr,
    M, N, K,
    s_dym, s_dyn, s_wn, s_wk, s_xm, s_xk, s_dxm, s_dxk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rm = rows < M
    k = tl.arange(0, BLOCK_K)
    km = k < K

    # dx_normed = dY @ W  — accumulate over the N contraction
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        nn = n0 + tl.arange(0, BLOCK_N)
        nmask = nn < N
        dY_blk = tl.load(dY_ptr + rows[:, None] * s_dym + nn[None, :] * s_dyn,
                         mask=rm[:, None] & nmask[None, :], other=0.0)
        W_blk = tl.load(W_ptr + nn[:, None] * s_wn + k[None, :] * s_wk,
                        mask=nmask[:, None] & km[None, :], other=0.0)
        acc += tl.dot(dY_blk, W_blk)
    dxn = acc  # (BLOCK_M, K) fp32 = dx_normed

    # LN backward: x̂ = (x-mean)·rstd ; dx̂ = dxn·γ
    x = tl.load(x_ptr + rows[:, None] * s_xm + k[None, :] * s_xk,
                mask=rm[:, None] & km[None, :], other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=rm, other=0.0)[:, None]
    rstd = tl.load(rstd_ptr + rows, mask=rm, other=0.0)[:, None]
    xhat = tl.where(km[None, :], (x - mean) * rstd, 0.0)
    g = tl.load(g_ptr + k, mask=km, other=0.0).to(tl.float32)[None, :]
    dxhat = dxn * g
    inv_k = 1.0 / K
    c2 = tl.sum(tl.where(km[None, :], dxhat, 0.0), axis=1) * inv_k       # meanₖ(dx̂)
    c1 = tl.sum(tl.where(km[None, :], dxhat * xhat, 0.0), axis=1) * inv_k  # meanₖ(dx̂·x̂)
    dx = rstd * (dxhat - c2[:, None] - xhat * c1[:, None])
    tl.store(dx_ptr + rows[:, None] * s_dxm + k[None, :] * s_dxk,
             dx.to(dx_ptr.dtype.element_ty), mask=rm[:, None] & km[None, :])

    # dγ = Σ_m dxn·x̂ ; dβ = Σ_m dxn  (reduce over this block's M rows, atomic across blocks)
    dg = tl.sum(tl.where(rm[:, None], dxn * xhat, 0.0), axis=0)
    db = tl.sum(tl.where(rm[:, None], dxn, 0.0), axis=0)
    tl.atomic_add(dgamma_ptr + k, dg, mask=km)
    tl.atomic_add(dbeta_ptr + k, db, mask=km)


def dgrad_lnbwd(dY, W, x, gamma, mean, rstd):
    """Fused 1+4: returns dx (M,K), dgamma (K,), dbeta (K,). dY (M,N), W (N,K)."""
    M, N = dY.shape
    K = W.shape[1]
    assert W.shape[0] == N
    dx = torch.empty(M, K, device=dY.device, dtype=dY.dtype)
    dgamma = torch.zeros(K, dtype=torch.float32, device=dY.device)
    dbeta = torch.zeros(K, dtype=torch.float32, device=dY.device)
    dY = dY.contiguous(); W = W.contiguous(); x = x.contiguous()
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _dgrad_lnbwd_kernel[grid](
        dY, W, x, gamma, mean, rstd, dx, dgamma, dbeta,
        M, N, K,
        dY.stride(0), dY.stride(1), W.stride(0), W.stride(1),
        x.stride(0), x.stride(1), dx.stride(0), dx.stride(1),
        BLOCK_K=triton.next_power_of_2(K),
    )
    return dx, dgamma, dbeta


# ============================== 2+3 : x_normed prologue + wgrad ==============================
_layernorm_linear_fused_bwd_xnorm_wgrad_prune = make_cache_prune(
    "layernorm_linear_fused_bwd_xnorm_wgrad", dtype_of=tensor_dtype_of("dY_ptr"),
    bucket_of=key_bucket_of("N", "K"),
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk, "BLOCK_M": bm}, num_warps=nw, num_stages=ns)
        for bn in (32, 64, 128)
        for bk in (32, 64, 128)
        for bm in (32, 64)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["N", "K"],
    prune_configs_by={"early_config_prune": _layernorm_linear_fused_bwd_xnorm_wgrad_prune},
)
@triton.jit
def _xnorm_wgrad_kernel(
    dY_ptr, x_ptr, g_ptr, b_ptr, mean_ptr, rstd_ptr, dW_ptr,
    M, N, K,
    s_dym, s_dyn, s_xm, s_xk, s_dwn, s_dwk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    nn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    nmask = nn < N
    kmask = kk < K
    g = tl.load(g_ptr + kk, mask=kmask, other=0.0).to(tl.float32)[None, :]
    b = tl.load(b_ptr + kk, mask=kmask, other=0.0).to(tl.float32)[None, :]

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        mm = m0 + tl.arange(0, BLOCK_M)
        mmask = mm < M
        dY_blk = tl.load(dY_ptr + mm[:, None] * s_dym + nn[None, :] * s_dyn,
                         mask=mmask[:, None] & nmask[None, :], other=0.0)  # (BM, BN)
        x_blk = tl.load(x_ptr + mm[:, None] * s_xm + kk[None, :] * s_xk,
                        mask=mmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)  # (BM, BK)
        mean = tl.load(mean_ptr + mm, mask=mmask, other=0.0)[:, None]
        rstd = tl.load(rstd_ptr + mm, mask=mmask, other=0.0)[:, None]
        xn = ((x_blk - mean) * rstd * g + b).to(dY_blk.dtype)  # x_normed tile (BM, BK)
        acc += tl.dot(tl.trans(dY_blk), xn)  # (BN, BK) += dYᵀ @ xn
    tl.store(dW_ptr + nn[:, None] * s_dwn + kk[None, :] * s_dwk,
             acc.to(dW_ptr.dtype.element_ty), mask=nmask[:, None] & kmask[None, :])


def xnorm_wgrad(dY, x, gamma, beta, mean, rstd, K):
    """Fused 2+3: dW = dYᵀ @ LN(x). dY (M,N), x (M,K) -> dW (N,K)."""
    M, N = dY.shape
    dW = torch.empty(N, K, device=dY.device, dtype=dY.dtype)
    dY = dY.contiguous(); x = x.contiguous()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(K, meta["BLOCK_K"]))  # noqa: E731
    _xnorm_wgrad_kernel[grid](
        dY, x, gamma, beta, mean, rstd, dW,
        M, N, K,
        dY.stride(0), dY.stride(1), x.stride(0), x.stride(1), dW.stride(0), dW.stride(1),
    )
    return dW

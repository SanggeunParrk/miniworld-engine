"""Portable Triton fused LayerNorm + Linear forward.

This is the GENERAL fallback for the LayerNormLinear op: the cute backend
(`gemm_layernorm_linear*.py`) forks quack's ``GemmSm90`` and uses WGMMA + TMA +
clusters, so it is **SM90 (Hopper: H100/H200) only** and asserts on anything
else. Triton compiles per-arch, so this kernel runs on Ampere (sm_80), Ada
(sm_89), Hopper, Blackwell, and ROCm — wherever Triton + tl.dot are supported.

Computes ``Y = LayerNorm(x) @ W^T + b`` directly (no parameter fold): each
program loads its M rows' full K=d_in vector, reduces mean/var on-chip (one pass,
fp32), normalizes with (gamma, beta), then matmuls the (BLOCK_M1, K) normalized
tile against a (K, BLOCK_N) slice of W (acc in fp32, matching the bf16 reference's
fp32 accumulation). Requires K to fit one block (BLOCK_K = next_pow2(K), so
K <= 1024 in practice — true for all pair/single hidden dims here); the wrapper
falls back to eager torch for larger K.
"""

from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for

import torch
import triton
import triton.language as tl


from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of



# BLOCK_K used to arrive as ``BLOCK_K=next_power_of_2(K)`` from the launcher -- a whole-row
# constant the tuner never saw. It is a CSV tile, reaching
# 1024 == the wrapper's K ceiling, so "one tile holds the whole row" (the schedule this
# kernel was written for) is still reachable; every smaller candidate is made correct by the
# k-loops below rather than silently wrong.
from miniworld_engine.autotune.buckets import bucket_mixed as _bucket
from miniworld_engine.autotune.shape_key import both_key, length_of


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


@triton.autotune(configs=configs_for("layernorm_linear_fwd_triton"), key=['N', 'K', 'shape_key'])
@triton.jit
def _lnl_fwd_kernel(
    x_ptr, w_ptr, b_ptr, g_ptr, beta_ptr, y_ptr,
    # K is tl.constexpr (d_in, fixed per module, and already in this kernel's autotune key) so the
    # `BLOCK_K >= K` guard below resolves at COMPILE time and only one branch is emitted.
    M, N, K: tl.constexpr, eps,
    stride_xm, stride_xk,
    stride_wn, stride_wk,  # W is (N, K) row-major: stride_wn=K, stride_wk=1
    stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, shape_key,
):
    # One program owns BLOCK_M1 rows and ALL of N: the LayerNorm STATISTICS are computed once
    # per row and reused across the N-loop (vs recomputing them per (M,N) tile). Grid is 1-D
    # over M-blocks. BLOCK_K tiles the d_in axis; when the tuner picks BLOCK_K >= K both
    # k-loops below are a single iteration and this is exactly the original whole-row schedule.
    pid_m = tl.program_id(0).to(tl.int64)
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M

    inv_k = 1.0 / K

    if BLOCK_K >= K:
        # COVERING TILE -> the original whole-row schedule: the full K row is read ONCE, the
        # normalized bf16 tile `xn` is built once and held in registers, and the N-loop only
        # streams W. The general branch below re-reads x in both statistics sweeps AND once per
        # N-tile (2 + N/BLOCK_N reads per row). Numerics are identical to that branch at
        # BLOCK_K >= K: its loops are single-trip and the arithmetic matches term for term.
        k = tl.arange(0, BLOCK_K)
        k_mask = k < K
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)
        mean = tl.sum(x, axis=1) * inv_k
        xc = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) * inv_k
        rstd = tl.rsqrt(var + eps)
        g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        xn = (xc * rstd[:, None] * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)
        for n0 in range(0, N, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            acc = tl.zeros([BLOCK_M1, BLOCK_N], dtype=tl.float32)
            w = tl.load(  # (BLOCK_K, BLOCK_N): w[k, n] = W[cols[n], k]
                w_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                mask=k_mask[:, None] & col_mask[None, :], other=0.0,
            )
            acc = tl.dot(xn, w, acc, out_dtype=tl.float32)  # bf16xbf16 -> fp32 acc
            if HAS_BIAS:
                acc += tl.load(b_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)[None, :]
            tl.store(
                y_ptr + rows[:, None] * stride_ym + cols[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=row_mask[:, None] & col_mask[None, :],
            )
    else:
        # --- pass 1: row statistics over K-tiles. Two sweeps (mean, then CENTERED variance) rather
        # than sum/sum-of-squares, so at BLOCK_K >= K the fp32 algebra is exactly the original's. ---
        s = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < K
            x = tl.load(
                x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            s += tl.sum(x, axis=1)
        mean = s * inv_k
        s = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < K
            x = tl.load(
                x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            xc = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
            s += tl.sum(xc * xc, axis=1)
        var = s * inv_k
        rstd = tl.rsqrt(var + eps)

        # --- pass 2: loop the projection over N-tiles; each N-tile contracts over the K-tiles,
        # normalizing the x tile in-register (never materializing the normalized row). ---
        for n0 in range(0, N, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            acc = tl.zeros([BLOCK_M1, BLOCK_N], dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                k_mask = k < K
                x = tl.load(
                    x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
                g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
                beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
                xc = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
                xn = (xc * rstd[:, None] * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)
                w = tl.load(  # (BLOCK_K, BLOCK_N): w[k, n] = W[cols[n], k]
                    w_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                    mask=k_mask[:, None] & col_mask[None, :], other=0.0,
                )
                acc = tl.dot(xn, w, acc, out_dtype=tl.float32)  # bf16xbf16 -> fp32 acc
            if HAS_BIAS:
                acc += tl.load(b_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)[None, :]
            tl.store(
                y_ptr + rows[:, None] * stride_ym + cols[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=row_mask[:, None] & col_mask[None, :],
            )


def layernorm_linear_triton_fwd(
    x: torch.Tensor,          # (..., K) = (..., d_in)
    ln_weight: torch.Tensor,  # (K,)
    ln_bias: torch.Tensor,    # (K,)
    weight: torch.Tensor,     # (N, K) = (d_out, d_in)
    bias: torch.Tensor | None,  # (N,)
    eps: float = 1e-5,
) -> torch.Tensor:
    """Portable Triton ``LayerNorm(x) @ W^T + b``. Falls back to eager torch if K>1024."""
    assert x.is_cuda and weight.is_cuda
    K = x.shape[-1]
    N = weight.shape[0]
    assert weight.shape[1] == K, f"weight (N,K) mismatch: {tuple(weight.shape)} vs K={K}"
    x2 = x.reshape(-1, K)
    M = x2.shape[0]

    if K > 1024:  # one-block-K assumption broken; correctness-first eager fallback
        import torch.nn.functional as F
        y = F.linear(F.layer_norm(x2.float(), (K,), ln_weight.float(), ln_bias.float(), eps),
                     weight.float(), None if bias is None else bias.float())
        if bias is not None:
            pass
        return y.to(x.dtype).reshape(*x.shape[:-1], N)

    x2 = x2.contiguous()
    w = weight.contiguous()
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731  (1-D: N looped in-kernel)
    _lnl_fwd_kernel[grid](
        x2, w, bias if bias is not None else x2, ln_weight.contiguous(), ln_bias.contiguous(), y,
        M, N, K, eps,
        x2.stride(0), x2.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        # L = x.shape[-2], read BEFORE the reshape to (M, K) -- one rule for pair
        # (B, L, L, D) and token/atom (B, L, D). Never M.
        HAS_BIAS=bias is not None, shape_key=both_key(length_of(x.shape)),
    )
    return y.reshape(*x.shape[:-1], N)

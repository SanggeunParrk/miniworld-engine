"""Portable Triton fused LayerNorm + Linear forward.

This is the GENERAL fallback for the LayerNormLinear op: the cute backend
(`gemm_layernorm_linear*.py`) forks quack's ``GemmSm90`` and uses WGMMA + TMA +
clusters, so it is **SM90 (Hopper: H100/H200) only** and asserts on anything
else. Triton compiles per-arch, so this kernel runs on Ampere (sm_80), Ada
(sm_89), Hopper, Blackwell, and ROCm — wherever Triton + tl.dot are supported.

Computes ``Y = LayerNorm(x) @ W^T + b`` directly (no parameter fold): each
program loads its M rows' full K=d_in vector, reduces mean/var on-chip (one pass,
fp32), normalizes with (gamma, beta), then matmuls the (BLOCK_M, K) normalized
tile against a (K, BLOCK_N) slice of W (acc in fp32, matching the bf16 reference's
fp32 accumulation). Requires K to fit one block (BLOCK_K = next_pow2(K), so
K <= 1024 in practice — true for all pair/single hidden dims here); the wrapper
falls back to eager torch for larger K.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bm in (16, 32, 64, 128)
        for bn in (64, 128, 256)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["N", "K"],
)
@triton.jit
def _lnl_fwd_kernel(
    x_ptr, w_ptr, b_ptr, g_ptr, beta_ptr, y_ptr,
    M, N, K, eps,
    stride_xm, stride_xk,
    stride_wn, stride_wk,  # W is (N, K) row-major: stride_wn=K, stride_wk=1
    stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program owns BLOCK_M rows and ALL of N: LayerNorm is computed ONCE per row
    # and reused across the N-loop (vs recomputing it per (M,N) tile, which re-reads X
    # and tanks large-K throughput). Grid is 1-D over M-blocks.
    pid_m = tl.program_id(0).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    row_mask = rows < M
    k_mask = k < K

    # --- load the full K row once, layernorm on-chip (one pass, fp32) ---
    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
        mask=row_mask[:, None] & k_mask[None, :], other=0.0,
    ).to(tl.float32)
    inv_k = 1.0 / K
    mean = tl.sum(x, axis=1) * inv_k
    xc = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) * inv_k
    rstd = tl.rsqrt(var + eps)
    g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
    xn = (xc * rstd[:, None] * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)  # (BM, BK)

    # --- loop the projection over N-tiles, reusing the one normalized X tile ---
    for n0 in range(0, N, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        w = tl.load(  # (BLOCK_K, BLOCK_N): w[k, n] = W[cols[n], k]
            w_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        acc = tl.dot(xn, w, out_dtype=tl.float32)  # bf16xbf16 -> fp32 acc
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
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731  (1-D: N looped in-kernel)
    _lnl_fwd_kernel[grid](
        x2, w, bias if bias is not None else x2, ln_weight.contiguous(), ln_bias.contiguous(), y,
        M, N, K, eps,
        x2.stride(0), x2.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_K=triton.next_power_of_2(K),
    )
    return y.reshape(*x.shape[:-1], N)

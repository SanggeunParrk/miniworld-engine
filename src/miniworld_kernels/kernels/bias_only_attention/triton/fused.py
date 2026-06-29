"""Fused bias-only triangle-attention -- NEGATIVE RESULT, kept as a signpost.

This strided-gather GEMM is ~8x SLOWER than torch.einsum (correctness is fine,
cosine 1.0). Reason: it reads value as V'[k,(i,d)] to avoid torch's permute, but
value[i,k,d] has i-stride = L*D, so the per-k load is badly non-coalesced. torch's
permute -> contiguous -> cuBLAS path wins decisively. Do NOT revive the
"avoid the permute via strided/gathered loads" idea -- it loses. The op-level
winner is plain torch.einsum; the real wins are module-level (LN + .contiguous +
gate). The H100 crossover is documented in the benchmark history and summarized
in the dispatch policy docs.

Op:  out[b,h,i,j,d] = sum_k softmax_k(bias[b,h,j,k]) * value[b,h,i,k,d]

Strategy (see memory bias-only-attention-baseline):
  - k is both the contraction axis and the softmax axis.
  - softmax(bias) does NOT depend on i, so compute A = softmax(bias) ONCE.
  - the contraction is a single big GEMM per (b,h):
        O[j, f]  where f = (i, d),   O[j,f] = sum_k A[j,k] * V'[k,f],  V'[k,(i,d)] = value[i,k,d]
  - torch_einsum already does this GEMM but pays a value-permute (in) and an
    output-permute (out).  This kernel reads value STRIDED and writes output
    STRIDED, killing both permute round-trips.

This first cut precomputes A with torch.softmax; the GEMM is the triton kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

gemm_configs = [
    triton.Config({"BLOCK_J": bj, "BLOCK_I": bi, "BLOCK_K": bk}, num_warps=w, num_stages=s)
    for bj in [32, 64]
    for bi in [1, 2, 4]
    for bk in [32, 64]
    for w in [4, 8]
    for s in [2, 3]
]


@triton.autotune(configs=gemm_configs, key=["L", "D"])
@triton.jit
def _bias_only_gemm(
    a_ptr,        # A = softmax(bias)  [B,H,Lj,Lk]
    v_ptr,        # value              [B,H,Li,Lk,D]
    o_ptr,        # out                [B,H,Li,Lj,D]
    stride_az, stride_ah, stride_aj, stride_ak,
    stride_vz, stride_vh, stride_vi, stride_vk, stride_vd,
    stride_oz, stride_oh, stride_oi, stride_oj, stride_od,
    H: tl.constexpr,
    L,
    D: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_j = tl.program_id(0)
    pid_i = tl.program_id(1)
    off_zh = tl.program_id(2).to(tl.int64)
    off_z = off_zh // H
    off_h = off_zh % H

    a_base = a_ptr + off_z * stride_az + off_h * stride_ah
    v_base = v_ptr + off_z * stride_vz + off_h * stride_vh
    o_base = o_ptr + off_z * stride_oz + off_h * stride_oh

    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)        # rows of output (j)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)        # i block
    offs_d = tl.arange(0, BLOCK_D)                          # d (padded to BLOCK_D)

    # free axis f = (i_local, d) -> column tile of width BLOCK_I*BLOCK_D
    # value column offset for (i_local, d): i*stride_vi + d*stride_vd
    col_v = offs_i[:, None] * stride_vi + offs_d[None, :] * stride_vd   # [BLOCK_I, BLOCK_D]
    col_o = offs_i[:, None] * stride_oi + offs_d[None, :] * stride_od   # [BLOCK_I, BLOCK_D]
    d_ok = offs_d < D
    i_ok = offs_i < L

    acc = tl.zeros([BLOCK_J, BLOCK_I * BLOCK_D], dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    for k0 in range(0, L, BLOCK_K):
        kk = k0 + offs_k
        k_ok = kk < L
        # A tile [BLOCK_J, BLOCK_K]
        a = tl.load(
            a_base + offs_j[:, None] * stride_aj + kk[None, :] * stride_ak,
            mask=(offs_j[:, None] < L) & k_ok[None, :],
            other=0.0,
        )
        # value tile V'[k, (i,d)] -> [BLOCK_K, BLOCK_I*BLOCK_D]
        v_off = kk[:, None, None] * stride_vk + col_v[None, :, :]       # [BK, BI, BD]
        v = tl.load(
            v_base + tl.reshape(v_off, (BLOCK_K, BLOCK_I * BLOCK_D)),
            mask=(k_ok[:, None, None] & i_ok[None, :, None] & d_ok[None, None, :]).reshape(
                BLOCK_K, BLOCK_I * BLOCK_D
            ),
            other=0.0,
        )
        acc = tl.dot(a.to(v.dtype), v, acc)

    # store O[j, (i,d)] -> out[b,h,i,j,d]
    o_off = offs_j[:, None, None] * stride_oj + col_o[None, :, :]       # [BJ, BI, BD]
    o_mask = (
        (offs_j[:, None, None] < L) & i_ok[None, :, None] & d_ok[None, None, :]
    ).reshape(BLOCK_J, BLOCK_I * BLOCK_D)
    tl.store(
        o_base + tl.reshape(o_off, (BLOCK_J, BLOCK_I * BLOCK_D)),
        acc.to(o_ptr.dtype.element_ty),
        mask=o_mask,
    )


def bias_only_fused_fwd(value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """value [B,H,L,L,D], bias [B,H,L,L] -> out [B,H,L,L,D]."""
    B, H, L, _, D = value.shape
    a = torch.softmax(bias.float(), dim=-1).to(value.dtype)  # [B,H,Lj,Lk]
    out = torch.empty_like(value)
    BLOCK_D = triton.next_power_of_2(D)
    grid = lambda META: (
        triton.cdiv(L, META["BLOCK_J"]),
        triton.cdiv(L, META["BLOCK_I"]),
        B * H,
    )
    _bias_only_gemm[grid](
        a, value, out,
        *a.stride(),
        *value.stride(),
        *out.stride(),
        H, L, D,
        BLOCK_D=BLOCK_D,
    )
    return out

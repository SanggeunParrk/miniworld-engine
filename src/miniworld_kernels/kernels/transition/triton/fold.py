"""Fused Triton weight-fold for the gated Transition expand (TRAINING path).

``fold_swiglu`` (the torch reference) builds, from Wa,Wb (N,K), gamma,beta (K,), the
interleaved gated GEMM operands the folded expand epilogue consumes:

    B[2j]   = (gamma ⊙ Wa[j]).bf16     B[2j+1] = (gamma ⊙ Wb[j]).bf16   -> (2N, K)
    S[2j]   = Σ_k B[2j,k]              S[2j+1] = Σ_k B[2j+1,k]          -> (2N,) f32
    B2[2j]  = Σ_k Wa[j,k]·beta[k]      B2[2j+1]= Σ_k Wb[j,k]·beta[k]    -> (2N,) f32

The torch version launches ~20 tiny ops (two scaled casts, two strided scatters into
B, two rowsums, two matvecs, two more scatters) — launch-bound at ~141us. In TRAINING
the weights change every optimizer step so this CANNOT be cached and is paid per
forward. This single kernel does it in ONE pass over the weights: each program owns a
block of j-rows, loads Wa[j,:]/Wb[j,:]/gamma/beta (K-wide), computes the folded rows +
their two reductions, and writes B (already interleaved), S, B2. Target < 15us.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_J": bj}, num_warps=nw)
        for bj in (8, 16, 32, 64, 128)
        for nw in (1, 2, 4, 8)
    ],
    key=["N", "K"],
)
@triton.jit
def _fold_kernel(
    wa_ptr, wb_ptr, gamma_ptr, beta_ptr,
    b_ptr, s_ptr, b2_ptr,
    N, K,
    stride_wn, stride_wk,   # Wa/Wb share strides (both (N,K) contiguous)
    stride_bn, stride_bk,   # B is (2N, K)
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    j = pid * BLOCK_J + tl.arange(0, BLOCK_J)        # (BLOCK_J,) over N
    k = tl.arange(0, BLOCK_K)                         # (BLOCK_K,) over K
    j_mask = j < N
    k_mask = k < K
    mask = j_mask[:, None] & k_mask[None, :]

    w_off = j[:, None] * stride_wn + k[None, :] * stride_wk
    wa = tl.load(wa_ptr + w_off, mask=mask, other=0.0).to(tl.float32)   # (BJ, BK)
    wb = tl.load(wb_ptr + w_off, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + k, mask=k_mask, other=0.0).to(tl.float32)  # (BK,)
    beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)

    # folded gated rows (gamma-scaled), cast to bf16 for B.
    ba = wa * gamma[None, :]
    bb = wb * gamma[None, :]
    # reductions over K.
    sa = tl.sum(ba, axis=1)            # S gate
    sb = tl.sum(bb, axis=1)            # S up
    b2a = tl.sum(wa * beta[None, :], axis=1)   # Wa@beta
    b2b = tl.sum(wb * beta[None, :], axis=1)   # Wb@beta

    # write B interleaved: row 2j = gate, 2j+1 = up.
    b_off_a = (2 * j[:, None]) * stride_bn + k[None, :] * stride_bk
    b_off_b = (2 * j[:, None] + 1) * stride_bn + k[None, :] * stride_bk
    tl.store(b_ptr + b_off_a, ba.to(b_ptr.dtype.element_ty), mask=mask)
    tl.store(b_ptr + b_off_b, bb.to(b_ptr.dtype.element_ty), mask=mask)

    # write S / B2 interleaved (f32).
    tl.store(s_ptr + 2 * j, sa, mask=j_mask)
    tl.store(s_ptr + 2 * j + 1, sb, mask=j_mask)
    tl.store(b2_ptr + 2 * j, b2a, mask=j_mask)
    tl.store(b2_ptr + 2 * j + 1, b2b, mask=j_mask)


def fold_swiglu_triton(
    Wa: torch.Tensor,        # (N, K)
    Wb: torch.Tensor,        # (N, K)
    ln_weight: torch.Tensor, # (K,) gamma
    ln_bias: torch.Tensor,   # (K,) beta
    *,
    w2_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-kernel fused fold. Returns (B (2N,K) w2_dtype, S (2N,) f32, B2 (2N,) f32)."""
    assert Wa.is_cuda and Wa.shape == Wb.shape and Wa.dim() == 2
    Wa = Wa.contiguous(); Wb = Wb.contiguous()
    N, K = Wa.shape
    B = torch.empty(2 * N, K, dtype=w2_dtype, device=Wa.device)
    S = torch.empty(2 * N, dtype=torch.float32, device=Wa.device)
    B2 = torch.empty(2 * N, dtype=torch.float32, device=Wa.device)
    BLOCK_K = triton.next_power_of_2(K)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_J"]),)
    _fold_kernel[grid](
        Wa, Wb, ln_weight.contiguous(), ln_bias.contiguous(),
        B, S, B2,
        N, K,
        Wa.stride(0), Wa.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_K=BLOCK_K,
    )
    return B, S, B2

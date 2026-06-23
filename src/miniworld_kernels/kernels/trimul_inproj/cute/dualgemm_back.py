"""Host side for the cute dual-gemm fused back-half (gate computed IN the kernel).

Builds the operands for a SINGLE gated GEMM that computes, with NO gate
materialization:

    proj_j = LN_D(tri) @ (gamma*Wp)_j         (LN folded; stats over tri's D)
    gate_j = sigmoid(x_n @ Wg_j)
    y_j    = proj_j * gate_j

via an interleaved block-diagonal B so the accumulator lays out as adjacent
(proj_j, gate_j) pairs — letting the gated (glu-style) halving epilogue combine
each pair with a CUSTOM act = (rstd*p - c1*S + B2) * sigmoid(g).

    A = [tri | x_n]  (M, 2D)   K = 2D
    B (2D, 2D):  B[:, 2j]   = [ (gamma*Wp)[:,j] ; 0      ]   -> acc[:,2j]   = tri @ (gamma*Wp)_j
                 B[:, 2j+1] = [ 0               ; Wg[:,j] ]   -> acc[:,2j+1] = x_n @ Wg_j
    mX = tri (M, D)            (LN stats over D, NOT over the 2D of A)
    S, B2 : per proj col j, folded from (gamma*Wp) and beta  (fold_for_gemm)

This module provides the operand construction; the kernel side adds a gated
halving epilogue with the LN-correct×sigmoid act (see gemm_layernorm_linear_fused
gated-dual mode). B=1, D=128, bf16.
"""

from __future__ import annotations

import torch


def build_dualgemm_operands(WL_unused, x_n, tri_bdll, Wp, Wg, ln_w, ln_b, eps=1e-5):
    """Return (A, B_blockdiag_interleaved, S, B2, mX) for the dual-gemm back kernel.

    Wp, Wg : (D, D) in x@W form (= weight.T).  tri_bdll: (B,D,L,L). x_n: (B,L,L,D).
    """
    from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear import (
        fold_for_gemm,
    )

    B, D, L, _ = tri_bdll.shape
    assert B == 1
    M = L * L
    dev, dt = x_n.device, x_n.dtype

    tri_md = tri_bdll.reshape(D, M).t().contiguous()   # (M, D)  channel vec per row
    xn_md = x_n.reshape(M, D)                          # (M, D)
    A = torch.cat([tri_md, xn_md], dim=1).contiguous()  # (M, 2D)  [tri | x_n]

    # fold gamma into Wp -> W2 (K=D, N=D), and S/B2 for the LN correction on proj.
    # fold_for_gemm expects nn.Linear weight (N, K); Wp is (K, N) = weight.T -> pass Wp.T
    W2, S, B2 = fold_for_gemm(Wp.t().contiguous(), ln_w, ln_b, None, w2_dtype=dt)  # W2:(K=D,N=D)

    # interleaved block-diag B (2D, 2D): even col 2j = proj (top half = W2[:,j]),
    # odd col 2j+1 = gate (bottom half = Wg[:,j]).
    Bm = torch.zeros(2 * D, 2 * D, device=dev, dtype=dt)
    Bm[:D, 0::2] = W2.t()      # proj: W2 from fold is (N,K) -> (K,N) for the K-block
    Bm[D:, 1::2] = Wg          # gate logits use x_n (bottom K block); Wg already (K,N)

    return A, Bm, S, B2, tri_md  # mX = tri_md (stats over D)

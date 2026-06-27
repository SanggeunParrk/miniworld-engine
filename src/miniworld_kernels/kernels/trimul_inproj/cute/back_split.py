"""SPLIT trimul back-half: ① cute LayerNormLinear  +  ② triton GateElem.

Contrast with the single fused `triton/back.py` (LN_out + @Wp + gate + mul in ONE
kernel = two GEMMs in one program → blows regs/shared at D≥256). Here the back is
two kernels, each with ONE GEMM:

    ① proj = LN_out(tri) @ Wp          (cute layernorm_linear_cute_fused; stats in-GEMM)
    ② y    = proj ⊙ sigmoid(x_n @ Wg)  (triton gate_elem; gate computed in-kernel)

Reuses the tuned cute layernorm_linear (TE-beating) for ①; ② is the new light
triton kernel. Costs an extra HBM round trip on `proj` vs the fused back, but each
kernel is half the reg/shared pressure → the hope is D≥256 compiles where the
single kernel does not. B=1, bf16.

Weight forms (mind the transpose):
  Wp_nn = to_out.weight  — nn.Linear (N, K) form  (cute LNL wants this, NOT .T)
  Wg_t  = to_gate.weight.T — x@W form (K, N)       (triton gate_elem wants this)
"""

from __future__ import annotations

import torch

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_kernels.kernels.trimul_inproj.cute import dispatch
from miniworld_kernels.kernels.trimul_inproj.triton.gate_elem import (
    gate_elem_quack_fused, gate_elem_triton,
)


def default_lnl_config(N):
    """tile_n tiles the OUTPUT N (= d_pair): tile_n=128 covers N∈{128,256,512}; N=64 -> 64.
    (K, the LN'd contraction dim, is handled by the GEMM K-loop — not tile_n.)"""
    tile_n = 64 if N < 128 else 128
    return dict(tile_m=128, tile_n=tile_n, cluster_m=1, cluster_n=1, pingpong=True)


def trimul_back_split(tri_bdll, x_n, Wp_nn, Wg_t, ln_w, ln_b, eps=1e-5, lnl_config=None):
    """tri_bdll:(B,K,L,L) with K=hidden (=D for square trimul, =2*d_hidden for bidir),
    x_n:(B,L,L,d_pair), Wp_nn:(d_pair,K)=to_out.weight (N,K), Wg_t:(d_pair,d_pair)=
    to_gate.weight.T -> y:(B,L,L,d_pair). B=1."""
    B, K, L, L2 = tri_bdll.shape
    assert B == 1 and L == L2
    N = Wp_nn.shape[0]                                             # output width = d_pair
    M = L * L
    if lnl_config is None:
        lnl_config = default_lnl_config(N)
    # ① cute LayerNormLinear: M-major view of tri (channel strided by M), no copy.
    #    LN over K channels, then @Wp (K -> N). K may differ from N (bidirectional).
    view = tri_bdll.reshape(B, K, M)[0].t()                       # (M, K)
    proj = layernorm_linear_cute_fused(view, ln_w, ln_b, Wp_nn, None, eps=eps,
                                       config=lnl_config)          # (M, N)
    # ② gate: dispatch fused-quack (act(A@B)⊙C, one launch) vs triton (cuBLAS gemm + ew),
    #    cache the per-shape winner (fused wins large L; triton can win tiny L).
    y = dispatch.pick("gate_infer", (M, N),
                      [("fused", lambda: gate_elem_quack_fused(x_n, proj, Wg_t)),
                       ("triton", lambda: gate_elem_triton(x_n, proj, Wg_t))])
    return y.view(B, L, L, N)

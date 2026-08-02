"""SM100 (B200) SPLIT trimul back-half — CUEQUIV-FREE.

Mirror of ``back_split.py`` (the H100 design) for Blackwell:

    ① proj = LN_out(tri) @ Wp        (sm100 LayerNormLinear, two-kernel: triton
                                       M-major LN + tm1 tcgen05 proj GEMM)
    ② y    = proj ⊙ sigmoid(x_n @ Wg)  (triton gate_elem; gate GEMM = cuBLAS)

NO cuequiv, NO quack. B=1, bf16 in / fp32 acc / bf16 out.

Weight forms (mind the transpose):
  Wp_nn = to_out.weight   — nn.Linear (N, K) form  (proj GEMM wants this)
  Wg_t  = to_gate.weight.T — x@W form (K, N)        (triton gate_elem wants this)
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.layernorm_linear.cute.ln_linear_sm100 import (
    layernorm_linear_sm100,
)
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_triton


def trimul_back_split_sm100(tri_bkll, x_n, Wp_nn, Wg_t, ln_w, ln_b, eps=1e-5):
    """tri_bkll:(B,K,L,L) B=1, x_n:(B,L,L,d_pair), Wp_nn:(N,K)=to_out.weight,
    Wg_t:(d_pair,d_pair)=to_gate.weight.T -> y:(B,L,L,N)."""
    B, K, L, L2 = tri_bkll.shape
    assert B == 1 and L == L2
    N = Wp_nn.shape[0]
    proj = layernorm_linear_sm100(tri_bkll, ln_w, ln_b, Wp_nn, eps)   # (M, N)
    y = gate_elem_triton(x_n, proj, Wg_t)                             # (M, N)
    return y.view(B, L, L, N)

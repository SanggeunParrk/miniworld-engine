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

from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear import (
    layernorm_linear_cute,
)
from miniworld_engine.kernels.trimul_inproj.cute import dispatch
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import (
    gate_elem_infer, gate_elem_quack_fused,
)


def trimul_back_split(tri_bdll, x_n, Wp_nn, Wg_t, ln_w, ln_b, residual, eps=1e-5,
                      lnl_config=None):
    """tri_bdll:(B,K,L,L) with K=hidden (=D for square trimul, =2*d_hidden for bidir),
    x_n:(B,L,L,d_pair), Wp_nn:(d_pair,K)=to_out.weight (N,K), Wg_t:(d_pair,d_pair)=
    to_gate.weight.T -> y:(B,L,L,d_pair). B=1.

    ``residual`` (== the module input pair) is required: every trimul back half returns the
    residual form. The triton gate fuses it into the store; the quack-fused gate cannot take a
    residual in its CUTLASS epilogue, so that branch pays for a separate add -- which is part of
    what the dispatch below is choosing between.
    """
    B, K, L, L2 = tri_bdll.shape
    assert B == 1 and L == L2
    N = Wp_nn.shape[0]                                             # output width = d_pair
    M = L * L
    if isinstance(lnl_config, dict):
        from miniworld_engine.autotune.cute_config import (
            config_to_kwargs, kwargs_to_config, plain_sm90_candidates,
        )
        lnl_config = kwargs_to_config({**config_to_kwargs(plain_sm90_candidates()[0]), **lnl_config})
    # ① cute LayerNormLinear: M-major view of tri (channel strided by M), no copy.
    #    LN over K channels, then @Wp (K -> N). K may differ from N (bidirectional).
    view = tri_bdll.reshape(B, K, M)[0].t()                       # (M, K)
    proj = layernorm_linear_cute(view, ln_w, ln_b, Wp_nn, None, eps=eps,
                                 config=lnl_config)                 # (M, N), M1
    # ② gate: dispatch fused-quack (act(A@B)⊙C, one launch) vs triton (cuBLAS gemm + ew),
    #    cache the per-shape winner (fused wins large L; triton can win tiny L).
    res_flat = residual.reshape(M, N)
    y = dispatch.pick("gate_infer", dispatch._operand_key(x_n, proj, Wg_t, res_flat),
                      [("fused", lambda: gate_elem_quack_fused(x_n, proj, Wg_t) + res_flat),
                       ("triton", lambda: gate_elem_infer(x_n, proj, Wg_t, res_flat))])
    return y.view(B, L, L, N)

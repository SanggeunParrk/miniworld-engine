"""Fused BIDIRECTIONAL dt-v1 — a fair bidirectional baseline built from dt-v1's own kernels.

The stock `fused_triangle_multiplicative_update_dtv1` is single-direction. Running it TWICE
(outgoing + incoming) double-counts the shared work (2× LN_in, 2× input GEMM launch, 2 output
blocks). This composes ONE fused bidirectional block with the SAME architecture as ours
(`cute/bidir_training.py`) so the comparison is apples-to-apples:

    x_n   = LN_in(x)                                            # dt-v1 fused input kernel
    L,R   = sigmoid(x_n@g_in)·(x_n@p_in), each 2h wide          #   (one (4h,M) gated GEMM)
    o_out = contract(L[:h], R[:h], outgoing)                    # dt-v1 _TriangleContractBMM
    o_in  = contract(L[h:], R[h:], incoming)                    #   (×2)
    tri   = cat([o_out, o_in])                                  # (2h, B, I, J)
    out_n = LN_out(tri)                                         # dt-v1 layer_norm_transpose (2h)
    y     = sigmoid(x_n @ g_out) · (out_n @ p_out)              # split output gate (see below)

dt-v1's fused `_OutputGEMM` assumes the gate-input and proj-input share K. Here the output
gate reads the d_pair-wide x_n while the projection reads the 2h-wide contraction — different
K — so the output gate is split into two GEMMs. OURS has the exact same split (GateElem over
d_pair + te-LayerNormLinear over 2h), so neither side gets a fused-output advantage. The
dominant kernels (fused input LN+GEMM, contraction, LN_out) are dt-v1's, unchanged.

B=1, bf16. h = d_hidden per direction; back over 2h channels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import (
    _InputLNAndGEMM,
    _output_layer_norm_transpose,
    _triangle_contract_bmm_dbij,
)


def fused_bidirectional_dtv1(
    x,                # (B, L, L, d_pair)
    mask,
    norm_in_weight, norm_in_bias,
    p_in_weight,      # (4h, d_pair) = cat([to_left.weight, to_right.weight])
    g_in_weight,      # (4h, d_pair) = cat([to_left_gate.weight, to_right_gate.weight])
    norm_out_weight, norm_out_bias,   # (2h,)
    p_out_weight,     # (d_pair, 2h) = to_out.weight
    g_out_weight,     # (d_pair, d_pair) = to_gate.weight
    h,
    eps=1e-5,
):
    b, i, j, d = x.shape
    m = b * i * j
    H = 2 * h
    x_flat = x.reshape(m, d)
    mask_flat = mask.reshape(-1) if mask is not None else None

    # input: one fused LN + gated GEMM, output 2·(2h) = 4h wide → chunk into left/right (2h each)
    ab_t, _, _, x_normed = _InputLNAndGEMM.apply(
        x_flat, norm_in_weight, norm_in_bias, g_in_weight, p_in_weight, mask_flat, eps, True)
    left_t, right_t = torch.chunk(ab_t, 2, dim=0)            # each (2h, M)
    left = left_t.view(H, b, i, j)
    right = right_t.view(H, b, i, j)

    # split channels: [:h] outgoing, [h:] incoming → cat to (2h, B, I, J)
    o_out = _triangle_contract_bmm_dbij(left[:h], right[:h], "outgoing")
    o_in = _triangle_contract_bmm_dbij(left[h:], right[h:], "incoming")
    tri = torch.cat([o_out, o_in], dim=0)                   # (2h, B, I, J)

    out_n = _output_layer_norm_transpose(tri, norm_out_weight, norm_out_bias, b, i * j, H, eps)
    out_n = out_n.reshape(m, H)

    # output gate (split: gate over d_pair-wide x_n, projection over 2h-wide out_n)
    proj = F.linear(out_n, p_out_weight)                    # (m, d)
    gate = torch.sigmoid(F.linear(x_normed, g_out_weight))  # (m, d)
    return (gate * proj).reshape(b, i, j, d)

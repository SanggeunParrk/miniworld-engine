"""SM100 (B200) BIDIRECTIONAL trimul forward — CUEQUIV-FREE.

Builds the repo's fused bidirectional trimul (outgoing + incoming) out of the
CURRENT single-direction sm100 kernels — no cuequiv, no quack layer_norm_transpose:

    LN_in (triton)
      -> tm1 front (bdll_sm100, GatedPersistentGemmKernel tcgen05) x2
             outgoing half = to_*.weight[:h]   -> left_o, right_o  [B, h, L, L]
             incoming half = to_*.weight[h:]   -> left_i, right_i  [B, h, L, L]
      -> outgoing einsum (bdik,bdjk->bdij) + incoming einsum (bdki,bdkj->bdij)
      -> cat over the channel axis -> tri [B, 2h, L, L]
      -> back_split_sm100 (sm100 LayerNormLinear over 2h + triton gate_elem)

This is exactly the single-direction ``_forward_cute_free`` composition applied to
BOTH directions, with a SHARED back-half over the 2h concatenation (ln_out/to_out
operate over 2h -> d), matching ``BidirectionalTriangleMultiplication`` (pytorch
reference). Because d_hidden == d_pair here, each per-direction weight slice is
square (d, h) = (128, 128), which the ``bdll_sm100`` front accepts unchanged.

B=1, bf16 in / fp32 acc (fp32 LN stats) / bf16 out. Forward/inference only.
"""

from __future__ import annotations

import torch

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.trimul_inproj.cute.back_split_sm100 import (
    trimul_back_split_sm100,
)


@torch.no_grad()
def bidirectional_trimul_sm100(
    pair,
    to_left_w, to_left_gate_w, to_right_w, to_right_gate_w,   # (2h, d) nn.Linear form
    to_gate_w, to_out_w,                                       # (d,d), (d,2h) nn.Linear form
    ln_in_w, ln_in_b, ln_out_w, ln_out_b,
    eps_in, eps_out, h,
    tm1_cute_forward, out_layout,
):
    """pair:(B,L,L,d) -> y:(B,L,L,d). B=1. h = d_hidden per direction."""
    b, l1, l2, d = pair.shape
    x = triton_layernorm(
        pair.reshape(b * l1 * l2, d), ln_in_w, ln_in_b, eps_in
    ).view(b, l1, l2, d)

    def _front(sl):
        # tm1 bdll_sm100 wants W:(in, out) = weight.T; per-direction slice is (h, d)->T (d, h).
        return tm1_cute_forward(
            x,
            to_left_w[sl].T, to_left_gate_w[sl].T,
            to_right_w[sl].T, to_right_gate_w[sl].T,
            out_layout=out_layout,
        )

    left_o, right_o = _front(slice(0, h))            # outgoing half [B, h, L, L]
    left_i, right_i = _front(slice(h, 2 * h))        # incoming half [B, h, L, L]
    out_o = torch.einsum("bdik,bdjk->bdij", left_o, right_o)   # outgoing
    out_i = torch.einsum("bdki,bdkj->bdij", left_i, right_i)   # incoming
    tri = torch.cat([out_o, out_i], dim=1)           # [B, 2h, L, L]

    # shared back-half over 2h: ln_out(2h) @ to_out(2h->d), gated by sigmoid(x @ to_gate)
    return trimul_back_split_sm100(
        tri, x, to_out_w, to_gate_w.T, ln_out_w, ln_out_b, eps_out,
    )

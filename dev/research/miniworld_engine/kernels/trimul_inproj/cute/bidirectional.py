"""Bidirectional trimul forward — OURS (our trimul_inproj fusion, bidirectional dims).

Same pipeline as the single-direction inference path, only the dims differ (exactly
as planned): the front gated GEMM emits left/right each ``2*d_hidden`` wide, the bmm
becomes TWO einsums (outgoing on the first ``d_hidden`` channels, incoming on the
second), and the split back's LayerNormLinear runs over ``2*d_hidden`` channels.

    LN_in(triton) -> trimul_inproj front (one gated GEMM, left/right [B,2h,L,L], bdll)
      -> outgoing einsum on [:h] + incoming einsum on [h:] -> cat [B,2h,L,L]
      -> split back: ① cute LayerNormLinear (K=2h -> N=d_pair)  ② triton GateElem (d_pair)

Weights (x@W form unless noted):
  WL,WLg,WR,WRg : (d_pair, 2h)  = to_{left,right}[_gate].weight.T
  Wg            : (d_pair, d_pair) = to_gate.weight.T
  Wp_nn         : (d_pair, 2h)  = to_out.weight  (nn.Linear (N,K) form — cute LNL wants this)
  b_lr          : (d_pair, 4*2h) prepacked  (prepack_lr_operand on the four 2h-wide weights)

B=1, bf16. Forward/inference only (no autograd attached).
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from miniworld_engine.kernels.trimul_inproj.cute.back_split import trimul_back_split
from miniworld_engine.kernels.trimul_inproj.cute.launch import trimul_inproj_cute_forward


@torch.compiler.disable()
@torch.no_grad()
def bidirectional_trimul_ours(pair, WL, WLg, WR, WRg, Wg, Wp_nn,
                              ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps, b_lr, h):
    """pair:(B,L,L,d_pair) -> y:(B,L,L,d_pair). h = d_hidden (per direction). B=1."""
    B, L, _, d = pair.shape
    xf = pair.reshape(B * L * L, d)
    xn = triton_layernorm(xf, ln_in_w, ln_in_b, eps).view(B, L, L, d)

    # front: left/right each 2h wide, bdll [B, 2h, L, L]
    left, right, _ = trimul_inproj_cute_forward(
        xn, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False,
        b_lr=b_lr, out_hidden=2 * h)

    # split channels: [:h] -> outgoing, [h:] -> incoming (matches the pytorch ref)
    lo, li = left[:, :h], left[:, h:]
    ro, ri = right[:, :h], right[:, h:]
    out_out = torch.einsum("bdik,bdjk->bdij", lo, ro)   # outgoing  (bikd,bjkd->bijd)
    out_in = torch.einsum("bdki,bdkj->bdij", li, ri)    # incoming  (bkid,bkjd->bijd)
    out = torch.cat([out_out, out_in], dim=1)           # [B, 2h, L, L]

    # split back: ① cute LayerNormLinear (LN over 2h, @Wp 2h->d_pair)  ② triton GateElem
    return trimul_back_split(out, xn, Wp_nn, Wg, ln_out_w, ln_out_b, eps)

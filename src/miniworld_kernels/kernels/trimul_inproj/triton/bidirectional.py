"""Bidirectional trimul in TRITON — outgoing + incoming in one call (fwd + bwd).

This is the TRITON counterpart of the cute bidirectional path
(``cute/bidirectional.py`` / ``cute/bidir_training.py``). It does NOT introduce a
new monolithic kernel: it *composes* the already-verified unidirectional triton
autograd pieces per direction and combines them — the same strategy the H100
tri-attention bidir path used (reuse the single-direction kernels rather than
rewrite). Every stage below is autograd-capable, so the training (fwd+bwd) path
is obtained for free — no hand-written backward to drift out of sync with fwd.

Math (matches ``modules/triangle_multiplication/bidirectional.py`` PYTORCH ref):

    x_n   = LayerNorm_in(pair)                                   # over d_pair
    # front, per direction (each a SQUARE d_pair->d_pair gated GEMM = triton_tm1):
    L_o, R_o = triton_tm1(x_n, WL[:h], WLg[:h], WR[:h], WRg[:h]) # outgoing half
    L_i, R_i = triton_tm1(x_n, WL[h:], WLg[h:], WR[h:], WRg[h:]) # incoming half
    # (optional) pair-mask multiply on left & right, both directions
    o_out = einsum("bikd,bjkd->bijd", L_o, R_o)                  # outgoing contraction
    o_in  = einsum("bkid,bkjd->bijd", L_i, R_i)                  # incoming contraction
    out   = cat([o_out, o_in], dim=-1)                           # (B,L,L,2h)
    out_n = LayerNorm_out(out)                                   # over 2h
    proj  = out_n @ Wout.T                                       # (2h -> d_pair)
    y     = sigmoid(x_n @ Wg.T) * proj                           # triton GateElem

The two front calls reuse ``triton_tm1`` (folder ``kernels/tm1``), which requires
SQUARE (d,d) weights — hence the per-direction ``h``-wide slices, valid only when
``d_hidden == d_pair`` (enforced by the caller, as the unidirectional triton path
does). The gate is the triton ``GateElem`` autograd Function from
``trimul_inproj/triton/gate_elem.py``. LayerNorms use ``F.layer_norm`` so their
weight/bias gradients flow. B may be >1 (carried through the leading dims); the
underlying kernels flatten to (M, D). bf16 and fp32 both supported.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.tm1.triton.main import triton_tm1
from miniworld_kernels.kernels.trimul_inproj.triton.gate_elem import GateElem


def bidirectional_trimul_triton(
    pair: torch.Tensor,          # (B, L, L, d_pair)
    WL: torch.Tensor,            # to_left.weight       (2h, d_pair)
    WLg: torch.Tensor,           # to_left_gate.weight  (2h, d_pair)
    WR: torch.Tensor,            # to_right.weight      (2h, d_pair)
    WRg: torch.Tensor,           # to_right_gate.weight (2h, d_pair)
    Wg: torch.Tensor,            # to_gate.weight       (d_pair, d_pair)
    Wout: torch.Tensor,          # to_out.weight        (d_pair, 2h)
    ln_in_w: torch.Tensor,       # (d_pair,)
    ln_in_b: torch.Tensor,       # (d_pair,)
    ln_out_w: torch.Tensor,      # (2h,)
    ln_out_b: torch.Tensor,      # (2h,)
    eps_in: float,
    eps_out: float,
    d_hidden: int,
    mask: torch.Tensor | None = None,   # (B, L) residue mask, optional
) -> torch.Tensor:
    """Composed bidirectional trimul (fwd + autograd bwd). Returns (B, L, L, d_pair)."""
    b, l1, l2, d = pair.shape
    h = d_hidden
    if h != d:
        raise ValueError(
            f"TRITON bidirectional trimul requires d_hidden == d_pair "
            f"(got d_hidden={h}, d_pair={d}); the reused triton_tm1 front needs "
            f"square (d,d) projections."
        )

    # ── LN_in over d_pair ────────────────────────────────────────────────
    x_n = F.layer_norm(pair, (d,), ln_in_w, ln_in_b, eps_in)   # (B,L,L,d)

    # ── front, per direction (square h==d slices, weights in (out,in) form) ──
    # triton_tm1 wants x@W form: weight.T = (d, h). Slices [:h]=outgoing, [h:]=incoming.
    def _front(sl: slice):
        return triton_tm1(
            x_n,
            WL[sl].T.contiguous(),
            WLg[sl].T.contiguous(),
            WR[sl].T.contiguous(),
            WRg[sl].T.contiguous(),
        )

    left_o, right_o = _front(slice(0, h))     # (B,L,L,h) outgoing
    left_i, right_i = _front(slice(h, 2 * h))  # (B,L,L,h) incoming

    if mask is not None:
        mask_2d = (mask.unsqueeze(-1) & mask.unsqueeze(-2))[..., None]  # (B,L,L,1)
        m = mask_2d.to(x_n.dtype)
        left_o = left_o * m
        right_o = right_o * m
        left_i = left_i * m
        right_i = right_i * m

    # ── two contractions (outgoing / incoming), then concat to 2h ─────────
    out_o = torch.einsum("bikd,bjkd->bijd", left_o, right_o)   # (B,L,L,h)
    out_i = torch.einsum("bkid,bkjd->bijd", left_i, right_i)   # (B,L,L,h)
    out = torch.cat([out_o, out_i], dim=-1)                     # (B,L,L,2h)

    # ── LN_out over 2h, proj down to d_pair, triton gate ─────────────────
    out_n = F.layer_norm(out, (2 * h,), ln_out_w, ln_out_b, eps_out)  # (B,L,L,2h)
    proj = out_n.reshape(b * l1 * l2, 2 * h) @ Wout.T                 # (M, d_pair)
    x_n_flat = x_n.reshape(b * l1 * l2, d)
    y = GateElem.apply(x_n_flat, proj, Wg.T.contiguous())            # (M, d_pair)
    return y.view(b, l1, l2, d)

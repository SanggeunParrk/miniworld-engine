"""PyTorch pieces of the DiT block variants (Boltz-2, RoseTTAFold3, Chai-1 token DiT) the portable Triton path reuses: the AdaLN,
the transition delta and the atom-window geometry. Every product in the parameters' dtype, norms in fp32.

    AdaLN(x, s)   sigmoid form:  sigmoid(W_sc LN_w(s) + b_sc) * LN(x) + W_sh LN_w(s)     (LN_w: weight-only, eps 1e-5)
                  identity form: (W_sc s + 1) * LN(x) + W_sh s                            (Chai-1)
    transition    t = AdaLN_T(x', s); [g | v] = W_1 t; h = silu(g) * v [* W_u t (up-gate, Boltz-2)];
                  dT = sigmoid(W_zT s + b_zT) * W_2 h
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ATOM_HEADS, ATOM_QUERIES, ATOM_KEYS = 4, 32, 128


def adaln(norm, x: torch.Tensor, s: torch.Tensor, eps: float) -> torch.Tensor:
    """``norm``: a DiT-variant AdaLN (``identity_scale``, ``to_scale``, ``to_shift``, ``cond_norm_weight``)."""
    xn = F.layer_norm(x.float(), x.shape[-1:], eps=eps).to(x.dtype)
    if norm.identity_scale:
        return (norm.to_scale(s) + 1) * xn + norm.to_shift(s)
    sn = F.layer_norm(s.float(), s.shape[-1:], weight=norm.cond_norm_weight.float(), eps=1e-5).to(s.dtype)
    return torch.sigmoid(norm.to_scale(sn)) * xn + norm.to_shift(sn)


def transition_delta(block, x, s):
    t = adaln(block.trans_adaln, x, s, block.adaln_eps)
    g, v = block.expand(t).chunk(2, dim=-1)
    h = F.silu(g) * v
    if block.up_gate:
        h = h * block.up(t)
    return torch.sigmoid(block.trans_zero(s)) * block.squeeze(h)

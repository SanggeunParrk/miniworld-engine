"""PyTorch reference of the DiT block variants (Boltz-2, RoseTTAFold3, Chai-1 token DiT), the math FoldForge's dense
``DiffusionTransformer`` runs for them. Every product in the parameters' dtype, norms and softmax in fp32.

One block, with ``x`` [A, L, D] the token rows of the A samples, ``s`` [L, c_s] or [A, L, c_s] the single conditioning,
``pair`` [L, L, C] the pair conditioning and ``mask`` [L] (bool, True = real token) or None:

    AdaLN(x, s)   sigmoid form:  sigmoid(W_sc LN_w(s) + b_sc) * LN(x) + W_sh LN_w(s)     (LN_w: weight-only, eps 1e-5)
                  identity form: (W_sc s + 1) * LN(x) + W_sh s                            (Chai-1)
    attention     a = AdaLN_A(x, s); q = W_q a + b_q, k = W_k a, v = W_v a [; q, k = LN(q), LN(k) (kq-norm, RF3)]
                  logits = q k^T / sqrt(d) + W_p LN_p(pair) (per head), keys off the mask -> -1e9; o = softmax v
                  [o = sigmoid(W_g a) * o (gating query; not Chai-1)]; dA = sigmoid(W_zA s + b_zA) * W_o o
    transition    t = AdaLN_T(x', s); [g | v] = W_1 t; h = silu(g) * v [* W_u t (up-gate, Boltz-2)];
                  dT = sigmoid(W_zT s + b_zT) * W_2 h
    schedule      sequential: x' = x + dA, out = x' + dT;  parallel (RF3, Chai-1): x' = x, out = x + dA + dT
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def adaln(norm, x: torch.Tensor, s: torch.Tensor, eps: float) -> torch.Tensor:
    """``norm``: a ``modules.dit_variant.VariantAdaLN``."""
    xn = F.layer_norm(x.float(), x.shape[-1:], eps=eps).to(x.dtype)
    if norm.identity_scale:
        return (norm.to_scale(s) + 1) * xn + norm.to_shift(s)
    sn = F.layer_norm(s.float(), s.shape[-1:], weight=norm.cond_norm_weight.float(), eps=1e-5).to(s.dtype)
    return torch.sigmoid(norm.to_scale(sn)) * xn + norm.to_shift(sn)


def pair_bias(block, pair: torch.Tensor) -> torch.Tensor:
    """[H, L, L] (natural units) from the pair [L, L, C]."""
    norm = block.pair_norm
    pn = F.layer_norm(pair.float(), pair.shape[-1:], weight=norm.weight.float(),
                      bias=None if norm.bias is None else norm.bias.float(), eps=norm.eps).to(pair.dtype)
    return block.to_pair_bias(pn).permute(2, 0, 1)


def attention_delta(block, x, s, z, mask):
    a = adaln(block.attn_adaln, x, s, block.adaln_eps)
    q, k, v = block.to_q(a), block.to_k(a), block.to_v(a)
    if block.kq_norm:
        q, k = (F.layer_norm(t.float(), t.shape[-1:], n.weight.float(), n.bias.float(), n.eps).to(t.dtype)
                for t, n in ((q, block.q_norm), (k, block.k_norm)))
    A, L, D = x.shape
    H = block.n_head
    d = D // H
    q, k, v = (t.view(A, L, H, d).transpose(1, 2).float() for t in (q, k, v))
    logits = q @ k.transpose(-1, -2) / d**0.5 + z.float()
    if mask is not None:
        logits = logits.masked_fill(~mask.bool()[None, None, None, :], -1e9)
    o = (logits.softmax(-1) @ v).transpose(1, 2).reshape(A, L, D).to(x.dtype)
    if block.gating:
        o = torch.sigmoid(block.to_g(a)) * o
    return torch.sigmoid(block.attn_zero(s)) * block.to_out(o)


def transition_delta(block, x, s):
    t = adaln(block.trans_adaln, x, s, block.adaln_eps)
    g, v = block.expand(t).chunk(2, dim=-1)
    h = F.silu(g) * v
    if block.up_gate:
        h = h * block.up(t)
    return torch.sigmoid(block.trans_zero(s)) * block.squeeze(h)


def block_forward(block, x, s, pair, mask, parallel: bool, z=None):
    z = pair_bias(block, pair) if z is None else z
    da = attention_delta(block, x, s, z, mask)
    if parallel:
        return x + da + transition_delta(block, x, s)
    x = x + da
    return x + transition_delta(block, x, s)


def stack_forward(stack, x, s, pair, mask):
    for block in stack.blocks:
        x = block_forward(block, x, s, pair, mask, stack.parallel)
    return x

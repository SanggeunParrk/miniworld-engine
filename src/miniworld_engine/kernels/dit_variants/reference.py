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


# ------------------------------------------------------------------------------------------------------------------------- atom
# The atom DiT (32 queries x 128 keys per window, 4 heads x 32): ``x`` [A, N, 128], ``c`` [N, 128] (or [A, N, 128]) the per-atom
# conditioning, ``z`` [nwin, 32, 128, d_pair] the trunked pair (key slot j of window w is laid out from the window's first key, as
# ``modules.local_dit.to_windows`` makes it), ``mask`` [N] (real atoms first) or None, ``uid`` [N] (Chai-1: a query sees only the keys of
# its own conformer) or None.
#
#   attention     xq = AdaLN_q(x, c), xk = AdaLN_k(x, c) (both of the block input: AlphaFold 3's parallel K / V norm);
#                 q = W_q xq + b_q, k = W_k xk, v = W_v xk [; q, k = LN(q), LN(k)]; logits = q k^T / sqrt(32) + bias[h, w, i, j];
#                 o = softmax v [; o = sigmoid(W_g xq) o] [; o = W_o o (Chai-1 has no output projection)]; dA = sigmoid(W_zA c + b) o
#   pair bias     per block LN_b + Linear_b(d_pair -> 4) (RF3), or one LN + Linear(d_pair -> 4 nb) for the stack (Boltz-2, Chai-1),
#                 output channel 4 b + h
#   key window    "slide": [32 w - 48, 32 w + 80) shifted back inside the real atoms (AlphaFold 3, FoldForge's fast mode);
#                 "pad": the same window clipped to [0, N), keys outside it and masked keys off

ATOM_HEADS, ATOM_QUERIES, ATOM_KEYS = 4, 32, 128


def atom_window_keys(n: int, mask, key_window: str, device):
    """(index [nwin, 128] of every window's keys, valid [nwin, 128])."""
    from miniworld_engine.modules.local_dit.module import slide_starts

    nwin = n // ATOM_QUERIES
    j = torch.arange(ATOM_KEYS, device=device)
    if key_window == "slide":
        n_real = (mask.sum() if mask is not None else torch.tensor(n, device=device)).reshape(1)
        idx = slide_starts(n_real, nwin)[0][:, None] + j
        valid = idx < n_real
    else:
        idx = (ATOM_QUERIES * torch.arange(nwin, device=device) - (ATOM_KEYS - ATOM_QUERIES) // 2)[:, None] + j
        valid = (idx >= 0) & (idx < n)
        if mask is not None:
            valid = valid & mask.bool()[idx.clamp(0, n - 1)]
    return idx.clamp(0, n - 1), valid


def atom_pair_bias(stack, i: int, z):
    """[4, nwin, 32, 128] natural-unit bias of block ``i``."""
    if stack.per_block_pair_norm:
        norm, proj, rows = stack.blocks[i].pair_norm, stack.blocks[i].to_pair_bias.weight, slice(None)
    else:
        norm, proj, rows = stack.pair_norm, stack.to_pair_bias.weight, slice(ATOM_HEADS * i, ATOM_HEADS * (i + 1))
    zn = F.layer_norm(z.float(), z.shape[-1:], norm.weight.float(), None if norm.bias is None else norm.bias.float(), norm.eps)
    return F.linear(zn.to(z.dtype), proj[rows]).permute(3, 0, 1, 2)


def atom_block_forward(stack, i, x, c, z, mask, uid):
    blk = stack.blocks[i]
    A, n, D = x.shape
    nwin, H, d = n // ATOM_QUERIES, ATOM_HEADS, D // ATOM_HEADS
    xq, xk = adaln(blk.q_adaln, x, c, blk.adaln_eps), adaln(blk.k_adaln, x, c, blk.adaln_eps)
    q, k, v = blk.to_q(xq), blk.to_k(xk), blk.to_v(xk)
    if blk.kq_norm:
        q, k = (F.layer_norm(t.float(), t.shape[-1:], m.weight.float(), m.bias.float(), m.eps).to(t.dtype)
                for t, m in ((q, blk.q_norm), (k, blk.k_norm)))
    idx, valid = atom_window_keys(n, mask, stack.key_window, x.device)
    keep = valid[:, None, :].expand(nwin, ATOM_QUERIES, ATOM_KEYS)
    if uid is not None:
        qu = uid.view(nwin, ATOM_QUERIES, 1)
        real = torch.ones_like(qu, dtype=torch.bool) if mask is None else mask.bool().view(nwin, ATOM_QUERIES, 1)
        keep = keep & ((qu == uid[idx][:, None, :]) | ~real)
    qw = q.view(A, nwin, ATOM_QUERIES, H, d).float()
    kw, vw = (t.view(A, n, H, d)[:, idx].float() for t in (k, v))                      # [A, nwin, 128, H, d]
    logits = torch.einsum("awqhd,awkhd->awhqk", qw, kw) * d**-0.5 + atom_pair_bias(stack, i, z).float().permute(1, 0, 2, 3)[None]
    logits = logits.masked_fill(~keep[None, :, None], float("-inf"))
    probs = torch.nan_to_num(logits.softmax(-1))
    o = torch.einsum("awhqk,awkhd->awqhd", probs, vw).reshape(A, n, D).to(x.dtype)
    if blk.gating:
        o = torch.sigmoid(blk.to_g(xq)) * o
    if blk.project_output:
        o = blk.to_out(o)
    da = torch.sigmoid(blk.attn_zero(c)) * o
    if stack.parallel:
        return x + da + transition_delta(blk, x, c)
    x = x + da
    return x + transition_delta(blk, x, c)


def atom_stack_forward(stack, x, c, z, mask=None, uid=None):
    for i in range(len(stack.blocks)):
        x = atom_block_forward(stack, i, x, c, z, mask, uid)
    return x

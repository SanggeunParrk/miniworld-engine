"""LocalDiTBlock's PyTorch path is the AF3 block-local attention: atom i of query window w = i // 32 attends to the atoms
[32 w - 48, 32 w + 80) clipped to [0, N), with the trunked pair as its bias. Checked against a literal per-atom loop, on CPU."""

import torch

from miniworld_engine.modules.local_dit import (
    KEY_OFFSET,
    KEYS,
    QUERIES,
    LocalDiTBlock,
    to_windows,
)


def _block(seed=0, cross=False):
    torch.manual_seed(seed)
    block = LocalDiTBlock(cross_attention=cross).double()
    with torch.no_grad():
        for name, p in block.named_parameters():
            if p.ndim > 1:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if name.endswith("weight") else 0.0))
    return block


def test_trunked_pair_is_the_dense_pair_seen_through_the_windows():
    n = 100
    dense = torch.randn(2, n, n, 16, dtype=torch.double)
    z = to_windows(dense)
    assert z.shape == (2, 4, QUERIES, KEYS, 16)
    for w in range(4):
        for i in range(QUERIES):
            for j in range(KEYS):
                q, k = QUERIES * w + i, QUERIES * w - KEY_OFFSET + j
                expected = dense[:, q, k] if q < n and 0 <= k < n else torch.zeros(2, 16, dtype=torch.double)
                torch.testing.assert_close(z[:, w, i, j], expected, atol=0, rtol=0)


def _literal(block, single, cond, dense, mask):
    """The attention update atom by atom: softmax over the keys of the atom's window only."""
    att = block.attention
    x = att.ada_ln_in(single, cond)
    n = x.shape[2]
    xkv = att.ada_ln_kv(x, cond) if block.cross_attention else x
    q, k, v, g = att.to_query(x), att.to_key(xkv), att.to_value(xkv), att.to_gate(x)
    bias = torch.nn.functional.linear(
        torch.nn.functional.layer_norm(dense, (16,), att.ln_pair.weight.to(dense.dtype), None, att.ln_pair.eps), att.to_bias.weight.to(dense.dtype))   # [B, N, N, H]
    out = torch.zeros_like(q)
    for i in range(n):
        w = i // QUERIES
        keys = [j for j in range(max(0, QUERIES * w - KEY_OFFSET), min(n, QUERIES * w + KEYS - KEY_OFFSET)) if mask is None or mask[0, j]]
        for h in range(4):
            sl = slice(32 * h, 32 * h + 32)
            s = torch.stack([(q[:, 0, i, sl] * k[:, 0, j, sl]).sum(-1) / 32**0.5 + bias[0, i, j, h] for j in keys], dim=-1)
            p = torch.softmax(s, dim=-1)
            out[:, 0, i, sl] = sum(p[:, t, None] * v[:, 0, j, sl] for t, j in enumerate(keys))
    return torch.sigmoid(att.to_scale(cond)) * att.to_out(torch.sigmoid(g) * out)


def test_reference_attention_is_the_per_atom_window_softmax():
    for n, masked, cross in ((70, False, False), (100, True, False), (70, False, True), (100, True, True)):
        block = _block(n, cross)
        single, cond = torch.randn(2, 1, n, 128, dtype=torch.double), torch.randn(2, 1, n, 128, dtype=torch.double)
        dense = torch.randn(1, n, n, 16, dtype=torch.double)
        mask = (torch.rand(1, n) > 0.2) if masked else None
        got = block.attention_delta(single, cond, to_windows(dense), mask)
        torch.testing.assert_close(got, _literal(block, single, cond, dense, mask), atol=1e-10, rtol=1e-8)


def test_forward_is_both_residuals():
    n = 64
    block = _block(3)
    single, cond = torch.randn(1, 1, n, 128, dtype=torch.double), torch.randn(1, 1, n, 128, dtype=torch.double)
    z = to_windows(torch.randn(1, n, n, 16, dtype=torch.double))
    got = block(single, cond, z)
    mid = single + block.attention_delta(single, cond, z)
    torch.testing.assert_close(got, block.transition(mid, cond), atol=1e-12, rtol=0)

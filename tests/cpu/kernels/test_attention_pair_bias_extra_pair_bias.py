"""``AttentionPairBias(extra_pair_bias=)`` / ``PairformerBlock(extra_pair_bias=)``: an additive term on the pair-bias logits (the structural-token
refiner's extra bias). CPU, the PyTorch statements, float32, against an explicit softmax-attention written from the module's own projections.
"""

import pytest
import torch

from miniworld_engine.modules.attention_pair_bias import AttentionPairBias
from miniworld_engine.modules.pairformer import PairformerBlock, PairformerConfig

B, L, D_SINGLE, D_PAIR, HEADS = 2, 12, 32, 16, 4


def _module(gate_bias=0.0, seed=0):
    torch.manual_seed(seed)
    m = AttentionPairBias(D_SINGLE, D_PAIR, HEADS, gate_bias=gate_bias)
    for p in m.parameters():  # to_out starts at zero: a zero update would hide any difference
        p.data.normal_(0.0, 0.4)
    return m


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    single = torch.randn(B, L, D_SINGLE, generator=g)
    pair = torch.randn(B, L, L, D_PAIR, generator=g)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, -3:] = False
    return single, pair, mask


def _reference(m, single, pair, mask, extra):
    """softmax(q k^T / sqrt(d) + to_bias(ln_pair(pair)) + extra) v, key-masked, sigmoid-gated, projected, plus the residual."""
    x = m.ln_single(single)
    q, k, v = (t(x).unflatten(-1, (HEADS, -1)).transpose(1, 2) for t in (m.to_query, m.to_key, m.to_value))
    bias = m.to_bias(m.ln_pair(pair)).permute(0, 3, 1, 2)
    if extra is not None:
        bias = bias + extra
    if mask is not None:
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
    att = ((q * q.shape[-1] ** -0.5) @ k.transpose(-1, -2) + bias).softmax(-1) @ v
    att = att.transpose(1, 2).flatten(-2)
    return single + m.to_out(torch.sigmoid(m.to_gate(x) + m.gate_bias) * att)


@pytest.mark.parametrize("gate_bias", [0.0, 1.0])
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("heads", [HEADS, 1], ids=["per_head", "shared_head"])
def test_extra_pair_bias_matches_reference(heads, masked, gate_bias):
    m = _module(gate_bias)
    single, pair, mask = _inputs()
    mask = mask if masked else None
    extra = 2 * torch.randn(B, heads, L, L, generator=torch.Generator().manual_seed(3))
    got = m(single, pair, mask, extra)
    torch.testing.assert_close(got, _reference(m, single, pair, mask, extra), atol=1e-5, rtol=1e-5)


def test_extra_pair_bias_changes_the_output_and_none_is_unchanged():
    m = _module()
    single, pair, mask = _inputs()
    extra = 2 * torch.randn(B, HEADS, L, L, generator=torch.Generator().manual_seed(3))
    plain = m(single, pair, mask)
    assert not torch.allclose(m(single, pair, mask, extra), plain, atol=1e-3)
    torch.testing.assert_close(m(single, pair, mask, torch.zeros_like(extra)), plain, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(plain, _reference(m, single, pair, mask, None), atol=1e-5, rtol=1e-5)


def test_extra_pair_bias_gradient():
    m = _module()
    single, pair, mask = _inputs()
    base = 2 * torch.randn(B, HEADS, L, L, generator=torch.Generator().manual_seed(3))
    grads = []
    for run in (lambda *a: m(*a), lambda *a: _reference(m, *a)):
        m.zero_grad()
        extra = base.clone().requires_grad_(True)
        s = single.clone().requires_grad_(True)
        y = run(s, pair, mask, extra)
        (y[:, :-3].square().sum()).backward()
        grads.append((extra.grad, s.grad, m.to_bias.weight.grad.clone()))
    assert grads[0][0].abs().sum() > 0
    for got, want in zip(*grads, strict=True):
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
def test_pairformer_block_forwards_extra_pair_bias(parallel):
    torch.manual_seed(2)
    cfg = PairformerConfig(d_pair=D_PAIR, d_hidden_tri_multi=D_PAIR, d_hidden_tri_attention=D_PAIR // 2, n_head_tri_attention=2, p_drop=0.0,
                           d_single=D_SINGLE, n_head_single=HEADS, parallel=parallel)
    block = PairformerBlock(cfg)
    for p in block.parameters():
        p.data.normal_(0.0, 0.2)
    single, pair, mask = _inputs()
    extra = 2 * torch.randn(B, HEADS, L, L, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        pair_a, single_a = block(pair, mask, single)
        pair_b, single_b = block(pair, mask, single, extra_pair_bias=extra)
        # the pair track does not read it; the single track does, through its attention only
        torch.testing.assert_close(pair_b, pair_a, atol=0, rtol=0)
        assert not torch.allclose(single_b[:, :-3], single_a[:, :-3], atol=1e-3)
        logits_pair = pair if parallel else pair_b  # the single attention reads the pair entering (parallel) / leaving (sequential) the block
        att = block.single_attention(single, logits_pair, mask, extra)
        expect = single + (att - single) + (block.single_transition(single) - single) if parallel else block.single_transition(att)
        torch.testing.assert_close(single_b, expect, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="single track"):
        PairformerBlock(PairformerConfig(d_pair=D_PAIR, d_hidden_tri_multi=D_PAIR, d_hidden_tri_attention=D_PAIR // 2,
                                         n_head_tri_attention=2, p_drop=0.0))(pair, mask, extra_pair_bias=extra)

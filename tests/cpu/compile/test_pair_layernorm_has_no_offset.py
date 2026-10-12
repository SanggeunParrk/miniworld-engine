"""The pair LayerNorm of MSAPairWeightedAveraging and AttentionPairBias has no offset.

It only feeds the softmax logits (to_bias has no bias), and an offset beta adds the same Wb_h . beta to every key of head h, which a softmax
over keys cancels exactly. CPU: the PyTorch statements, float32.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniworld_engine.modules.attention_pair_bias.module import AttentionPairBias
from miniworld_engine.modules.msa_pair_weighted_averaging.module import MSAPairWeightedAveraging

D_PAIR, D_MSA, D_SINGLE, HEADS, L = 24, 16, 16, 2, 10


def _modules():
    torch.manual_seed(0)
    pwa = MSAPairWeightedAveraging(D_MSA, D_PAIR, n_head=HEADS, d_hidden=8)
    apb = AttentionPairBias(D_SINGLE, D_PAIR, n_head=HEADS)
    for m in (pwa, apb):
        for p in m.parameters():                       # to_out starts at zero: a zero update would hide any difference
            p.data.normal_(0.0, 0.5)
        m.eval()
    return pwa, apb


class _OffsetLayerNorm(nn.Module):
    """The pair LayerNorm as it was: with an offset."""

    def __init__(self, ln, beta):
        super().__init__()
        self.weight, self.eps, self.beta = ln.weight, ln.eps, beta

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.beta, self.eps)


@pytest.mark.parametrize("name", ["pwa", "apb"])
def test_there_is_no_offset_parameter(name):
    module = dict(zip(("pwa", "apb"), _modules(), strict=True))[name]
    assert module.ln_pair.bias is None
    assert "ln_pair.bias" not in module.state_dict()
    assert module.ln_pair.weight.shape == (D_PAIR,)


@pytest.mark.parametrize("name", ["pwa", "apb"])
def test_a_checkpoint_with_the_offset_still_loads_strictly(name):
    module = dict(zip(("pwa", "apb"), _modules(), strict=True))[name]
    state = module.state_dict()
    state["ln_pair.bias"] = torch.randn(D_PAIR)
    module.load_state_dict(state)                                        # strict: the old key is dropped, nothing is left over
    assert "ln_pair.bias" not in module.state_dict()
    parent = nn.ModuleDict({"block": module})                            # and under a prefix
    prefixed = {f"block.{k}": v for k, v in state.items()}
    parent.load_state_dict(prefixed)


@pytest.mark.parametrize("masked", [False, True])
def test_pwa_output_is_independent_of_any_offset(masked):
    pwa, _ = _modules()
    msa = torch.randn(1, 5, L, D_MSA)
    pair = torch.randn(1, L, L, D_PAIR)
    mask = None if not masked else (torch.arange(L) % 3 != 0)[None]
    expected = pwa(msa, pair, mask)
    assert (expected - msa).abs().max() > 1e-3                          # the update is not zero
    ln = pwa.ln_pair
    pwa.ln_pair = _OffsetLayerNorm(ln, 3.0 * torch.randn(D_PAIR))
    torch.testing.assert_close(pwa(msa, pair, mask), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("masked", [False, True])
def test_apb_output_is_independent_of_any_offset(masked):
    _, apb = _modules()
    single = torch.randn(1, L, D_SINGLE)
    pair = torch.randn(1, L, L, D_PAIR)
    mask = None if not masked else (torch.arange(L) % 3 != 0)[None]
    expected = apb(single, pair, mask)
    assert (expected - single).abs().max() > 1e-3
    ln = apb.ln_pair
    apb.ln_pair = _OffsetLayerNorm(ln, 3.0 * torch.randn(D_PAIR))
    torch.testing.assert_close(apb(single, pair, mask), expected, rtol=1e-4, atol=1e-4)

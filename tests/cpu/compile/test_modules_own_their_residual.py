"""Every ``x = x + f(x)`` module returns ``x + f(x)``: a block chains them and never adds the residual itself.

At initialization each of these modules has a zero output projection, so its update is exactly zero: the output must be
the input bit for bit, and the input's gradient exactly one (two would mean the residual is added twice, which is what a
caller still writing ``x = x + module(x)`` gets). With random weights, ``forward`` is ``x + delta`` where a module exposes
its update.
"""

import pytest
import torch

from miniworld_engine.modules import (
    AugmentedAttentionPairBias,
    BidirectionalTriangleAttention,
    ConditionedTransition,
    TrianglePairAttention,
)
from miniworld_engine.modules.bias_only_dit import BiasOnlyAttention, BiasOnlyDiTBlock
from miniworld_engine.modules.dit import DiTBlock


def _token(a=2, length=6, d_single=32, d_cond=16, d_pair=8):
    single = torch.randn(a, 1, length, d_single, requires_grad=True)
    cond = torch.randn(a, 1, length, d_cond)
    pair = torch.randn(1, length, length, d_pair)
    return single, cond, pair


def _pair(length=5, d_pair=16):
    return torch.randn(1, length, length, d_pair, requires_grad=True)


CASES = {
    "AugmentedAttentionPairBias": lambda: (
        lambda m, s, c, p: m(s, c, p), AugmentedAttentionPairBias(32, 16, 8, 4), _token),
    "ConditionedTransition": lambda: (lambda m, s, c, p: m(s, c), ConditionedTransition(32, 16, 2), _token),
    "DiTBlock": lambda: (lambda m, s, c, p: m(s, c, p), DiTBlock(32, 16, 8, 4), _token),
    "BiasOnlyAttention": lambda: (lambda m, s, c, p: m(s, c, p), BiasOnlyAttention(32, 16, 8, 4), _token),
    "BiasOnlyDiTBlock": lambda: (lambda m, s, c, p: m(s, c, p), BiasOnlyDiTBlock(32, 16, 8, 4), _token),
    "TrianglePairAttention(starting)": lambda: (lambda m, p: m(p), TrianglePairAttention(16, 4, starting=True), _pair),
    "TrianglePairAttention(ending)": lambda: (lambda m, p: m(p), TrianglePairAttention(16, 4, starting=False), _pair),
    "BidirectionalTriangleAttention": lambda: (lambda m, p: m(p), BidirectionalTriangleAttention(16, 4), _pair),
}


@pytest.mark.parametrize("name", list(CASES))
def test_zero_update_returns_the_input_with_one_identity_gradient(name):
    torch.manual_seed(0)
    call, module, make = CASES[name]()
    inputs = make() if make is _token else (make(),)
    x = inputs[0]
    out = call(module, *inputs)
    torch.testing.assert_close(out, x, atol=0, rtol=0)
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)


def _randomize(module):
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(std=0.2)
    return module


def test_augmented_attention_forward_is_input_plus_delta():
    torch.manual_seed(1)
    m = _randomize(AugmentedAttentionPairBias(32, 16, 8, 4))
    single, cond, pair = _token()
    delta = m.delta(single, cond, pair)
    assert delta.abs().max() > 0
    torch.testing.assert_close(m(single, cond, pair), single + delta, atol=0, rtol=0)


def test_conditioned_transition_forward_is_input_plus_delta():
    torch.manual_seed(2)
    m = _randomize(ConditionedTransition(32, 16, 2))
    single, cond, _ = _token()
    delta = m.delta(single, cond)
    assert delta.abs().max() > 0
    torch.testing.assert_close(m(single, cond), single + delta, atol=0, rtol=0)

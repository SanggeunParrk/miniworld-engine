"""A100 native 32x128 attention inside the AF3 local block, including cross AdaLN and live graph operands."""

import copy

import pytest
import torch
from settings_helpers import override_setting
from tests.integrations.test_b200_local_dit_gpu import (
    _block,
    _inputs,
    _mask,
    _rel,
    _train,
)

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import LocalDiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
        reason="A100 required",
    ),
]


@pytest.fixture(autouse=True)
def enable_candidate(monkeypatch):
    override_setting(monkeypatch, "local_dit_sm80", "1")


@pytest.mark.parametrize(
    ("samples", "n", "cross", "masked"),
    [
        (1, 65, True, True),
        (3, 130, True, False),
        (2, 300, True, True),
        (4, 384, False, True),
    ],
)
def test_training_matches_existing_b200_accuracy_band(samples, n, cross, masked):
    ref, engine = _block(cross=cross)
    baseline = (
        LocalDiTBlock(cross_attention=cross, implementation=ImplementationType.PYTORCH)
        .cuda()
        .bfloat16()
    )
    baseline.load_state_dict(engine.state_dict())
    inputs = _inputs(samples, n)
    mask = _mask(n, 11) if masked else None
    weight = torch.randn(samples, 1, n, 128, device="cuda", dtype=torch.float64)
    truth = _train(ref, inputs, torch.float64, weight, mask)
    actual = _train(engine, inputs, torch.bfloat16, weight, mask)
    standard = _train(baseline, inputs, torch.bfloat16, weight, mask)
    assert actual.keys() == truth.keys()
    for name in truth:
        assert torch.isfinite(actual[name]).all(), name
        error, floor = (
            _rel(actual[name], truth[name]),
            _rel(standard[name], truth[name]),
        )
        assert error < 1.5 * floor + 3e-3, (
            f"{name}: CUDA {error:.3e}, module {floor:.3e}"
        )


@pytest.mark.parametrize(
    ("samples", "n", "cross", "masked"),
    [
        (1, 65, True, True),
        (5, 130, True, False),
        (2, 300, True, True),
        (5, 1024, False, False),
    ],
)
def test_inference_matches_existing_b200_accuracy_band(samples, n, cross, masked):
    ref, engine = _block(seed=2, cross=cross)
    baseline = (
        LocalDiTBlock(cross_attention=cross, implementation=ImplementationType.PYTORCH)
        .cuda()
        .bfloat16()
    )
    baseline.load_state_dict(engine.state_dict())
    inputs = _inputs(samples, n, 3)
    mask = _mask(n, 13) if masked else None
    with torch.no_grad():
        truth = ref(*(t.double() for t in inputs), mask)
        actual = engine(*(t.bfloat16() for t in inputs), mask)
        standard = baseline(*(t.bfloat16() for t in inputs), mask)
    assert _rel(actual, truth) < 1.5 * _rel(standard, truth) + 3e-3


def test_all_masked_attention_is_zero_and_residual_is_preserved():
    _, engine = _block(cross=True)
    inputs = [t.bfloat16() for t in _inputs(2, 130)]
    with torch.no_grad():
        delta = engine.attention_delta(
            *inputs, torch.zeros(1, 130, device="cuda", dtype=torch.bool)
        )
    assert torch.count_nonzero(delta) == 0
    with torch.no_grad():
        engine.transition.squeeze.weight.zero_()
    inputs = [t.requires_grad_() for t in inputs]
    output = engine(*inputs, torch.zeros(1, 130, device="cuda", dtype=torch.bool))
    assert torch.equal(output, inputs[0])
    output.sum().backward()
    assert torch.equal(inputs[0].grad, torch.ones_like(inputs[0]))


def test_fullgraph_compile_and_graph_reflect_inputs_masks_pairs_weights():
    _, engine = _block(seed=4, cross=True)
    inputs = [t.bfloat16() for t in _inputs(2, 256, 5)]
    mask = _mask(256, 15)
    compiled = torch.compile(engine, fullgraph=True, dynamic=False)
    with torch.no_grad():
        eager = engine(*inputs, mask)
        torch.testing.assert_close(compiled(*inputs, mask), eager, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = compiled(*inputs, mask)
        for mutate in [
            lambda: inputs[0].add_(0.125),
            lambda: inputs[1].mul_(0.9),
            lambda: inputs[2].add_(0.125),
            lambda: mask.logical_not_(),
            lambda: engine.attention.to_bias.weight.mul_(0.8),
            lambda: engine.attention.to_query.weight.add_(0.02),
            lambda: engine.transition.squeeze.weight.mul_(0.9),
        ]:
            before = output.clone()
            mutate()
            graph.replay()
            torch.cuda.synchronize()
            expected = engine(*inputs, mask)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            assert not torch.equal(output, before), "Graph ignored changed operands"


def test_fullgraph_training_output_and_every_gradient():
    _, engine = _block(seed=7, cross=True)
    inputs = [t.bfloat16() for t in _inputs(3, 256, 8)]
    mask = _mask(256, 17)
    weight = torch.randn(3, 1, 256, 128, device="cuda", dtype=torch.float64)
    eager = _train(engine, inputs, torch.bfloat16, weight, mask)
    compiled = torch.compile(copy.deepcopy(engine), fullgraph=True, dynamic=False)
    leaves = [t.clone().requires_grad_() for t in inputs]
    output = compiled(*leaves, mask)
    (output.double() * weight).sum().backward()
    torch.testing.assert_close(output, eager["out"], rtol=0, atol=0)
    for name, leaf in zip(("dsingle", "dcond", "dpair"), leaves, strict=True):
        assert _rel(leaf.grad, eager[name]) < 2e-2, name
    for name, param in compiled._orig_mod.named_parameters():
        assert _rel(param.grad, eager[name]) < 2e-2, name

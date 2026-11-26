"""Native Philox row dropout, module gradients and live compiled graph replay."""

import pytest
import torch
from settings_helpers import override_setting

from miniworld_engine.kernels.triangle_attention.cuda import (
    dropout_row_scale_sm80 as row_scale_sm80,
)
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention import TriangleAttention
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def ampere(monkeypatch):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("A100 required")
    override_setting(monkeypatch, "dropout_scale_sm80", "1")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("probability", [0.15, 0.25])
def test_native_rng_output_state_and_independent_compiled_calls(
    dtype, dimension, probability
):
    reference = torch.empty(2, 128, 128, 64, device="cuda", dtype=dtype)
    shape = list(reference.shape)
    shape[dimension] = 1
    torch.manual_seed(413)
    expected = (torch.rand(shape, device="cuda", dtype=dtype) > probability).to(
        dtype
    ) / (1 - probability)
    expected_state = torch.cuda.get_rng_state()
    torch.manual_seed(413)
    actual = row_scale_sm80.scale(reference, probability, dimension)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.cuda.get_rng_state(), expected_state)
    compiled = torch.compile(
        lambda x: (
            row_scale_sm80.scale(x, probability, dimension),
            row_scale_sm80.scale(x, probability, dimension),
        ),
        fullgraph=True,
        options={"triton.cudagraphs": False},
    )
    first, second = compiled(reference)
    assert not torch.equal(first, second)


def _module(kind, width):
    kwargs = {"implementation": ImplementationType.MINIWORLD, "p_drop": 0.25}
    if kind == "bidir":
        module = BidirectionalTriangleMultiplication(width, **kwargs)
    elif kind in ("starting", "ending"):
        module = TriangleAttention(
            width, width // 32, d_hidden=width, starting=kind == "starting", **kwargs
        )
    else:
        module = TriangleMultiplication(
            width, d_hidden=width, outgoing=kind == "outgoing", **kwargs
        )
    module = module.cuda().bfloat16().train()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.ndim >= 2:
                parameter.normal_(std=parameter.shape[-1] ** -0.5)
            elif "weight" in name:
                parameter.copy_(1 + 0.1 * torch.randn_like(parameter))
            else:
                parameter.normal_(std=0.05)
    return module


@pytest.mark.parametrize("width", [64, 128, 256, 384, 512])
@pytest.mark.parametrize(
    "kind", ["outgoing", "incoming", "bidir", "starting", "ending"]
)
def test_seeded_module_output_and_all_gradients_are_bitwise_preserved(
    kind, width, monkeypatch
):
    torch.manual_seed(93)
    module = _module(kind, width)
    pair = torch.randn(
        1, 128, 128, width, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    mask = torch.rand(1, 128, device="cuda") > 0.2
    cotangent = torch.randn_like(pair)
    values = [pair, *module.parameters()]
    results = []
    for flag in ("0", "1"):
        override_setting(monkeypatch, "dropout_scale_sm80", flag)
        torch.manual_seed(812)
        output = module(pair, mask)
        gradients = torch.autograd.grad(output, values, cotangent)
        results.append((output.detach(), *gradients))
    for before, after in zip(*results, strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize("kind", ["outgoing", "bidir", "starting", "ending"])
def test_compiled_training_graph_refreshes_rng_and_reads_live_operands(kind):
    torch.manual_seed(84)
    module = _module(kind, 128)
    pair = torch.randn(
        1, 128, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    mask = torch.rand(1, 128, device="cuda") > 0.2
    cotangent = torch.randn_like(pair)
    compiled = torch.compile(
        module, fullgraph=True, options={"triton.cudagraphs": False}
    )
    values = [pair, *module.parameters()]

    def step():
        output = compiled(pair, mask)
        return (output, *torch.autograd.grad(output, values, cotangent))

    for _ in range(3):
        step()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    graph.replay()
    first = captured[0].clone()
    graph.replay()
    assert not torch.equal(first, captured[0])
    with torch.no_grad():
        pair.add_(0.1 * torch.randn_like(pair))
        mask.logical_not_()
        module.to_out.weight.mul_(0.7)
        module.ln_pair.bias.add_(0.1)
    torch.manual_seed(714)
    graph.replay()
    actual = [value.clone() for value in captured]
    torch.manual_seed(714)
    expected = step()
    for replay, eager_compiled in zip(actual, expected, strict=True):
        assert torch.equal(replay, eager_compiled)


def test_external_implementation_keeps_native_rng(monkeypatch):
    module = TriangleMultiplication(
        128, implementation=ImplementationType.PYTORCH, p_drop=0.25
    ).cuda()
    pair = torch.empty(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(
        row_scale_sm80, "scale", lambda *args: pytest.fail("External path used our RNG")
    )
    torch.manual_seed(32)
    actual = module._make_drop_row_scale(pair, 0.25)
    torch.manual_seed(32)
    expected = (torch.rand(1, 1, 128, 128, device="cuda", dtype=pair.dtype) > 0.25).to(
        pair.dtype
    ) / 0.75
    assert torch.equal(actual, expected)

"""D512 native training dispatch, original accuracy bands, and live graph state."""

import pytest
import torch
from settings_helpers import override_setting
from tests.integrations.test_a100_trimul_gpu import (
    KINDS,
    NAMES,
    _inputs,
    _module,
    _rel,
    _run,
)

from miniworld_engine.integrations import trimul_sm80
from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide, sm80_wide512
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
AMPERE = torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


@pytest.mark.skipif(not AMPERE, reason="A100 required")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("masters", [torch.bfloat16, torch.float32])
def test_d512_module_dispatch_preserves_every_gradient(kind, masters, monkeypatch):
    module = _module(kind, 512, ImplementationType.MINIWORLD)
    z, mask, ds, dy = _inputs(128, 512)
    reference = _run(
        module,
        z,
        mask,
        ds,
        dy,
        dtype=torch.float32,
        implementation=ImplementationType.PYTORCH,
    )
    floor = _run(module, z, mask, ds, dy, implementation=ImplementationType.PYTORCH)
    module = module.to(masters).train()
    module._make_drop_row_scale = lambda pair, p: ds.to(pair.dtype)
    pair = z.bfloat16().requires_grad_()
    assert trimul_sm80.serves(module, pair, mask)
    assert trimul_sm80._wide_kernels(pair) is sm80_wide512
    y = module(pair, mask)
    parameters = dict(module.named_parameters())
    gradients = torch.autograd.grad(
        y, [pair, *[parameters[name] for name in NAMES]], dy.bfloat16()
    )
    result = {
        "out": y.detach(),
        "dz": gradients[0],
        **dict(zip(NAMES, gradients[1:], strict=False)),
    }
    for name, want in reference.items():
        assert _rel(result[name].float(), want) <= max(
            1.25 * _rel(floor[name], want), 2e-3
        ), name
    for name, gradient in zip(NAMES, gradients[1:], strict=False):
        assert gradient.dtype == parameters[name].dtype
    override_setting(monkeypatch, "trimul_sm80_wide512", "0")
    assert trimul_sm80._wide_kernels(pair) is sm80_wide
    assert not trimul_sm80.serves(module, pair, mask)
    with torch.no_grad():
        assert trimul_sm80.serves(module, pair, mask)
        assert trimul_sm80._wide_kernels(pair) is sm80_wide


@pytest.mark.skipif(not AMPERE, reason="A100 required")
@pytest.mark.parametrize("kind", KINDS)
def test_d512_compiled_training_graph_reads_live_input_mask_weights(kind):
    torch._dynamo.reset()
    module = _module(kind, 512, ImplementationType.MINIWORLD).bfloat16().train()
    z, mask, ds, dy = _inputs(128, 512)
    ds = ds.bfloat16()
    pair, cotangent = z.bfloat16().requires_grad_(), dy.bfloat16()
    module._make_drop_row_scale = lambda value, p: ds.to(value.dtype)
    compiled = torch.compile(
        module, fullgraph=True, options={"triton.cudagraphs": False}
    )
    values = [pair, *module.parameters()]

    def step():
        y = compiled(pair, mask)
        grads = torch.autograd.grad(y, values, cotangent)
        return [y.detach(), *[g.detach() for g in grads]]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = step()
    graph.replay()
    before = outputs[0].clone()
    with torch.no_grad():
        pair.mul_(0.8)
        mask.logical_not_()
        module.to_out.weight.mul_(0.7)
        module.ln_pair.bias.add_(0.01)
    graph.replay()
    actual = [x.clone() for x in outputs]
    expected = step()
    assert not torch.equal(actual[0], before)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=False))
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
    kernels = [
        event.name
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert any("gate_bwd_kernel<512" in name for name in kernels)
    assert not any("triton" in name.lower() for name in kernels)
    # The measured backend still contains cuBLAS GEMMs; no whole-CUDA claim here.

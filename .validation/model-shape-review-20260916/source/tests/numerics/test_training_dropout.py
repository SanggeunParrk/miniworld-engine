"""Compiled training must retain .25 dropout, residuals, and all gradients."""
from unittest.mock import patch

import pytest
import torch
from benchmarks.runners.measurement import (
    compile_module_for_benchmark,
    observe_execution,
    require_compile_evidence,
)

from miniworld_engine.modules import TriangleAttention, TriangleMultiplication
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def close(name, actual, expected):
    assert actual is not None, name
    assert expected is not None, name
    assert torch.isfinite(actual).all(), name
    assert torch.isfinite(expected).all(), name
    relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
    assert relative < .03, (name, float(relative))


@pytest.mark.parametrize("kind", ["outgoing", "incoming", "bidir", "attention_start", "attention_end"])
@pytest.mark.parametrize("implementation", ["miniworld", "cuequivariance"])
def test_compiled_dropout_matches_full_reference_gradient_and_restores_rng(kind, implementation):
    torch._dynamo.reset()
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    attention = kind.startswith("attention")
    cls = (TriangleAttention if attention else BidirectionalTriangleMultiplication
           if kind == "bidir" else TriangleMultiplication)
    options = ({"starting": kind == "attention_start"} if attention else {}
               if kind == "bidir" else {"outgoing": kind == "outgoing"})
    actual = cls(128, implementation=implementation, p_drop=.25, **options).cuda().to(torch.bfloat16).train()
    reference = cls(128, implementation=ImplementationType.PYTORCH, p_drop=.25, **options).cuda().float().train()
    with torch.no_grad():
        for module in actual.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(std=128 ** -.5)
                if module.bias is not None:
                    module.bias.normal_(std=.05)
    reference.load_state_dict(actual.state_dict())
    x = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().float().requires_grad_(True)
    mask = torch.ones(1, 128, device="cuda", dtype=torch.bool)
    mask[:, ::7] = False
    shape = list(x.shape)
    shape[2 if kind == "attention_end" else 1] = 1
    scale = (torch.rand(shape, device="cuda", dtype=x.dtype) > .25).to(x.dtype) / .75
    assert (scale == 0).any()
    assert (scale > 0).any()
    method = "_make_drop_scale" if attention else "_make_drop_row_scale"
    generate = lambda pair, probability: scale.to(pair.dtype)
    compile_module_for_benchmark(actual)
    with patch.object(actual, method, generate), patch.object(reference, method, generate):
        y, yr = actual(x, mask), reference(xr, mask)
        dy = torch.randn_like(y)
        y.backward(dy)
        yr.backward(dy.float())
        close("output", y, yr)
        close("input", x.grad, xr.grad)
        for (name, param), (ref_name, ref_param) in zip(
                actual.named_parameters(), reference.named_parameters(), strict=True):
            assert name == ref_name
            close(name, param.grad, ref_param.grad)
    assert method not in actual.__dict__
    # Exercise the restored production RNG and compiled backward, as the timer does.
    outputs = []
    with observe_execution() as evidence:
        for _ in range(2):
            x.grad = None
            actual.zero_grad(set_to_none=True)
            y = actual(x, mask)
            y.backward(dy)
            outputs.append(y.detach().clone())
    require_compile_evidence(True, evidence)
    assert not torch.equal(outputs[0], outputs[1]), "dropout mask was accidentally frozen for timing"
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

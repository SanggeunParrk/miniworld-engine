"""fp32 activations (TF32 tensor cores): the portable Transition's training forward keeps the expand halves in fp16 for the backward
-- the same 10-bit mantissa as TF32, half the bytes of fp32. fp16 does not have fp32's range, so they are stored divided by a device
scalar derived from a bound on |a|, |b| (``residual.half_scale``: LayerNorm output is bounded by sqrt(D) |gamma| + |beta|, times the
weights' row L1 norms): never an overflow, a scale of 1 for ordinary magnitudes, no host sync.

The claim: the fp16 halves cost nothing in accuracy against the fp32 halves (error against an fp64 run of the same inputs and cotangent,
at most 1.5x), at ordinary magnitudes and at magnitudes whose pre-activations are ~1e5, which overflow fp16 unscaled."""
import copy

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu


def _grads(mod, x, dy):
    mod = copy.deepcopy(mod)
    xx = x.clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return {"out": y.detach().double(), "dx": xx.grad.double(), **{n: p.grad.double() for n, p in mod.named_parameters()}}


def _rel(got, want):
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


@pytest.mark.parametrize("scale", [1.0, 2e4])
def test_fp16_halves_cost_no_accuracy_and_never_overflow(scale, monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.triton import main as kernel
    from miniworld_engine.kernels.transition.triton.residual import half_scale
    from miniworld_engine.modules import Transition

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(force_portable_kernels=True)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.manual_seed(7)
        d, shape = 128, (1, 48, 48, 128)
        module = Transition(d, n=4, implementation=ImplementationType.MINIWORLD).cuda().float()
        with torch.no_grad():
            for name, p in module.named_parameters():
                if p.ndim == 2:
                    p.normal_(std=d**-0.5)
                    if "expand" in name:
                        p.mul_(scale)
        x = torch.randn(shape, device="cuda")
        dy = torch.randn(shape, device="cuda")
        s = half_scale(module.ln_in.weight, module.ln_in.bias, module.expand_a.weight, module.expand_b.weight)
        assert (float(s) > 1.0) == (scale > 1), float(s)           # only the magnitudes fp16 could not hold are scaled

        ref = _grads(copy.deepcopy(module).double(), x.double(), dy.double())
        monkeypatch.setattr(kernel, "_SAVED_HALVES_FP32", False)
        half = _grads(module, x, dy)
        monkeypatch.setattr(kernel, "_SAVED_HALVES_FP32", True)
        full = _grads(module, x, dy)
        for name, want in ref.items():
            assert torch.isfinite(half[name]).all(), name
            assert _rel(half[name], want) <= max(1.5 * _rel(full[name], want), 1e-5), (
                name, _rel(half[name], want), _rel(full[name], want))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

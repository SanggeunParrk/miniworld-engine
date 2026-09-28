"""The fused sm_80 Transition (one forward kernel, two backward kernels) matches the Triton residual path it replaces on A100,
and only takes the calls it is built for.  Same claims as the sm_90a test: the gate is the safety story, the numbers are no
further from fp32 than today's path, the module really dispatches to it, and a replay is bit-identical."""
import copy

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
AMPERE = CUDA and torch.cuda.get_device_capability() == (8, 0)
needs_ampere = pytest.mark.skipif(not AMPERE, reason="the fused sm80 Transition is sm_80 only")


def _weights(d=128, h=512, dev="cuda"):
    return torch.randn(h, d, device=dev, dtype=torch.bfloat16), torch.randn(d, h, device=dev, dtype=torch.bfloat16)


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_gate_rejects_everything_it_is_not_built_for():
    from miniworld_engine.kernels.transition.cuda import fused_sm80

    wa, ws = _weights()
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    assert fused_sm80.supported(x, wa, ws) is AMPERE
    assert not fused_sm80.supported(x.float(), wa, ws)
    assert not fused_sm80.supported(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16), *_weights(256, 1024))
    assert not fused_sm80.supported(torch.randn(128, 128, device="cuda", dtype=torch.bfloat16), wa, ws)   # not whole 256-row tiles
    assert not fused_sm80.supported(x.cpu(), wa.cpu(), ws.cpu())
    assert fused_sm80.supported(torch.randn(1, 16, 16, 128, device="cuda", dtype=torch.bfloat16), wa, ws) is AMPERE


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_env_switch_turns_the_gate_off(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_sm80

    wa, ws = _weights()
    monkeypatch.setenv("MINIWORLD_TRANSITION_FUSED_SM80", "0")
    assert not fused_sm80.supported(torch.randn(256, 128, device="cuda", dtype=torch.bfloat16), wa, ws)


def _build(shape, seed=72):
    from miniworld_engine.modules import Transition

    torch.manual_seed(seed)
    module = Transition(shape[-1], n=4, implementation=ImplementationType.TRITON).cuda().bfloat16()
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 2:          # the zero-init squeeze would make four of five gradients exactly zero everywhere
                param.normal_(std=shape[-1] ** -0.5)
    return module, torch.randn(shape, device="cuda", dtype=torch.bfloat16), torch.randn(shape, device="cuda", dtype=torch.bfloat16)


def _run(module, x, dy, *, fused, monkeypatch, fp32=False):
    mod = copy.deepcopy(module)
    if fp32:
        mod, x, dy = mod.float(), x.float(), dy.float()
    monkeypatch.setenv("MINIWORLD_TRANSITION_FUSED_SM80", "1" if fused else "0")
    xx = x.clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return {"out": y.detach().float(), "dx": xx.grad.detach().float(),
            **{name: p.grad.detach().float() for name, p in mod.named_parameters()}}


def _rel(got, want):
    return float((got - want).norm() / want.norm().clamp_min(1e-20))


@needs_ampere
@pytest.mark.parametrize("shape", [(1, 32, 32, 128), (2, 16, 16, 128)])
def test_is_no_less_accurate_than_the_triton_path(shape, monkeypatch):
    module, x, dy = _build(shape)
    reference = _run(module, x, dy, fused=False, fp32=True, monkeypatch=monkeypatch)
    fused = _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    triton = _run(module, x, dy, fused=False, monkeypatch=monkeypatch)
    for name in reference:
        got, base = _rel(fused[name], reference[name]), _rel(triton[name], reference[name])
        assert got <= max(base * 1.5, 1e-6), f"{name}: fused {got:.3e} vs triton {base:.3e}"


@needs_ampere
@pytest.mark.parametrize("shape", [(1, 32, 32, 128), (2, 16, 16, 128)])
def test_agrees_with_the_triton_path_to_bf16(shape, monkeypatch):
    module, x, dy = _build(shape)
    fused = _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    triton = _run(module, x, dy, fused=False, monkeypatch=monkeypatch)
    tolerances = {"out": 4e-3, "dx": 5e-3}
    for name in triton:
        assert _rel(fused[name], triton[name]) < tolerances.get(name, 6e-3), name


@needs_ampere
def test_the_module_actually_dispatches_to_it(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_sm80

    calls = []
    original = fused_sm80.transition_fused_sm80
    monkeypatch.setattr(fused_sm80, "transition_fused_sm80", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    module, x, dy = _build((1, 32, 32, 128))
    _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    assert calls, "the fused path was never entered"


@needs_ampere
def test_inference_matches_training_forward():
    """No grad -> the forward skips the xn / stats stores (runtime guard); the output bits must not change."""
    from miniworld_engine.kernels.transition.cuda import fused_sm80

    module, x, _ = _build((1, 32, 32, 128))
    args = (module.ln_in.weight, module.ln_in.bias, module.expand_a.weight, module.expand_b.weight, module.squeeze.weight, module.ln_in.eps)
    with torch.no_grad():
        inf = fused_sm80.transition_fused_sm80(x, *args)
    tr = fused_sm80.transition_fused_sm80(x.clone().requires_grad_(), *args)
    assert torch.equal(inf, tr.detach())


@needs_ampere
def test_replay_is_bit_identical():
    from miniworld_engine.kernels.transition.cuda import fused_sm80

    torch.manual_seed(11)
    d, h, m = 128, 512, 1024
    x = torch.randn(m, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gamma = torch.rand(d, device="cuda", dtype=torch.bfloat16) + 0.5
    beta = torch.randn(d, device="cuda", dtype=torch.bfloat16) * 0.1
    wa = (torch.randn(h, d, device="cuda") * d**-0.5).bfloat16().requires_grad_()
    wb = (torch.randn(h, d, device="cuda") * d**-0.5).bfloat16().requires_grad_()
    ws = (torch.randn(d, h, device="cuda") * h**-0.5).bfloat16().requires_grad_()
    dy = torch.randn(m, d, device="cuda", dtype=torch.bfloat16)

    def once():
        for t in (x, wa, wb, ws):
            t.grad = None
        y = fused_sm80.transition_fused_sm80(x, gamma, beta, wa, wb, ws, 1e-5)
        y.backward(dy)
        return [y.detach().clone()] + [t.grad.clone() for t in (x, wa, wb, ws)]

    first, second = once(), once()
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


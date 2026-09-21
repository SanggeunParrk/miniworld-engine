"""The fused sm_90a Transition matches the Triton residual path it replaces, and only takes the
calls it is built for.

Two things are worth pinning down. The gate is the whole safety story -- the kernel has tile
shapes, a shared-memory budget and a CTA split written for one shape, so anything it wrongly
accepts is a wrong answer rather than a slow one. And the numbers have to agree with the path
that is in use today, not just with a reference, because switching it on changes what every
existing training run computes.
"""
import copy

import pytest
import torch

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
HOPPER = CUDA and torch.cuda.get_device_capability() == (9, 0)
needs_hopper = pytest.mark.skipif(not HOPPER, reason="the fused Transition is sm_90a only")


def _weights(d=128, h=512, dev="cuda"):
    wa = torch.randn(h, d, device=dev, dtype=torch.bfloat16)
    ws = torch.randn(d, h, device=dev, dtype=torch.bfloat16)
    return wa, ws


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_gate_rejects_everything_it_is_not_built_for():
    from miniworld_engine.kernels.transition.cuda import fused_sm90a

    wa, ws = _weights()
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    assert fused_sm90a.supported(x, wa, ws) is HOPPER

    # Wrong dtype, wrong width, wrong hidden multiple, and a row count that is not a whole
    # 128-row tile: the persistent grid has no ragged tail.
    assert not fused_sm90a.supported(x.float(), wa, ws)
    assert not fused_sm90a.supported(torch.randn(128, 256, device="cuda", dtype=torch.bfloat16),
                                     *_weights(256, 1024))
    assert not fused_sm90a.supported(torch.randn(127, 128, device="cuda", dtype=torch.bfloat16), wa, ws)
    assert not fused_sm90a.supported(x.cpu(), wa.cpu(), ws.cpu())
    # A 4-D pair activation is flattened over every leading axis, so 2*8*8 = 128 rows passes.
    assert fused_sm90a.supported(
        torch.randn(2, 8, 8, 128, device="cuda", dtype=torch.bfloat16), wa, ws) is HOPPER


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_env_switch_turns_the_gate_off(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_sm90a

    wa, ws = _weights()
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setenv("MINIWORLD_TRANSITION_FUSED_SM90A", "0")
    assert not fused_sm90a.supported(x, wa, ws)


def _build(shape, seed=72):
    from miniworld_engine.modules import Transition

    torch.manual_seed(seed)
    module = Transition(shape[-1], n=4, implementation="triton").cuda().bfloat16()
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 2:
                # The squeeze weight is zero-init, which makes W_s = 0, dh = 0 and four of the
                # five parameter gradients exactly zero in every backend -- a comparison that
                # agrees perfectly while proving nothing.
                param.normal_(std=shape[-1] ** -0.5)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    return module, x, dy


def _run(module, x, dy, *, fused, fp32=False):
    from miniworld_engine import settings

    # deepcopy: nn.Module.float() is in place, so converting the shared module would corrupt
    # the bf16 run that comes after it.
    mod = copy.deepcopy(module)
    if fp32:
        mod = mod.float()
        x, dy = x.float(), dy.float()
    settings.configure(engine_backend="triton", transition_residual_fusion=True,
                       transition_fused_sm90a=fused)
    xx = x.clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return {"out": y.detach().float(), "dx": xx.grad.detach().float(),
            **{name: p.grad.detach().float() for name, p in mod.named_parameters()}}


def _rel(got, want):
    return float((got - want).norm() / want.norm().clamp_min(1e-20))


@needs_hopper
@pytest.mark.parametrize("shape", [(1, 32, 32, 128), (2, 16, 16, 128)])
def test_is_no_less_accurate_than_the_triton_path(shape, monkeypatch):
    """Both paths round to bf16 at the same places but not in the same order, so they differ
    from each other by about as much as either differs from fp32 -- 2.5e-3 on the output, which
    says nothing on its own. The claim that matters is that the fused path is no further from
    an fp32 run than the path it replaces."""
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build(shape)
    reference = _run(module, x, dy, fused=False, fp32=True)
    fused = _run(module, x, dy, fused=True)
    triton = _run(module, x, dy, fused=False)

    for name in reference:
        got, base = _rel(fused[name], reference[name]), _rel(triton[name], reference[name])
        assert got <= max(base * 1.5, 1e-6), f"{name}: fused {got:.3e} vs triton {base:.3e}"


@needs_hopper
@pytest.mark.parametrize("shape", [(1, 32, 32, 128), (2, 16, 16, 128)])
def test_agrees_with_the_triton_path_to_bf16(shape, monkeypatch):
    """Switching this on changes what every existing run computes, so bound how much. The
    tolerances are one bf16 rounding on the activations and a little more on the parameter
    gradients, which accumulate over all M rows in a different CTA order."""
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build(shape)
    fused = _run(module, x, dy, fused=True)
    triton = _run(module, x, dy, fused=False)

    tolerances = {"out": 4e-3, "dx": 5e-4}
    for name in triton:
        assert _rel(fused[name], triton[name]) < tolerances.get(name, 2e-3), name


@needs_hopper
def test_the_module_actually_dispatches_to_it(monkeypatch):
    """A parity test passes just as well when nothing is wired up, so check the call happens."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import fused_sm90a

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    calls = []
    original = fused_sm90a.transition_fused_sm90a
    monkeypatch.setattr(fused_sm90a, "transition_fused_sm90a",
                        lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    module, x, dy = _build((1, 32, 32, 128))
    _run(module, x, dy, fused=True)
    assert calls, "the fused path was never entered"


@needs_hopper
def test_replay_is_bit_identical():
    """No atomics anywhere, so the same inputs give the same bits -- which is what makes a
    training run reproducible and a regression bisectable."""
    from miniworld_engine.kernels.transition.cuda import fused_sm90a

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
        y = fused_sm90a.transition_fused_sm90a(x, gamma, beta, wa, wb, ws, 1e-5)
        y.backward(dy)
        return [y.detach().clone()] + [t.grad.clone() for t in (x, wa, wb, ws)]

    first, second = once(), once()
    assert all(torch.equal(a, b) for a, b in zip(first, second))

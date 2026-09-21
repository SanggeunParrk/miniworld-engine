"""The fused sm_90a Transition matches the Triton residual path it replaces, and only takes the
calls it is built for.

Two things are worth pinning down. The gate is the whole safety story -- the kernel has tile
shapes, a shared-memory budget and a CTA split written for one shape, so anything it wrongly
accepts is a wrong answer rather than a slow one. And the numbers have to agree with the path
that is in use today, not just with a reference, because switching it on changes what every
existing training run computes.
"""
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


def _module_run(shape, *, fused, seed=72):
    from miniworld_engine import settings
    from miniworld_engine.modules import Transition

    settings.configure(engine_backend="triton", transition_residual_fusion=True,
                       transition_fused_sm90a=fused)
    torch.manual_seed(seed)
    module = Transition(shape[-1], n=4, implementation="triton").cuda().bfloat16()
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 2:
                param.normal_(std=shape[-1] ** -0.5)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = module(x)
    y.backward(torch.randn_like(y))
    grads = {name: p.grad.detach().float() for name, p in module.named_parameters()}
    return y.detach().float(), x.grad.detach().float(), grads


@needs_hopper
@pytest.mark.parametrize("shape", [(1, 32, 32, 128), (2, 16, 16, 128)])
def test_matches_the_triton_residual_path(shape, monkeypatch):
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    fused_y, fused_dx, fused_grads = _module_run(shape, fused=True)
    triton_y, triton_dx, triton_grads = _module_run(shape, fused=False)

    def close(a, b, tol):
        return float((a - b).norm() / b.norm().clamp_min(1e-20)) < tol

    assert close(fused_y, triton_y, 2e-3), "forward"
    assert close(fused_dx, triton_dx, 2e-3), "input gradient"
    for name in triton_grads:
        assert close(fused_grads[name], triton_grads[name], 5e-3), name


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
    _module_run((1, 32, 32, 128), fused=True)
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

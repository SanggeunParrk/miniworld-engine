"""The wide-width sm_90a Transition (D = 64, 256, 384, 512) matches the Triton residual path it replaces, and only takes
the calls it is built for. Same claims as ``test_transition_fused_sm90a_gpu`` (D = 128), per width.
"""
import copy

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
HOPPER = CUDA and torch.cuda.get_device_capability() == (9, 0)
needs_hopper = pytest.mark.skipif(not HOPPER, reason="the wide Transition kernels are sm_90a only")
WIDTHS = (64, 256, 384, 512)


def _weights(d, h=None, dev="cuda"):
    h = h or 4 * d
    return (torch.randn(h, d, device=dev, dtype=torch.bfloat16), torch.randn(d, h, device=dev, dtype=torch.bfloat16))


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
@pytest.mark.parametrize("d", WIDTHS)
def test_gate_rejects_everything_it_is_not_built_for(d):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as wide

    wa, ws = _weights(d)
    x = torch.randn(128, d, device="cuda", dtype=torch.bfloat16)
    assert wide.supported(x, wa, ws) is HOPPER
    assert not wide.supported(x.float(), wa, ws)
    assert not wide.supported(x, *_weights(d, 2 * d))                        # hidden must be 4 D
    assert not wide.supported(torch.randn(127, d, device="cuda", dtype=torch.bfloat16), wa, ws)
    assert not wide.supported(x.cpu(), wa.cpu(), ws.cpu())
    assert wide.supported(torch.randn(2, 8, 8, d, device="cuda", dtype=torch.bfloat16), wa, ws) is HOPPER


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_widths_without_a_build_and_the_env_switch(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as wide

    for d in (32, 96, 128, 192, 768):                                        # 128 is fused_sm90a's
        assert not wide.supported(torch.randn(128, d, device="cuda", dtype=torch.bfloat16), *_weights(d))
    wa, ws = _weights(256)
    monkeypatch.setenv("MINIWORLD_TRANSITION_WIDE_SM90A", "0")
    assert not wide.supported(torch.randn(128, 256, device="cuda", dtype=torch.bfloat16), wa, ws)


def _build(shape, seed=72):
    from miniworld_engine.modules import Transition

    torch.manual_seed(seed)
    d = shape[-1]
    module = Transition(d, n=4, implementation=ImplementationType.TRITON).cuda().bfloat16()
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 2:
                param.normal_(std=param.shape[-1] ** -0.5)                   # the squeeze weight is zero-init
            elif param is module.ln_in.weight:
                param.copy_(1 + 0.2 * torch.randn_like(param))                # default gamma / beta hide LN bugs
            else:
                param.normal_(std=0.2)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    return module, x, dy


def _run(module, x, dy, *, fused, fp32=False):
    from miniworld_engine import settings

    mod = copy.deepcopy(module)                                               # .float() is in place
    if fp32:
        mod = mod.float()
        x, dy = x.float(), dy.float()
    settings.configure(engine_backend="auto" if fused else "triton", transition_residual_fusion=True, transition_fused_sm90a=fused)
    xx = x.clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return {"out": y.detach().float(), "dx": xx.grad.detach().float(),
            **{name: p.grad.detach().float() for name, p in mod.named_parameters()}}


def _rel(got, want):
    return float((got - want).norm() / want.norm().clamp_min(1e-20))


SHAPES = [(1, 16, 16), (2, 16, 32)]


@needs_hopper
@pytest.mark.parametrize("d", WIDTHS)
@pytest.mark.parametrize("lead", SHAPES)
def test_is_no_less_accurate_than_the_triton_path(d, lead, monkeypatch):
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build((*lead, d))
    reference = _run(module, x, dy, fused=False, fp32=True)
    wide = _run(module, x, dy, fused=True)
    triton = _run(module, x, dy, fused=False)
    for name in reference:
        got, base = _rel(wide[name], reference[name]), _rel(triton[name], reference[name])
        assert got <= max(base * 1.5, 1e-6), f"D{d} {name}: wide {got:.3e} vs triton {base:.3e}"


@needs_hopper
@pytest.mark.parametrize("d", WIDTHS)
def test_agrees_with_the_triton_path_to_bf16(d, monkeypatch):
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build((1, 16, 32, d))
    wide = _run(module, x, dy, fused=True)
    triton = _run(module, x, dy, fused=False)
    tolerances = {"out": 4e-3, "dx": 5e-3}
    for name in triton:
        assert _rel(wide[name], triton[name]) < tolerances.get(name, 6e-3), f"D{d} {name}"


@needs_hopper
@pytest.mark.parametrize("d", WIDTHS)
def test_the_module_actually_dispatches_to_it(d, monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as wide

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    calls = []
    original = wide.transition_wide_sm90a
    monkeypatch.setattr(wide, "transition_wide_sm90a", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    module, x, dy = _build((1, 16, 16, d))
    _run(module, x, dy, fused=True)
    assert calls, "the wide path was never entered"


@needs_hopper
@pytest.mark.parametrize("d", WIDTHS)
def test_no_grad_forward_is_the_training_forward(d, monkeypatch):
    """The inference build skips the saves at run time; the output must not move by a bit."""
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(engine_backend="auto", transition_residual_fusion=True, transition_fused_sm90a=True)
    module, x, _ = _build((1, 16, 16, d))
    with torch.no_grad():
        inference = module(x)
    training = module(x.clone().requires_grad_())
    assert torch.equal(inference, training.detach())


@needs_hopper
@pytest.mark.parametrize("d", WIDTHS)
def test_replay_is_bit_identical(d, monkeypatch):
    """Deterministic everywhere except the engine LayerNorm backward used at D >= 384, whose dgamma / dbeta are summed
    with atomics -- those two are exempt there, exactly as they are on the Triton path."""
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build((1, 16, 32, d))
    first, second = _run(module, x, dy, fused=True), _run(module, x, dy, fused=True)
    exempt = {"ln_in.weight", "ln_in.bias"} if d >= 384 else set()
    for name in first:
        if name not in exempt:
            assert torch.equal(first[name], second[name]), f"D{d} {name}"


@needs_hopper
@pytest.mark.parametrize("d", [384, 512])
def test_saved_h_changes_nothing_but_the_kernel(d, monkeypatch):
    """MINIWORLD_TRANSITION_WIDE_SAVE_H=1 keeps the forward's h and skips its store in the gate kernel. Only dWs reads h, and
    the forward's h comes from a different GEMM tiling than the gate kernel's recomputed one (same bf16 xn and weights, another
    fp32 summation order), so dWs may move in the last bits; everything else is bit-identical."""
    from miniworld_engine import settings

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    module, x, dy = _build((1, 16, 32, d))
    monkeypatch.setenv("MINIWORLD_TRANSITION_WIDE_SAVE_H", "0")
    recomputed = _run(module, x, dy, fused=True)
    monkeypatch.setenv("MINIWORLD_TRANSITION_WIDE_SAVE_H", "1")
    saved = _run(module, x, dy, fused=True)
    for name in recomputed:
        if name in ("ln_in.weight", "ln_in.bias"):                           # engine LN backward: atomics
            assert _rel(saved[name], recomputed[name]) < 1e-5, name
        elif name == "squeeze.weight":
            assert _rel(saved[name], recomputed[name]) < 2e-3, name
        else:
            assert torch.equal(saved[name], recomputed[name]), name

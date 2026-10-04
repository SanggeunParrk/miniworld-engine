"""The fused sm_100a Transition (one forward kernel, one backward kernel + a partial reduction) matches the Triton residual path it
replaces on B200, and only takes the calls it is built for.  Same claims as the sm_90a / sm_80 tests: the gate is the safety story,
the numbers are no further from fp32 than today's path, the module really dispatches to it, and a replay is bit-identical.  The
shapes cover one tile (a 2-CTA pair with a dummy partner), odd and even tile counts, and the L384 pair size."""
import copy

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
B200 = CUDA and torch.cuda.get_device_capability() == (10, 0)
needs_b200 = pytest.mark.skipif(not B200, reason="the fused sm100a Transition is sm_100 only")

SHAPES = [(1, 8, 16, 128), (1, 40, 48, 128), (2, 64, 64, 128), (1, 384, 384, 128)]


def _weights(d=128, h=512, dev="cuda", dtype=torch.bfloat16):
    return torch.randn(h, d, device=dev, dtype=dtype), torch.randn(d, h, device=dev, dtype=dtype)


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_gate_rejects_everything_it_is_not_built_for():
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    wa, ws = _weights()
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    assert fused_sm100a.supported(x, wa, ws) is B200
    assert not fused_sm100a.supported(x.float(), wa, ws)
    assert fused_sm100a.supported(x, *_weights(dtype=torch.float32)) is B200     # fp32 master weights: the entry casts them
    assert not fused_sm100a.supported(x, *_weights(dtype=torch.float16))
    assert not fused_sm100a.supported(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16), *_weights(256, 1024))
    assert not fused_sm100a.supported(torch.randn(64, 128, device="cuda", dtype=torch.bfloat16), wa, ws)   # not a whole 128-row tile
    assert not fused_sm100a.supported(x.cpu(), wa.cpu(), ws.cpu())
    assert fused_sm100a.supported(torch.randn(1, 8, 16, 128, device="cuda", dtype=torch.bfloat16), wa, ws) is B200


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_env_switch_turns_the_gate_off(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    wa, ws = _weights()
    monkeypatch.setenv("MINIWORLD_TRANSITION_FUSED_SM100A", "0")
    assert not fused_sm100a.supported(torch.randn(256, 128, device="cuda", dtype=torch.bfloat16), wa, ws)


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
    monkeypatch.setenv("MINIWORLD_TRANSITION_FUSED_SM100A", "1" if fused else "0")
    xx = x.clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return {"out": y.detach().float(), "dx": xx.grad.detach().float(),
            **{name: p.grad.detach().float() for name, p in mod.named_parameters()}}


def _rel(got, want):
    return float((got - want).norm() / want.norm().clamp_min(1e-20))


@needs_b200
@pytest.mark.parametrize("shape", SHAPES)
def test_is_no_less_accurate_than_the_triton_path(shape, monkeypatch):
    module, x, dy = _build(shape)
    reference = _run(module, x, dy, fused=False, fp32=True, monkeypatch=monkeypatch)
    fused = _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    triton = _run(module, x, dy, fused=False, monkeypatch=monkeypatch)
    for name in reference:
        got, base = _rel(fused[name], reference[name]), _rel(triton[name], reference[name])
        assert got <= max(base * 1.5, 1e-6), f"{name}: fused {got:.3e} vs triton {base:.3e}"


@needs_b200
@pytest.mark.parametrize("shape", SHAPES)
def test_agrees_with_the_triton_path_to_bf16(shape, monkeypatch):
    module, x, dy = _build(shape)
    fused = _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    triton = _run(module, x, dy, fused=False, monkeypatch=monkeypatch)
    tolerances = {"out": 4e-3, "dx": 5e-3}
    for name in triton:
        assert _rel(fused[name], triton[name]) < tolerances.get(name, 6e-3), name


@needs_b200
@pytest.mark.parametrize("shape", [(1, 8, 16, 128), (1, 8, 64, 128)])
def test_fewer_tiles_than_weight_replicas(shape, monkeypatch):
    """With fewer tiles than the backward's weight-role replicas some weight CTAs own no tile; their partials must be zeros, not
    whatever the (uninitialized) partial buffer or tensor memory held. Poison the caching allocator with NaN first."""
    poison = torch.full((64 << 20,), float("nan"), device="cuda")
    del poison
    module, x, dy = _build(shape)
    fused = _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    triton = _run(module, x, dy, fused=False, monkeypatch=monkeypatch)
    for name in triton:
        assert torch.isfinite(fused[name]).all(), name
        assert _rel(fused[name], triton[name]) < {"out": 4e-3, "dx": 5e-3}.get(name, 6e-3), name


@needs_b200
def test_the_module_actually_dispatches_to_it(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    calls = []
    original = fused_sm100a.transition_fused_sm100a
    monkeypatch.setattr(fused_sm100a, "transition_fused_sm100a", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    module, x, dy = _build((1, 32, 32, 128))
    _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    assert calls, "the fused path was never entered"


@needs_b200
def test_inference_matches_training_forward():
    """No grad -> the forward skips the xn / statistics stores (runtime guard); the output bits must not change."""
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    module, x, _ = _build((1, 40, 48, 128))
    args = (module.ln_in.weight, module.ln_in.bias, module.expand_a.weight, module.expand_b.weight, module.squeeze.weight,
            module.ln_in.eps)
    with torch.no_grad():
        inf = fused_sm100a.transition_fused_sm100a(x, *args)
    tr = fused_sm100a.transition_fused_sm100a(x.clone().requires_grad_(), *args)
    assert torch.equal(inf, tr.detach())


@needs_b200
def test_replay_is_bit_identical():
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    torch.manual_seed(11)
    d, h, m = 128, 512, 128 * 150
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
        y = fused_sm100a.transition_fused_sm100a(x, gamma, beta, wa, wb, ws, 1e-5)
        y.backward(dy)
        grads = []
        for t in (x, wa, wb, ws):
            assert t.grad is not None
            grads.append(t.grad.clone())
        return [y.detach().clone(), *grads]

    first, second = once(), once()
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))

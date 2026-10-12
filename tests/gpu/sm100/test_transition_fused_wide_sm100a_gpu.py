"""The sm_100a Transition for every (width, n) but D = 128 / n = 4 (``fused_wide_sm100a``: n = 4 at D = 64 / 256 / 384 / 512,
n = 2 at D = 64 / 128 / 256 / 384 / 512 / 768) matches the Triton residual path it replaces on B200, and only takes the calls it is
built for. Same claims as the D = 128 test (``test_transition_fused_sm100a_gpu.py``): the gate is the safety story, the numbers are
no further from fp32 than today's path, the module really dispatches to it, and a replay is bit-identical. The shapes cover one
tile (a 2-CTA pair with a dummy partner), odd and even tile counts, and the L384 pair; at D >= 384 also 1200 tiles, above the
item-schedule threshold (1152), so both SwiGLU schedules run; and MSA-shaped activations [B, N_msa, L, D]."""
import copy

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
B200 = CUDA and torch.cuda.get_device_capability() == (10, 0)
needs_b200 = pytest.mark.skipif(not B200, reason="the wide sm100a Transition is sm_100 only")

SHAPES = ((64, 4), (256, 4), (384, 4), (512, 4), (64, 2), (128, 2), (256, 2), (384, 2), (512, 2), (768, 2))
ROWS = [(1, 8, 16), (1, 24, 16), (1, 40, 48), (1, 384, 384)]
CASES = ([((*r, d), n) for d, n in SHAPES for r in ROWS]
         + [((1, 400, 384, d), n) for d, n in SHAPES if d >= 384]
         + [((1, 64, 48, d), n) for d, n in ((64, 4), (128, 4))]           # MSA: [B, N_msa, L, d_msa]
         + [((1, 768, 384), 2), ((1, 384, 384), 4), ((2, 640, 384), 2)])   # single stream [B, L, D]: the small-M forward


def _id(case):
    shape, n = case
    return f"{'x'.join(map(str, shape))}-n{n}"


def _weights(d, n=4, dev="cuda", dtype=torch.bfloat16):
    return torch.randn(n * d, d, device=dev, dtype=dtype), torch.randn(d, n * d, device=dev, dtype=dtype)


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
@pytest.mark.parametrize(("d", "n"), SHAPES)
def test_gate_rejects_everything_it_is_not_built_for(d, n):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as w

    wa, ws = _weights(d, n)
    x = torch.randn(256, d, device="cuda", dtype=torch.bfloat16)
    assert w.supported(x, wa, ws) is B200
    assert not w.supported(x.float(), wa, ws)
    assert w.supported(x, *_weights(d, n, dtype=torch.float32)) is B200          # fp32 master weights: the entry casts them
    assert not w.supported(x, *_weights(d, n, dtype=torch.float16))
    assert not w.supported(x, *_weights(d, 3))                                                        # n = 3: no build
    assert not w.supported(torch.randn(64, d, device="cuda", dtype=torch.bfloat16), wa, ws)           # not a whole 128-row tile
    assert not w.supported(x.cpu(), wa.cpu(), ws.cpu())


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_gate_leaves_the_other_shapes_alone():
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as w

    x128 = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    assert not w.supported(x128, *_weights(128, 4))                    # D = 128, n = 4: fused_sm100a
    x768 = torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)
    assert not w.supported(x768, *_weights(768, 4))                    # D = 768 has an n = 2 build only


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_env_switch_turns_the_gate_off(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as w

    wa, ws = _weights(256)
    monkeypatch.setenv("MINIWORLD_TRANSITION_FUSED_SM100A", "0")
    assert not w.supported(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16), wa, ws)


def _build(shape, n=4, seed=72):
    from miniworld_engine.modules import Transition

    torch.manual_seed(seed)
    module = Transition(shape[-1], n=n, implementation=ImplementationType.TRITON).cuda().bfloat16()
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 2:          # the zero-init squeeze would make four of five gradients exactly zero everywhere
                param.normal_(std=param.shape[1] ** -0.5)
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
@pytest.mark.parametrize("case", CASES, ids=_id)
def test_is_no_less_accurate_than_the_triton_path(case, monkeypatch):
    shape, n = case
    module, x, dy = _build(shape, n)
    reference = _run(module, x, dy, fused=False, fp32=True, monkeypatch=monkeypatch)
    fused = _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    triton = _run(module, x, dy, fused=False, monkeypatch=monkeypatch)
    for name in reference:
        got, base = _rel(fused[name], reference[name]), _rel(triton[name], reference[name])
        assert got <= max(base * 1.5, 1e-6), f"{name}: fused {got:.3e} vs triton {base:.3e}"


@needs_b200
@pytest.mark.parametrize("case", CASES, ids=_id)
def test_agrees_with_the_triton_path_to_bf16(case, monkeypatch):
    shape, n = case
    module, x, dy = _build(shape, n)
    fused = _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    triton = _run(module, x, dy, fused=False, monkeypatch=monkeypatch)
    for name in triton:
        assert torch.isfinite(fused[name]).all(), name
        assert _rel(fused[name], triton[name]) < {"out": 4e-3, "dx": 6e-3}.get(name, 8e-3), name


@needs_b200
@pytest.mark.parametrize("case", [((1, 8, 16, d), n) for d, n in ((64, 4), (64, 2), (128, 2))] + [((1, 8, 64, 64), 4)], ids=_id)
def test_two_role_backward_with_fewer_tiles_than_weight_replicas(case, monkeypatch):
    """The two-role backward (D = 64, and D = 128 at n = 2): with fewer tiles than weight-role CTAs some own no tile; their
    partials must be zeros, not whatever the (uninitialized) partial buffer or tensor memory held. Poison the allocator first."""
    poison = torch.full((64 << 20,), float("nan"), device="cuda")
    del poison
    shape, n = case
    module, x, dy = _build(shape, n)
    fused = _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    triton = _run(module, x, dy, fused=False, monkeypatch=monkeypatch)
    for name in triton:
        assert torch.isfinite(fused[name]).all(), name
        assert _rel(fused[name], triton[name]) < {"out": 4e-3, "dx": 6e-3}.get(name, 8e-3), name


@needs_b200
@pytest.mark.parametrize(("d", "n"), SHAPES)
def test_the_module_actually_dispatches_to_it(d, n, monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as w

    calls = []
    original = w.transition_wide_sm100a
    monkeypatch.setattr(w, "transition_wide_sm100a", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    module, x, dy = _build((1, 32, 32, d), n)
    _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    assert calls, "the wide path was never entered"


def _args(module):
    return (module.ln_in.weight, module.ln_in.bias, module.expand_a.weight, module.expand_b.weight, module.squeeze.weight,
            module.ln_in.eps)


@needs_b200
@pytest.mark.parametrize(("d", "n"), SHAPES)
def test_inference_matches_training_forward(d, n):
    """No grad -> the forward skips the saves (and at D >= 384 runs the SwiGLU build without a / b stores); the output bits must
    not change."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as w

    module, x, _ = _build((1, 40, 48, d), n)
    with torch.no_grad():
        inf = w.transition_wide_sm100a(x, *_args(module))
    tr = w.transition_wide_sm100a(x.clone().requires_grad_(), *_args(module))
    assert torch.equal(inf, tr.detach())


@needs_b200
@pytest.mark.parametrize(("d", "n"), SHAPES)
def test_replay_is_bit_identical(d, n):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as w

    torch.manual_seed(11)
    h, m = n * d, 128 * 150
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
        y = w.transition_wide_sm100a(x, gamma, beta, wa, wb, ws, 1e-5)
        y.backward(dy)
        grads = [t.grad for t in (x, wa, wb, ws)]
        assert all(g is not None for g in grads)
        return [y.detach().clone()] + [g.clone() for g in grads if g is not None]

    first, second = once(), once()
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))

"""The fused sm_100a Transition (one forward kernel, one backward kernel + a partial reduction) matches the Triton residual path it
replaces on B200, and only takes the calls it is built for.  Same claims as the sm_90a / sm_80 tests: the gate is the safety story,
the numbers are no further from fp32 than today's path, the module really dispatches to it, and a replay is bit-identical.  The
shapes cover one tile (a 2-CTA pair with a dummy partner), odd and even tile counts, and the L384 pair size."""

import copy

import pytest
import torch
from settings_helpers import override_setting

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
B200 = CUDA and torch.cuda.get_device_capability() == (10, 0)
needs_b200 = pytest.mark.skipif(
    not B200, reason="the fused sm100a Transition is sm_100 only"
)

SHAPES = [(1, 8, 16, 128), (1, 40, 48, 128), (2, 64, 64, 128), (1, 384, 384, 128)]


def _weights(d=128, h=512, dev="cuda", dtype=torch.bfloat16):
    return torch.randn(h, d, device=dev, dtype=dtype), torch.randn(
        d, h, device=dev, dtype=dtype
    )


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_gate_rejects_everything_it_is_not_built_for():
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    wa, ws = _weights()
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    assert fused_sm100a.supported(x, wa, ws) is B200
    assert not fused_sm100a.supported(x.float(), wa, ws)
    assert (
        fused_sm100a.supported(x, *_weights(dtype=torch.float32)) is B200
    )  # fp32 master weights: the entry casts them
    assert not fused_sm100a.supported(x, *_weights(dtype=torch.float16))
    assert not fused_sm100a.supported(
        torch.randn(256, 256, device="cuda", dtype=torch.bfloat16), *_weights(256, 1024)
    )
    assert not fused_sm100a.supported(
        torch.randn(64, 128, device="cuda", dtype=torch.bfloat16), wa, ws
    )  # not a whole 128-row tile
    assert not fused_sm100a.supported(x.cpu(), wa.cpu(), ws.cpu())
    assert (
        fused_sm100a.supported(
            torch.randn(1, 8, 16, 128, device="cuda", dtype=torch.bfloat16), wa, ws
        )
        is B200
    )


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_env_switch_turns_the_gate_off(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    wa, ws = _weights()
    override_setting(monkeypatch, "transition_fused_sm100a", "0")
    assert not fused_sm100a.supported(
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16), wa, ws
    )


def _build(shape, seed=72):
    from miniworld_engine.modules import Transition

    torch.manual_seed(seed)
    module = (
        Transition(shape[-1], n=4, implementation=ImplementationType.MINIWORLD)
        .cuda()
        .bfloat16()
    )
    with torch.no_grad():
        for param in module.parameters():
            if (
                param.ndim == 2
            ):  # the zero-init squeeze would make four of five gradients exactly zero everywhere
                param.normal_(std=shape[-1] ** -0.5)
    return (
        module,
        torch.randn(shape, device="cuda", dtype=torch.bfloat16),
        torch.randn(shape, device="cuda", dtype=torch.bfloat16),
    )


def _run(module, x, dy, *, fused, monkeypatch, fp32=False):
    mod = copy.deepcopy(module)
    if fp32:
        mod, x, dy = mod.float(), x.float(), dy.float()
    override_setting(monkeypatch, "transition_fused_sm100a", "1" if fused else "0")
    xx = x.clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return {
        "out": y.detach().float(),
        "dx": xx.grad.detach().float(),
        **{name: p.grad.detach().float() for name, p in mod.named_parameters()},
    }


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
        got, base = (
            _rel(fused[name], reference[name]),
            _rel(triton[name], reference[name]),
        )
        assert got <= max(base * 1.5, 1e-6), (
            f"{name}: fused {got:.3e} vs triton {base:.3e}"
        )


@needs_b200
@pytest.mark.parametrize("shape", SHAPES)
def test_agrees_with_the_triton_path_to_bf16(shape, monkeypatch):
    module, x, dy = _build(shape)
    fused = _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    triton = _run(module, x, dy, fused=False, monkeypatch=monkeypatch)
    # since main e99ffb631 the unfused B200 path is the module's PyTorch statements (no Triton fallback on B200), which round the
    # bf16 output a little differently: out agrees to 4.17-4.22e-3 (2026-10-29); accuracy against fp32 is
    # test_is_no_less_accurate_than_the_triton_path. Its parameter gradients are bf16 (the module is bf16), one rounding of
    # 2^-8 relative each, so the parameter rows compare at 8e-3 (ln_in.weight 6.28e-3 on shape2).
    tolerances = {"out": 5e-3, "dx": 5e-3}
    for name in triton:
        assert _rel(fused[name], triton[name]) < tolerances.get(name, 8e-3), name


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
        assert _rel(fused[name], triton[name]) < {"out": 5e-3, "dx": 5e-3}.get(   # see test_agrees_with_the_triton_path_to_bf16
            name, 8e-3
        ), name


@needs_b200
def test_the_module_actually_dispatches_to_it(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    calls = []
    original = fused_sm100a.transition_fused_sm100a
    monkeypatch.setattr(
        fused_sm100a,
        "transition_fused_sm100a",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )
    module, x, dy = _build((1, 32, 32, 128))
    _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    assert calls, "the fused path was never entered"


@needs_b200
def test_inference_matches_training_forward():
    """No grad -> the forward skips the xn / statistics stores (runtime guard); the output bits must not change."""
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    module, x, _ = _build((1, 40, 48, 128))
    args = (
        module.ln_in.weight,
        module.ln_in.bias,
        module.expand_a.weight,
        module.expand_b.weight,
        module.squeeze.weight,
        module.ln_in.eps,
    )
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


@needs_b200
def test_weight_cast_matches_torch_bit_for_bit():
    """The one-launch fp32 -> bf16 cast of the three weights rounds exactly like ``Tensor.to(torch.bfloat16)``, special values
    included (ties to even, subnormals, signed zeros, inf, NaN)."""
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    torch.manual_seed(5)
    wa, wb = (torch.randn(512, 128, device="cuda") * 3 for _ in range(2))
    ws = torch.randn(128, 512, device="cuda") * 1e-3
    special = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), 1e-40, -1e-41, 3e38, 1.0 + 2**-8, 1.0 + 3 * 2**-8, 1.0 + 2**-9],
        device="cuda",
    )
    wa.view(-1)[: special.numel()] = special
    wb.view(-1)[-special.numel():] = special
    got = fused_sm100a._cast_launch(wa, wb, ws)
    for g, w in zip(got, (wa, wb, ws), strict=True):
        want = w.to(torch.bfloat16)
        assert g.dtype is torch.bfloat16 and g.shape == w.shape
        assert torch.equal(g.view(torch.int16), want.view(torch.int16))
    nan = wa.clone()
    nan.view(-1)[7] = float("nan")
    assert torch.isnan(fused_sm100a._cast_launch(nan, wb, ws)[0].view(-1)[7])


@needs_b200
def test_fp32_master_weights_equal_the_bf16_cast_run():
    """fp32 master parameters (the bf16-mixed training case) take the one-launch cast: the forward and dx are bit-identical to a
    run on the same weights cast to bf16 by torch, and the fp32 weight gradients round to the bf16 run's gradients."""
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    torch.manual_seed(3)
    d, h, m = 128, 512, 128 * 20
    x = torch.randn(m, d, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(m, d, device="cuda", dtype=torch.bfloat16)
    gamma = torch.rand(d, device="cuda") + 0.5
    beta = torch.randn(d, device="cuda") * 0.1
    w32 = [
        torch.randn(h, d, device="cuda") * d**-0.5,
        torch.randn(h, d, device="cuda") * d**-0.5,
        torch.randn(d, h, device="cuda") * h**-0.5,
    ]

    def run(ws_):
        xx = x.clone().requires_grad_()
        params = [w.clone().requires_grad_() for w in ws_]
        y = fused_sm100a.transition_fused_sm100a(xx, gamma, beta, *params, 1e-5)
        y.backward(dy)
        return y.detach(), xx.grad, [p.grad for p in params]

    y32, dx32, g32 = run(w32)
    y16, dx16, g16 = run([w.to(torch.bfloat16) for w in w32])
    assert torch.equal(y32, y16)
    assert torch.equal(dx32, dx16)
    for a, b in zip(g32, g16, strict=True):
        assert a.dtype is torch.float32 and b.dtype is torch.bfloat16
        assert torch.equal(a.to(torch.bfloat16), b)


@needs_b200
@pytest.mark.parametrize("scale", [4.0, 10.0])
@pytest.mark.parametrize(
    "case", [((1, 384, 384, 128), 4), ((1, 1024, 384, 128), 4), ((4, 128, 128, 64), 4)], ids=lambda c: f"{c[0]}-n{c[1]}"
)
def test_tanh_sigmoid_holds_the_fp64_bar_at_large_preactivations(case, scale, monkeypatch):
    """The D = 64 / 128 kernels evaluate the SwiGLU sigmoid as 0.5 tanh.approx(a / 2) + 0.5 (sm100.cuh ``sigmoid_tr``; forward and
    backward the same function). Its absolute error is largest where the gate saturates (|a| of several units: 0.5 + 0.5 t cancels for
    a << 0), so weights scaled by 4x / 10x push most pre-activations there; output and every gradient must still be no further from the
    fp64 module than 1.5x the bf16 PyTorch statements are. Measured 2026-10-30: 0.51-0.91x, equal to the exp2 / rcp sigmoid's to 3 digits."""
    shape, n = case
    torch.manual_seed(72)
    from miniworld_engine.modules import Transition

    module = Transition(shape[-1], n=n, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 2:
                param.normal_(std=scale * param.shape[1] ** -0.5)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    mod64 = copy.deepcopy(module).double()
    override_setting(monkeypatch, "transition_fused_sm100a", "0")
    xx = x.double().requires_grad_()
    y = mod64(xx)
    y.backward(dy.double())
    reference = {"out": y.detach(), "dx": xx.grad, **{k: p.grad for k, p in mod64.named_parameters()}}
    fused = _run(module, x, dy, fused=True, monkeypatch=monkeypatch)
    torch_ = _run(module, x, dy, fused=False, monkeypatch=monkeypatch)
    for name, want in reference.items():
        got = float((fused[name].double() - want).norm() / want.norm())
        base = float((torch_[name].double() - want).norm() / want.norm())
        assert got <= 1.5 * base, f"{name}: fused {got:.3e} vs bf16 statements {base:.3e}"

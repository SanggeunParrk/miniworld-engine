"""The generic sm_100a Transition (``kernels/transition/cuda/fused_generic_sm100a``): every call the tuned B200 builds do not take --
fp32 activations on the TF32 tensor cores, widths / expansions without a build, and any row count through ``ops.transition`` (FoldForge's
pairformer transitions: a single track [1, L, 384] and a pair [1, L, L, d] at an L off the 128-row tiles).

Pinned: the gate; that the module and ``ops.transition`` really launch these kernels where nothing else serves (and keep the tuned builds
where those serve); parity with an fp64 reference within 1.5x the error of the same-dtype PyTorch statements (output and every gradient;
fp32 against the PyTorch module on TF32); a CUDA-graph replay bitwise equal to the eager call; no Triton kernel."""

import pytest
import torch

from miniworld_engine import ops, settings
from miniworld_engine.kernels.transition.cuda import fused_generic_sm100a as g
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.transition.module import Transition

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100) required"),
]


@pytest.fixture(autouse=True)
def _tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    previous = settings.configure(force_portable_kernels=False, portable_on_hopper_blackwell=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev
    settings.configure(**{f: getattr(previous, f) for f in ("force_portable_kernels", "portable_on_hopper_blackwell")})


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _spy(monkeypatch):
    seen = []
    real = g.transition_generic_sm100a

    def wrapped(*a, **k):
        seen.append(tuple(a[0].shape))
        return real(*a, **k)

    monkeypatch.setattr(g, "transition_generic_sm100a", wrapped)
    return seen


def _modules(d, n, dtype):
    torch.manual_seed(0)
    ref = Transition(d, n, implementation=ImplementationType.PYTORCH).cuda()
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ours = Transition(d, n, implementation=ImplementationType.MINIWORLD).cuda()
    ours.load_state_dict(ref.state_dict())
    base = Transition(d, n, implementation=ImplementationType.PYTORCH).cuda()
    base.load_state_dict(ref.state_dict())
    r64 = Transition(d, n, implementation=ImplementationType.PYTORCH).cuda().double()
    r64.load_state_dict({k: v.double() for k, v in ref.state_dict().items()})
    if dtype is torch.float32:
        for m in (ours, base):
            m.float()
    return ours, base, r64


def test_gate():
    x = torch.randn(300, 192, device="cuda", dtype=torch.bfloat16)
    wa, ws = torch.randn(768, 192, device="cuda"), torch.randn(192, 768, device="cuda")
    assert g.supported(x, wa, ws) and g.supported(x.float(), wa, ws)
    assert g.supported(x, wa.bfloat16(), ws.bfloat16())
    assert not g.supported(x.half(), wa, ws)
    assert not g.supported(x[:, :160], wa[:, :160], ws[:160])                          # D not a multiple of 64
    assert not g.supported(x, wa[:704], ws[:, :704])                                    # hidden not a multiple of 128
    assert not g.supported(torch.randn(8, 1088, device="cuda", dtype=torch.bfloat16),
                           torch.randn(1088 * 2, 1088, device="cuda"), torch.randn(1088, 1088 * 2, device="cuda"))   # D > 1024
    assert not g.supported(x.cpu(), wa.cpu(), ws.cpu())
    torch.backends.cuda.matmul.allow_tf32 = False
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        assert not g.supported(x.float(), wa, ws)                                       # strict fp32 keeps the PyTorch reference
        assert g.supported(x, wa, ws)
    finally:
        torch.set_float32_matmul_precision(prev)


# (shape, d, n, dtype): fp32 at the tuned widths (no fp32 build there), bf16 at widths / expansions without a build, ragged rows
CASES = [
    ((1, 96, 96), 128, 4, torch.float32),          # fp32 pair (TF32), whole tiles
    ((1, 203, 203), 128, 4, torch.float32),        # fp32 pair, ragged rows
    ((1, 300), 384, 4, torch.float32),             # fp32 single track
    ((2, 77), 768, 2, torch.float32),              # fp32 diffusion single conditioning
    ((1, 300), 768, 4, torch.bfloat16),            # n = 4 at D = 768: no tuned build
    ((1, 40, 40), 192, 4, torch.bfloat16),         # a width without a build
    ((1, 500), 1024, 2, torch.bfloat16),           # widest
]


def _id(c):
    shape, d, n, dt = c
    return f"{'x'.join(map(str, shape))}-d{d}-n{n}-{'fp32' if dt is torch.float32 else 'bf16'}"


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
@pytest.mark.parametrize("train", [False, True], ids=["inference", "training"])
def test_module_runs_the_generic_kernels_and_matches_the_reference(monkeypatch, case, train):
    shape, d, n, dt = case
    ours, base, r64 = _modules(d, n, dt)
    seen = _spy(monkeypatch)
    torch.manual_seed(1)
    x = torch.randn(*shape, d, device="cuda")
    xo, xb, x64 = (x.to(dt).clone().requires_grad_(train), x.to(dt).clone().requires_grad_(train), x.double().requires_grad_(train))
    ac = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dt is torch.bfloat16)
    with torch.set_grad_enabled(train), torch.autocast(**ac):
        got = ours(xo)
        bl = base(xb)
    with torch.set_grad_enabled(train):
        want = r64(x64)
    assert seen, "the generic kernels did not run"
    assert got.dtype == dt and got.shape == x.shape
    e, e0 = _rel(got, want), _rel(bl, want)
    assert e <= 1.5 * e0 + 1e-6, (e, e0)
    if train:
        dy = torch.randn_like(want)
        got.backward(dy.to(got.dtype))
        bl.backward(dy.to(bl.dtype))
        want.backward(dy)
        assert _rel(xo.grad, x64.grad) <= 1.5 * _rel(xb.grad, x64.grad) + 1e-6
        for (name, p), (_, q), (_, r) in zip(ours.named_parameters(), base.named_parameters(), r64.named_parameters(), strict=True):
            assert _rel(p.grad, r.grad) <= 1.5 * _rel(q.grad, r.grad) + 1e-6, (name, _rel(p.grad, r.grad), _rel(q.grad, r.grad))


def _op_args(d, n, dt):
    torch.manual_seed(0)
    return dict(
        ln_in_weight=1 + 0.1 * torch.randn(d, device="cuda"), ln_in_bias=0.1 * torch.randn(d, device="cuda"),
        expand_a_weight=(torch.randn(n * d, d, device="cuda") / d ** 0.5).to(dt),
        expand_b_weight=(torch.randn(n * d, d, device="cuda") / d ** 0.5).to(dt),
        squeeze_weight=(torch.randn(d, n * d, device="cuda") / (n * d) ** 0.5).to(dt), n=n,
    )


def _op_ref(x, w):
    from miniworld_engine.kernels.transition.reference import transition_pytorch

    return transition_pytorch(x, w["ln_in_weight"].to(x.dtype), w["ln_in_bias"].to(x.dtype), w["expand_a_weight"].to(x.dtype),
                              w["expand_b_weight"].to(x.dtype), w["squeeze_weight"].to(x.dtype), w["n"], 1e-5)


# FoldForge's pairformer transitions through ops.transition (bf16 weights, bf16 activations, inference) at rows off the 128-row tiles:
# where a tuned build exists the rows are padded for it (faster than the generic kernels), every other width runs the generic ones
OP_CASES = [((1, 300, 384), 4, False), ((1, 300, 384), 2, False), ((1, 203, 203, 128), 4, False), ((1, 150, 150, 256), 2, False),
            ((1, 70, 70, 384), 4, False), ((1, 4, 300, 64), 4, False), ((1, 50, 50, 512), 4, False),
            ((1, 300, 768), 4, True), ((1, 77, 77, 192), 4, True), ((1, 300, 1024), 2, True)]


def _spy_tuned(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_sm100a, fused_wide_sm100a

    seen = []
    for mod, name in ((fused_sm100a, "transition_fused_sm100a"), (fused_wide_sm100a, "transition_wide_sm100a")):
        real = getattr(mod, name)

        def wrapped(*a, _real=real, **k):
            seen.append(a[0].numel() // a[0].shape[-1])
            return _real(*a, **k)

        monkeypatch.setattr(mod, name, wrapped)
    return seen


@pytest.mark.parametrize(("shape", "n", "generic"), OP_CASES, ids=[f"{'x'.join(map(str, s))}-n{n}" for s, n, _ in OP_CASES])
def test_ops_transition_ragged_rows(monkeypatch, shape, n, generic):
    d = shape[-1]
    w = _op_args(d, n, torch.bfloat16)
    seen, tuned = _spy(monkeypatch), _spy_tuned(monkeypatch)
    torch.manual_seed(1)
    x = torch.randn(*shape, device="cuda")
    with torch.no_grad():
        got = ops.transition(x.bfloat16(), **w)
        bl = _op_ref(x.bfloat16(), w)
        want = _op_ref(x.double(), w)
    assert got.shape == x.shape
    if generic:
        assert seen and not tuned, (seen, tuned)
    else:
        assert tuned and tuned[0] % 128 == 0 and not seen, (seen, tuned)   # the padded tuned build
    assert _rel(got, want) <= 1.5 * _rel(bl, want) + 1e-6, (_rel(got, want), _rel(bl, want))


def test_ops_transition_whole_tiles_keep_the_tuned_build(monkeypatch):
    w = _op_args(128, 4, torch.bfloat16)
    seen = _spy(monkeypatch)
    with torch.no_grad():
        ops.transition(torch.randn(1, 64, 64, 128, device="cuda", dtype=torch.bfloat16), **w)
    assert not seen


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_graph_replay_is_bitwise_the_eager_call_and_launches_no_triton(dt):
    from tests.cuda_graph_nodes import graph_kernels

    d, n = (384, 4) if dt is torch.float32 else (768, 4)
    w = _op_args(d, n, dt)
    x = torch.randn(1, 333, d, device="cuda", dtype=dt)
    with torch.no_grad():
        eager = ops.transition(x, **w)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(gr):
            out = ops.transition(x, **w)
        gr.replay()
        torch.cuda.synchronize()
    assert torch.equal(out, eager)
    names = set(graph_kernels(gr))
    assert any("tg_gemm_sm100" in k for k in names) and any("tg_ln_fwd" in k for k in names), names
    assert not any(k.startswith("_") and not k.startswith("_Z") and "_kernel" in k for k in names), names


def test_training_replay_is_bit_identical():
    d, n = 768, 4
    ours, _, _ = _modules(d, n, torch.bfloat16)
    x = torch.randn(1, 333, d, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)

    def step():
        xx = x.clone().requires_grad_(True)
        for p in ours.parameters():
            p.grad = None
        y = ours(xx)
        y.backward(dy)
        return [y.detach().clone(), xx.grad.clone()] + [p.grad.clone() for p in ours.parameters()]

    a, b = step(), step()
    assert all(torch.equal(u, v) for u, v in zip(a, b))

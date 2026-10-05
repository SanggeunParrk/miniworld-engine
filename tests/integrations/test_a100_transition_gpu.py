"""Transition and the bare SwiGLU FFN on A100 (kernels/transition/cuda/fused_wide_sm80.py and fused_sm80.py) against the fp32 PyTorch module: every (width, n) of the
registry, the pair / single / MSA activations, forward and backward, eager and compiled, CUDA graphs, the env switches and the gates.  Errors are held to the bf16
PyTorch module's own error in the same regime."""

import copy

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.modules import Transition
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


#: (d_hidden, n) of the registry's Transition rows (D = 128 / n = 4 is the fused_sm80 kernel when the rows are a multiple of 256, else the wide one)
SHAPES = [(64, 2), (64, 4), (128, 2), (128, 4), (256, 2), (256, 4), (384, 2), (384, 4), (768, 2)]
STREAMS = {"pair": lambda d: (1, 24, 24, d), "single": lambda d: (1, 130, d), "msa": lambda d: (1, 8, 40, d), "ragged": lambda d: (3, 77, d)}


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _modules(d, n, seed=0):
    """(fp32 PyTorch reference, the engine's MINIWORLD module in bf16, the bf16 PyTorch module): the same bf16-representable weights."""
    torch.manual_seed(seed)
    base = Transition(d, n=n, implementation=ImplementationType.PYTORCH)
    with torch.no_grad():
        for p in base.parameters():
            if p.ndim == 2:                       # the zero-initialised squeeze would make four of the six gradients exactly zero
                p.normal_(std=p.shape[1] ** -0.5)
            else:
                p.add_(torch.randn_like(p) * 0.1)
    ours = Transition(d, n=n, implementation=ImplementationType.MINIWORLD)
    ours.load_state_dict(base.state_dict())
    return copy.deepcopy(base).cuda().float(), ours.cuda().to(torch.bfloat16), base.cuda().to(torch.bfloat16)


def _run(mod, x, dy):
    xx = x.detach().clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return y.detach().float(), {"dx": xx.grad.float(), **{k: p.grad.float() for k, p in mod.named_parameters()}}


@pytest.mark.parametrize("stream", sorted(STREAMS))
@pytest.mark.parametrize(("d", "n"), SHAPES)
def test_transition_matches_fp32_forward_and_backward(d, n, stream):
    ref, ours, base = _modules(d, n)
    shape = STREAMS[stream](d)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(shape, device="cuda", generator=g)
    dy = torch.randn(shape, device="cuda", generator=g)
    yw, gw = _run(ref, x, dy)
    yg, gg = _run(ours, x.bfloat16(), dy.bfloat16())
    yb, gb = _run(base, x.bfloat16(), dy.bfloat16())
    assert _rel(yg, yw) <= 1.1 * _rel(yb, yw) + 1e-4, ("out", _rel(yg, yw), _rel(yb, yw))
    for name in gw:
        e, e0 = _rel(gg[name], gw[name]), _rel(gb[name], gw[name])
        assert e <= 1.1 * e0 + 1e-4, (name, e, e0)


@pytest.mark.parametrize(("d", "n"), SHAPES)
def test_transition_inference_matches_and_equals_the_training_forward(d, n):
    ref, ours, base = _modules(d, n)
    x = torch.randn(1, 20, 20, d, device="cuda")
    with torch.no_grad():
        want, got, b = ref(x), ours(x.bfloat16()), base(x.bfloat16())
        again = ours(x.bfloat16())
    assert got.dtype == torch.bfloat16
    assert got.shape == x.shape
    assert _rel(got, want) <= 1.1 * _rel(b, want) + 1e-4
    train = ours(x.bfloat16().requires_grad_())
    assert torch.equal(got, train.detach())                                      # the no-grad forward skips only the saves
    assert torch.equal(got, again)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("d", [64, 256, 384])
def test_frozen_transition_input_gradient(d, compiled):
    """Frozen parameters propagate dx; unfreezing restores parameter gradients."""
    _, ours, _ = _modules(d, 4)
    reference = Transition(d, n=4, implementation=ImplementationType.PYTORCH).cuda()
    reference.load_state_dict(ours.state_dict())
    ours.requires_grad_(False)
    reference.requires_grad_(False)
    x = torch.randn(2, 137, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    dy = torch.randn_like(x)
    fn = torch.compile(ours, fullgraph=True) if compiled else ours
    try:
        y, yr = fn(x), reference(xr)
        dx, = torch.autograd.grad(y, x, dy)
        dxr, = torch.autograd.grad(yr, xr, dy.float())
        assert _rel(y, yr) < .015
        assert _rel(dx, dxr) < .02
        assert all(p.grad is None for p in ours.parameters())
        ours.squeeze.weight.requires_grad_(True)
        fn(x).backward(dy)
        assert ours.squeeze.weight.grad is not None
        assert torch.isfinite(ours.squeeze.weight.grad).all()
    finally:
        torch._dynamo.reset()


def test_the_module_dispatches_to_the_cuda_paths(monkeypatch):
    """Which A100 path serves which call: the D128 / n4 whole-tile call the fused kernel, everything else the wide one; Triton only when switched off."""
    from miniworld_engine.kernels.transition.cuda import fused_sm80, fused_wide_sm80

    calls = []
    for mod, name in ((fused_sm80, "transition_fused_sm80"), (fused_wide_sm80, "transition_wide_sm80")):
        original = getattr(mod, name)
        monkeypatch.setattr(mod, name, lambda *a, _o=original, _n=name, **k: (calls.append(_n), _o(*a, **k))[1])
    with torch.no_grad():
        for (d, n), rows, expect in (((128, 4), 9216, "transition_fused_sm80"), ((128, 4), 1024, "transition_wide_sm80"), ((128, 4), 9000, "transition_wide_sm80"),
                                     ((256, 4), 1024, "transition_wide_sm80"), ((64, 2), 130, "transition_wide_sm80"), ((768, 2), 130, "transition_wide_sm80")):
            _, ours, _ = _modules(d, n)
            calls.clear()
            ours(torch.randn(rows, d, device="cuda", dtype=torch.bfloat16))
            assert calls == [expect], (d, n, rows, calls)


def test_the_switches_and_the_gates(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    wa = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
    ws = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1000, 256, device="cuda", dtype=torch.bfloat16)
    assert fw.supported(x, wa, ws)
    assert not fw.supported(x.float(), wa, ws)                                                                      # dtype
    assert not fw.supported(x.cpu(), wa.cpu(), ws.cpu())                                                            # device
    assert not fw.supported(torch.randn(10, 200, device="cuda", dtype=torch.bfloat16), wa[:, :200].contiguous(), ws[:200])  # width without a build
    assert not fw.supported(x, wa[:1000].contiguous(), ws[:, :1000].contiguous())                                      # hidden not a multiple of 64
    assert not fw.supported(x[:0], wa, ws)
    for name in ("MINIWORLD_TRANSITION_WIDE_SM80", "MINIWORLD_TRANSITION_FUSED_SM80"):
        monkeypatch.setenv(name, "0")
        assert not fw.supported(x, wa, ws)
        monkeypatch.delenv(name)
    assert fw.supported(x, wa, ws)


def test_with_the_switch_off_the_module_runs_triton_and_agrees(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    _, ours, _ = _modules(256, 4)
    x = torch.randn(1, 24, 24, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        got = ours(x)
        monkeypatch.setenv("MINIWORLD_TRANSITION_FUSED_SM80", "0")
        calls = []
        monkeypatch.setattr(fw, "transition_wide_sm80", lambda *a, **k: calls.append(1))
        want = ours(x)
    assert not calls
    assert _rel(got, want) < 6e-3


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize(("d", "n"), [(64, 2), (256, 4), (768, 2)])
def test_compiled_module_matches_eager(d, n, train):
    """The extension lookup stays out of the traced region; the compiled call is bit-identical to eager (the ops are opaque)."""
    torch._dynamo.reset()
    _, ours, _ = _modules(d, n)
    x = torch.randn(1, 130, d, device="cuda", dtype=torch.bfloat16)
    comp = torch.compile(ours, fullgraph=True)
    try:
        if not train:
            with torch.no_grad():
                assert torch.equal(comp(x), ours(x))
            return
        dy = torch.randn_like(x)
        ye, ge = _run(ours, x, dy)
        for p in ours.parameters():
            p.grad = None
        xc = x.clone().requires_grad_()
        yc = comp(xc)
        yc.backward(dy)
        assert torch.equal(yc.detach().float(), ye)
        assert torch.equal(xc.grad.float(), ge["dx"])
        for k, p in ours.named_parameters():
            assert torch.equal(p.grad.float(), ge[k]), k
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize(("d", "n"), [(64, 2), (384, 4)])
def test_cuda_graph_capture_and_replay(d, n):
    """Forward and a whole training step capture (no host syncs, the extension and the cuBLAS handles are resolved before) and replay bit-identically."""
    _, ours, _ = _modules(d, n)
    x = torch.randn(1, 130, d, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    xx = x.clone().requires_grad_()

    def step():
        for t in (xx, *ours.parameters()):
            t.grad = None
        y = ours(xx)
        y.backward(dy)
        return y, [xx.grad, *[p.grad for p in ours.parameters()]]

    stream = torch.cuda.Stream()                                      # eager steps and the capture share one stream (the autograd nodes remember it)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
        want, wg = step()
        want, wg = want.clone(), [g.clone() for g in wg]
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out, grads = step()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, want)
    for g, w in zip(grads, wg, strict=True):
        assert torch.equal(g, w)


# ---------------------------------------------------------------------------------------------------------------------------- the bare SwiGLU FFN
FFN_SHAPES = [(128, 256), (384, 768), (64, 128), (256, 1024)]


def _ffn_ref(x, wa, wb, ws):
    return F.linear(F.silu(F.linear(x, wa)) * F.linear(x, wb), ws)


def test_swiglu_ffn_frozen_input_gradient():
    from miniworld_engine import ops

    torch.manual_seed(23)
    x = torch.randn(137, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    wa, wb = (torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16) / 16 for _ in range(2))
    ws = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16) / 32
    xr = x.detach().float().requires_grad_()
    dy = torch.randn_like(x)
    y = ops.swiglu_ffn(x, wa, wb, ws)
    reference = _ffn_ref(xr, wa.float(), wb.float(), ws.float())
    dx, = torch.autograd.grad(y, x, dy)
    dxr, = torch.autograd.grad(reference, xr, dy.float())
    assert _rel(y, reference) < .015
    assert _rel(dx, dxr) < .02


@pytest.mark.parametrize("shape", [(5, 77), (1, 20, 20), (3, 130)])
@pytest.mark.parametrize(("d", "h"), FFN_SHAPES)
def test_swiglu_ffn_matches_fp32_forward_and_backward(d, h, shape):
    from miniworld_engine import ops

    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(*shape, d, device="cuda", generator=g)
    dy = torch.randn_like(x)
    wa, wb = (torch.randn(h, d, device="cuda", generator=g) * d**-0.5 for _ in range(2))
    ws = torch.randn(d, h, device="cuda", generator=g) * h**-0.5

    def grads(dtype, fn):
        leaves = [t.clone().to(dtype).requires_grad_() for t in (x, wa, wb, ws)]
        y = fn(*leaves)
        y.backward(dy.to(dtype))
        return [y.detach().float(), *[t.grad.float() for t in leaves]]

    want = grads(torch.float32, _ffn_ref)
    base = grads(torch.bfloat16, _ffn_ref)
    got = grads(torch.bfloat16, ops.swiglu_ffn)
    for e, e0 in zip((_rel(a, w) for a, w in zip(got, want, strict=True)), (_rel(a, w) for a, w in zip(base, want, strict=True)), strict=True):
        assert e <= 1.1 * e0 + 1e-4, (e, e0)


def test_swiglu_ffn_dispatch_switch_and_fallback(monkeypatch):
    from miniworld_engine import ops
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    calls = []
    original = fw.swiglu_ffn_sm80
    monkeypatch.setattr(fw, "swiglu_ffn_sm80", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    wa, wb = (torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2))
    ws = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    x = torch.randn(1, 300, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        got = ops.swiglu_ffn(x, wa, wb, ws)
        assert calls == [1]
        monkeypatch.setenv("MINIWORLD_TRANSITION_FUSED_SM80", "0")
        want = ops.swiglu_ffn(x, wa, wb, ws)
        assert calls == [1]                                                                                     # the switch hands the call to Triton
        assert _rel(got, want) < 6e-3
        monkeypatch.delenv("MINIWORLD_TRANSITION_FUSED_SM80")
        assert _rel(ops.swiglu_ffn(x.float(), wa.float(), wb.float(), ws.float()), want.float()) < 1e-2      # fp32 operands: not this path
    assert calls == [1]


def test_swiglu_ffn_compiled_and_graph_replay():
    from miniworld_engine import ops

    wa, wb = (torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2))
    ws = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    x = torch.randn(2, 300, 128, device="cuda", dtype=torch.bfloat16)
    torch._dynamo.reset()
    with torch.no_grad():
        want = ops.swiglu_ffn(x, wa, wb, ws)
        assert torch.equal(torch.compile(lambda a: ops.swiglu_ffn(a, wa, wb, ws), fullgraph=True)(x), want)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ops.swiglu_ffn(x, wa, wb, ws)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = ops.swiglu_ffn(x, wa, wb, ws)
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(out, want)
    torch._dynamo.reset()

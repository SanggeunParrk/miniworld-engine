"""H100 TriMul with a hidden width twice the pair width, one direction, at D64 / D128 (``TriangleMultiplication(D, d_hidden=2 D)``:
AF3 / Protenix template blocks, pair 64, hidden 128) on the bidirectional D64 / D128 kernels.

Per token it is the bidirectional block (4 D planes out of K1, LN_out over 2 D channels, the same backward); only the cuBLAS contraction
between the kernels pairs the planes in the module's one direction over all 2 D channels (``trimul_h100._hidden_twice``). Every output
and gradient is checked against the FP32 PyTorch reference module, the native entry is spied on (so a PyTorch fallback cannot pass),
and the modules are driven as MiniWorld's template stack drives them (FP32 parameters, a bf16 pair, a token mask, row dropout)."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import trimul_h100
from miniworld_engine.kernels.trimul_inproj.cuda import (
    h100_d64_b200gate,
    h100_inference,
    h100_training,
)
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def hopper():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 (sm_90) required")
    if torch.cuda.get_device_properties(0).multi_processor_count != 132:
        pytest.skip("full H100 required (the D128 cooperative grids)")
    old = settings.configure(engine_backend="auto")
    torch.compiler.reset()
    yield
    settings.configure(**vars(old))


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


def make(width, outgoing, implementation=ImplementationType.MINIWORLD, seed=0):
    torch.manual_seed(seed)
    m = TriangleMultiplication(width, d_hidden=2 * width, outgoing=outgoing, implementation=implementation, p_drop=0.25).cuda()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def reference_of(m):
    """The same module, implementation=PYTORCH, FP32 parameters (the reference the CUDA path replaces)."""
    ref = TriangleMultiplication(m.ln_pair.weight.shape[0], d_hidden=m.d_hidden, outgoing=m.outgoing,
                                 implementation=ImplementationType.PYTORCH, p_drop=m.p_drop).cuda()
    ref.load_state_dict(m.state_dict())
    return ref.train(m.training)


def inputs(B, L, D, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, generator=g)
    mask = torch.rand(B, L, device="cuda", generator=g) > 0.15                                    # a different mask per sample
    ds = (torch.rand(B, 1, L, D, device="cuda", generator=g) > 0.25).to(torch.bfloat16) / 0.75    # a different dropout scale per sample
    dy = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, mask, ds, dy


def spy(monkeypatch, module, name):
    """Record (samples, direction) of every call of ``module.name``."""
    calls = []
    fn = getattr(module, name)

    def wrapped(*a, **k):
        calls.append((a[0].shape[0], k.get("direction", 0)))
        return fn(*a, **k)

    monkeypatch.setattr(module, name, wrapped)
    return calls


def training_entry(monkeypatch, width):
    return spy(monkeypatch, h100_d64_b200gate if width == 64 else h100_training, "bidirectional_trimul")


DIRECTIONS = [True, False]
DIR_IDS = ["out", "in"]


# ---------------------------------------------------------------------------------------------------------------------- inference
INFER = [(D, B, L) for D in (64, 128) for B, L in ((1, 128), (1, 256), (1, 384), (4, 128), (4, 256), (4, 384), (1, 200), (2, 200))]


@pytest.mark.parametrize("outgoing", DIRECTIONS, ids=DIR_IDS)
@pytest.mark.parametrize("case", INFER, ids=[f"D{D}-B{B}-L{L}" for D, B, L in INFER])
def test_inference_matches_fp32_reference(case, outgoing, monkeypatch):
    D, B, L = case
    m = make(D, outgoing).eval()
    ref = reference_of(m)
    x, mask, _, _ = inputs(B, L, D)
    calls = spy(monkeypatch, h100_inference, "inference")
    with torch.no_grad():
        assert trimul_h100.serves_inference(m, x, bidirectional=False)
        got = m(x, mask)
        assert [c[0] for c in calls] == [B], calls              # one native K1 / contraction / K3 call, not the PyTorch statements
        want = ref(x.float(), mask)
        assert torch.isfinite(got.float()).all()
        err = relative(got, want)
        assert err < 6e-3, err
        err0 = relative(m(x), ref(x.float()))                   # no mask
        assert err0 < 6e-3, err0
        assert torch.equal(got, m(x, mask))                     # bitwise run to run


# ---------------------------------------------------------------------------------------------------------------------- training
TRAIN = [(D, B, L) for D in (64, 128) for B, L in ((1, 128), (1, 256), (1, 384), (4, 128), (4, 256), (4, 384))]


def _train(m, x, mask, ds, dy, monkeypatch):
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    xx = x.detach().clone().requires_grad_()
    params = list(m.parameters())
    y = m(xx, mask)
    grads = torch.autograd.grad(y, [xx, *params], dy)
    return y.detach(), [g.detach().clone() for g in grads]


@pytest.mark.parametrize("outgoing", DIRECTIONS, ids=DIR_IDS)
@pytest.mark.parametrize("case", TRAIN, ids=[f"D{D}-B{B}-L{L}" for D, B, L in TRAIN])
def test_training_matches_fp32_reference(case, outgoing, monkeypatch):
    """bf16 pair, FP32 master parameters, a token mask, row dropout: y, dx and every parameter gradient against the FP32 reference."""
    D, B, L = case
    m = make(D, outgoing).train()
    ref = reference_of(m)
    x, mask, ds, dy = inputs(B, L, D)
    calls = training_entry(monkeypatch, D)
    assert trimul_h100.serves_single(m, x)
    y, grads = _train(m, x, mask, ds, dy, monkeypatch)
    assert calls == [(B, 1 if outgoing else 2)], calls            # one native call for the batch, in the module's direction
    yr, gr = _train(ref, x.float(), mask, ds.float(), dy.float(), monkeypatch)
    names = ["y", "x", *(n for n, _ in m.named_parameters())]
    errors = {n: relative(a, b) for n, a, b in zip(names, (y, *grads), (yr, *gr), strict=True)}
    assert errors["y"] < 6e-3, errors
    assert max(v for n, v in errors.items() if n != "y") < 2e-2, errors
    assert all(g.dtype == p.dtype and g.shape == p.shape for g, p in zip(grads[1:], m.parameters(), strict=True))
    y2, g2 = _train(m, x, mask, ds, dy, monkeypatch)              # bitwise run to run
    assert torch.equal(y, y2)
    assert all(torch.equal(a, b) for a, b in zip(grads, g2, strict=True))


@pytest.mark.parametrize("width", [64, 128])
def test_training_without_mask(width, monkeypatch):
    m = make(width, True).train()
    ref = reference_of(m)
    x, _, ds, dy = inputs(1, 256, width)
    calls = training_entry(monkeypatch, width)
    y, grads = _train(m, x, None, ds, dy, monkeypatch)
    assert calls == [(1, 1)], calls
    yr, gr = _train(ref, x.float(), None, ds.float(), dy.float(), monkeypatch)
    errors = [relative(a, b) for a, b in zip((y, *grads), (yr, *gr), strict=True)]
    assert errors[0] < 6e-3, errors
    assert max(errors[1:]) < 2e-2, errors


@pytest.mark.parametrize("width", [64, 128])
def test_dropout_draw_nograd_equals_training(width):
    """The module's own row-dropout draw (D64: the native Philox scale): a no-grad training-mode forward (the forward_nograd launch) equals
    the training forward for one draw."""
    m = make(width, True).train()
    x, mask, _, _ = inputs(2, 256, width)
    torch.manual_seed(5)
    with torch.no_grad():
        a = m(x, mask)
    torch.manual_seed(5)
    b = m(x.clone().requires_grad_(), mask)
    assert torch.equal(a, b.detach())
    with torch.no_grad():
        assert not torch.equal(a, m.eval()(x, mask))              # the draw did drop rows


# ---------------------------------------------------------------------------------------------------------------------- CUDA graphs
GRAPH = [(64, 4, True), (64, 4, False), (128, 1, True), (128, 2, False)]


@pytest.mark.parametrize("case", GRAPH, ids=[f"D{D}-B{B}-{'out' if o else 'in'}" for D, B, o in GRAPH])
def test_training_graph_replay(case, monkeypatch):
    """A captured forward + backward replays with live inputs, masks, dropout scales and weights."""
    D, B, outgoing = case
    L = 256
    m = make(D, outgoing).train()
    x, mask, ds, dy = inputs(B, L, D)
    x = x.requires_grad_()
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    params = list(m.parameters())
    calls = training_entry(monkeypatch, D)

    def step():
        y = m(x, mask)
        return y, torch.autograd.grad(y, [x, *params], dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y, grads = step()
    assert calls, calls
    assert all(c == (B, 1 if outgoing else 2) for c in calls), calls
    with torch.no_grad():
        x.normal_()
        params[0].mul_(0.95)
        mask.copy_(torch.rand_like(mask, dtype=torch.float) > 0.3)
        ds.copy_((torch.rand_like(ds, dtype=torch.float) > 0.25).to(ds.dtype) / 0.75)
        dy.normal_()
    graph.replay()
    torch.cuda.synchronize()
    want_y, want_g = step()
    assert torch.equal(y, want_y)
    assert all(torch.equal(a, b) for a, b in zip(grads, want_g, strict=True))


@pytest.mark.parametrize("case", GRAPH, ids=[f"D{D}-B{B}-{'out' if o else 'in'}" for D, B, o in GRAPH])
def test_inference_graph_replay(case):
    D, B, outgoing = case
    m = make(D, outgoing).eval()
    x, mask, _, _ = inputs(B, 256, D)
    with torch.no_grad():
        m(x, mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = m(x, mask)
        x.normal_()
        mask.copy_(torch.rand_like(mask, dtype=torch.float) > 0.3)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, m(x, mask))


# ---------------------------------------------------------------------------------------------------------------------- contract
def test_chunk_plan_is_the_bidirectional_one():
    """B samples beyond one launch are chunked with the bidirectional working set (the same planes per token)."""
    for width in (64, 128):
        one = TriangleMultiplication(width, d_hidden=2 * width, implementation=ImplementationType.MINIWORLD).cuda()
        bidir = BidirectionalTriangleMultiplication(width, implementation=ImplementationType.MINIWORLD).cuda()
        for length in (512, 1024, 2048):
            x = torch.empty(1, 1, 1, width, device="cuda", dtype=torch.bfloat16).expand(48, length, length, width)
            for inference in (False, True):
                assert trimul_h100._plan(one, x, inference) == trimul_h100._plan(bidir, x, inference)


def test_other_hidden_widths_not_served():
    """Other ratios, hidden 2 D at D256, and the bidirectional module with hidden != D keep the module's other paths."""
    x64 = torch.zeros(1, 128, 128, 64, device="cuda", dtype=torch.bfloat16)
    x256 = torch.zeros(1, 128, 128, 256, device="cuda", dtype=torch.bfloat16)
    odd = TriangleMultiplication(64, d_hidden=96, implementation=ImplementationType.MINIWORLD).cuda()
    wide = TriangleMultiplication(256, d_hidden=512, implementation=ImplementationType.MINIWORLD).cuda()
    bidir = BidirectionalTriangleMultiplication(64, d_hidden=128, implementation=ImplementationType.MINIWORLD).cuda()
    assert not trimul_h100.serves_single(odd, x64)
    assert not trimul_h100.serves_single(wide, x256)
    assert not trimul_h100.serves(bidir, x64)
    with torch.no_grad():
        assert not trimul_h100.serves_inference(odd, x64, bidirectional=False)
        assert not trimul_h100.serves_inference(wide, x256, bidirectional=False)
        assert not trimul_h100.serves_inference(bidir, x64, bidirectional=True)

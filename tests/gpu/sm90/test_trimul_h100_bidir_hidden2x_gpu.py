"""H100 bidirectional TriMul with a hidden width twice the pair width in each direction at D64
(``BidirectionalTriangleMultiplication(64, d_hidden=128)``: 8 D planes out of K1, the output LayerNorm over 256 channels).

Inference: the packaged K1 / K3 unit (c_z 64, 256 channels) with the two-direction contraction.  Training: ``h100_d64_h2`` (the D64 front and
K3 at 256 channels, the backward ``d64_h2_bwd.cu`` with the 256 channels in two halves).  Every output and gradient is checked against the
FP32 PyTorch reference module, the native entry is spied on (a PyTorch fallback cannot pass), and the modules are driven as MiniWorld's
template stack drives them (FP32 parameters, a bf16 pair, a token mask, row dropout)."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import trimul_h100
from miniworld_engine.kernels.trimul_inproj.cuda import h100_d64_h2, h100_inference
from miniworld_engine.modules.exceptions import ImplementationType
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
    old = settings.configure(force_portable_kernels=False)
    torch.compiler.reset()
    yield
    settings.configure(**vars(old))


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


def make(implementation=ImplementationType.MINIWORLD, seed=0, hidden=128, dtype=torch.float32):
    torch.manual_seed(seed)
    m = BidirectionalTriangleMultiplication(64, d_hidden=hidden, implementation=implementation, p_drop=0.25).cuda()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m.to(dtype)


def reference_of(m):
    """The same module, implementation=PYTORCH, FP32 parameters (the reference the CUDA path replaces)."""
    ref = BidirectionalTriangleMultiplication(64, d_hidden=m.d_hidden, implementation=ImplementationType.PYTORCH, p_drop=m.p_drop).cuda()
    ref.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
    return ref.train(m.training)


def inputs(B, L, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, L, L, 64, device="cuda", dtype=torch.bfloat16, generator=g)
    mask = torch.rand(B, L, device="cuda", generator=g) > 0.15                                    # a different mask per sample
    ds = (torch.rand(B, 1, L, 64, device="cuda", generator=g) > 0.25).to(torch.bfloat16) / 0.75   # a different dropout scale per sample
    dy = torch.randn(B, L, L, 64, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, mask, ds, dy


def spy(monkeypatch, module, name):
    """Record the sample count of every call of ``module.name``."""
    calls = []
    fn = getattr(module, name)

    def wrapped(*a, **k):
        calls.append(a[0].shape[0])
        return fn(*a, **k)

    monkeypatch.setattr(module, name, wrapped)
    return calls


# ---------------------------------------------------------------------------------------------------------------------- inference
INFER = [(B, L) for B, L in ((1, 128), (1, 256), (1, 384), (4, 128), (4, 256), (1, 200), (2, 200))]


@pytest.mark.parametrize("case", INFER, ids=[f"B{B}-L{L}" for B, L in INFER])
def test_inference_matches_fp32_reference(case, monkeypatch):
    B, L = case
    m = make().eval()
    ref = reference_of(m)
    x, mask, _, _ = inputs(B, L)
    calls = spy(monkeypatch, h100_inference, "inference")
    with torch.no_grad():
        assert trimul_h100.serves_inference(m, x, bidirectional=True)
        got = m(x, mask)
        assert calls == [B], calls              # one native K1 / contraction / K3 call, not the PyTorch statements
        want = ref(x.float(), mask)
        assert torch.isfinite(got.float()).all()
        err = relative(got, want)
        assert err < 6e-3, err
        err0 = relative(m(x), ref(x.float()))   # no mask
        assert err0 < 6e-3, err0
        assert torch.equal(got, m(x, mask))     # bitwise run to run


# ---------------------------------------------------------------------------------------------------------------------- training
TRAIN = [(B, L) for B, L in ((1, 128), (1, 256), (1, 384), (2, 128), (3, 256))]


def _train(m, x, mask, ds, dy, monkeypatch):
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    xx = x.detach().clone().requires_grad_()
    params = list(m.parameters())
    y = m(xx, mask)
    grads = torch.autograd.grad(y, [xx, *params], dy)
    return y.detach(), [g.detach().clone() for g in grads]


@pytest.mark.parametrize("case", TRAIN, ids=[f"B{B}-L{L}" for B, L in TRAIN])
def test_training_matches_fp32_reference(case, monkeypatch):
    """bf16 pair, FP32 master parameters, a token mask, row dropout: y, dx and every parameter gradient against the FP32 reference."""
    B, L = case
    m = make().train()
    ref = reference_of(m)
    x, mask, ds, dy = inputs(B, L)
    calls = spy(monkeypatch, h100_d64_h2, "bidirectional_trimul")
    assert trimul_h100.serves(m, x)
    y, grads = _train(m, x, mask, ds, dy, monkeypatch)
    assert calls == [1] * B, calls                  # one native call per sample (``_plan``: chunk 1)
    yr, gr = _train(ref, x.float(), mask, ds.float(), dy.float(), monkeypatch)
    names = ["y", "x", *(n for n, _ in m.named_parameters())]
    errors = {n: relative(a, b) for n, a, b in zip(names, (y, *grads), (yr, *gr), strict=True)}
    assert errors["y"] < 6e-3, errors
    assert max(v for n, v in errors.items() if n != "y") < 2e-2, errors
    assert all(g.dtype == p.dtype and g.shape == p.shape for g, p in zip(grads[1:], m.parameters(), strict=True))
    y2, g2 = _train(m, x, mask, ds, dy, monkeypatch)    # bitwise run to run
    assert torch.equal(y, y2)
    assert all(torch.equal(a, b) for a, b in zip(grads, g2, strict=True))


def test_training_without_mask_bf16_parameters(monkeypatch):
    m = make(dtype=torch.bfloat16).train()
    ref = reference_of(m)
    x, _, ds, dy = inputs(1, 256)
    calls = spy(monkeypatch, h100_d64_h2, "bidirectional_trimul")
    y, grads = _train(m, x, None, ds, dy, monkeypatch)
    assert calls == [1], calls
    yr, gr = _train(ref, x.float(), None, ds.float(), dy.float(), monkeypatch)
    errors = [relative(a, b) for a, b in zip((y, *grads), (yr, *gr), strict=True)]
    assert errors[0] < 6e-3, errors
    assert max(errors[1:]) < 3e-2, errors
    assert all(g.dtype == p.dtype and g.shape == p.shape for g, p in zip(grads[1:], m.parameters(), strict=True))


def test_dropout_draw_nograd_equals_training():
    """The module's own row-dropout draw (the native Philox scale): a no-grad training-mode forward (the forward_nograd launch) equals the
    training forward for one draw."""
    m = make().train()
    x, mask, _, _ = inputs(2, 256)
    torch.manual_seed(5)
    with torch.no_grad():
        a = m(x, mask)
    torch.manual_seed(5)
    b = m(x.clone().requires_grad_(), mask)
    assert torch.equal(a, b.detach())
    with torch.no_grad():
        assert not torch.equal(a, m.eval()(x, mask))    # the draw did drop rows


def test_checkpointed_chunks_equal_plain(monkeypatch):
    """B samples whose saved state does not fit run checkpointed (each chunk's forward again in its backward): the same y and gradients."""
    m = make().train()
    x, mask, ds, dy = inputs(3, 128)
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    y0, g0 = _train(m, x, mask, ds, dy, monkeypatch)
    monkeypatch.setattr(trimul_h100, "_plan", lambda module, pair, inference: (1, True) if pair.shape[0] > 1 else (1, False))
    y1, g1 = _train(m, x, mask, ds, dy, monkeypatch)
    assert torch.equal(y0, y1)
    assert all(torch.equal(a, b) for a, b in zip(g0, g1, strict=True))


# ---------------------------------------------------------------------------------------------------------------------- CUDA graphs
def test_training_graph_replay(monkeypatch):
    """A captured forward + backward replays with live inputs, masks, dropout scales and weights."""
    B, L = 2, 256
    m = make().train()
    x, mask, ds, dy = inputs(B, L)
    x = x.requires_grad_()
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    params = list(m.parameters())
    calls = spy(monkeypatch, h100_d64_h2, "bidirectional_trimul")

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


def test_inference_graph_replay():
    m = make().eval()
    x, mask, _, _ = inputs(2, 256)
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
def test_served_contract():
    m = make()
    x = torch.zeros(1, 128, 128, 64, device="cuda", dtype=torch.bfloat16)
    assert trimul_h100._hidden_twice_bidirectional(m, x)
    assert trimul_h100.serves(m, x)
    with torch.no_grad():
        assert trimul_h100.serves_inference(m, x, bidirectional=True)
    # a ragged training length and another hidden width keep the module's other paths
    assert not trimul_h100.serves(m, torch.zeros(1, 130, 130, 64, device="cuda", dtype=torch.bfloat16))
    odd = BidirectionalTriangleMultiplication(64, d_hidden=96, implementation=ImplementationType.MINIWORLD).cuda()
    assert not trimul_h100.serves(odd, x)

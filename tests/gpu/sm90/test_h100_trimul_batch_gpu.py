"""H100 TriMul D64, native batch: B samples in one call (tokens b-major, one launch per stage) against B calls of one sample.

The kernels are the same per token; what the batch changes is the slow token axis of K1 / K3 (rows = B L), the sample index of the token
mask (K1, the FP32 pair mask of the training preparation), the per-sample row-dropout scale (K3 and B1 read row s L + j) and cuBLAS's
batch dim (channels x B planes). Each sample draws its own dropout scale and mask, so an index mix-up between samples shows as a large
error. The modules are driven exactly as MiniWorld's template stack drives them (FP32 parameters, a bf16 pair, a token mask)."""

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.integrations import trimul_h100
from miniworld_engine.kernels.trimul_inproj.cuda import (
    h100_d64_b200gate,
    h100_inference,
    h100_uni_d64_b200gate,
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

D = 64


@pytest.fixture(autouse=True)
def hopper():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 (sm_90) required")
    torch.compiler.reset()


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


def make(direction, seed=0):
    torch.manual_seed(seed)
    m = (BidirectionalTriangleMultiplication(D, implementation=ImplementationType.MINIWORLD, p_drop=0.25) if direction == 0 else
         TriangleMultiplication(D, outgoing=direction == 1, implementation=ImplementationType.MINIWORLD, p_drop=0.25)).cuda()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def inputs(B, L, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, generator=g)
    mask = torch.rand(B, L, device="cuda", generator=g) > 0.15                                    # a different mask per sample
    ds = (torch.rand(B, 1, L, D, device="cuda", generator=g) > 0.25).to(torch.bfloat16) / 0.75    # a different dropout scale per sample
    dy = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, mask, ds, dy


def reference(m, direction, x, mask, ds):
    """FP32 statement of the module (the PyTorch reference of the residual update)."""
    p = dict(m.named_parameters())
    xn = F.layer_norm(x.float(), (D,), p["ln_pair.weight"], p["ln_pair.bias"], 1e-5)
    pm = (mask[:, :, None] & mask[:, None, :])[..., None].float()
    left = torch.sigmoid(F.linear(xn, p["to_left_gate.weight"])) * F.linear(xn, p["to_left.weight"]) * pm
    right = torch.sigmoid(F.linear(xn, p["to_right_gate.weight"])) * F.linear(xn, p["to_right.weight"]) * pm
    if direction == 0:
        tri = torch.cat((torch.einsum("bikd,bjkd->bijd", left[..., :D], right[..., :D]),
                         torch.einsum("bkid,bkjd->bijd", left[..., D:], right[..., D:])), -1)
    elif direction == 1:
        tri = torch.einsum("bikd,bjkd->bijd", left, right)
    else:
        tri = torch.einsum("bkid,bkjd->bijd", left, right)
    out = F.layer_norm(tri, (tri.shape[-1],), p["ln_out.weight"], p["ln_out.bias"], 1e-5)
    upd = F.linear(out, p["to_out.weight"]) * torch.sigmoid(F.linear(xn, p["to_gate.weight"]))
    return x.float() + (upd if ds is None else upd * ds.float())


def counting(monkeypatch, module, name):
    calls = []
    fn = getattr(module, name)

    def wrapped(*a, **k):
        calls.append(a[0].shape[0] if hasattr(a[0], "shape") else a[1].shape[0])
        return fn(*a, **k)

    monkeypatch.setattr(module, name, wrapped)
    return calls


def native_entry(monkeypatch, direction, training):
    if not training:
        return counting(monkeypatch, h100_inference, "inference")
    if direction == 0:
        return counting(monkeypatch, h100_d64_b200gate, "bidirectional_trimul")
    return counting(monkeypatch, h100_uni_d64_b200gate, "single_trimul")


CASES = [(L, B, direction) for (L, B) in ((128, 2), (128, 8), (256, 4), (384, 4), (768, 2)) for direction in (0, 1, 2)]
IDS = [f"L{L}-B{B}-dir{d}" for L, B, d in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_inference_batch_matches_samples(case, monkeypatch):
    L, B, direction = case
    m = make(direction).eval()
    x, mask, _, _ = inputs(B, L)
    calls = native_entry(monkeypatch, direction, False)
    with torch.no_grad():
        assert trimul_h100.serves_inference(m, x, bidirectional=direction == 0)
        got = m(x, mask)
        assert calls == [B], calls                                   # one native call for the batch, not the PyTorch statements
        want = torch.cat([m(x[i:i + 1], mask[i:i + 1]) for i in range(B)], 0)
        assert torch.isfinite(got.float()).all()
        assert relative(got, want) < 1e-3, relative(got, want)       # same per-token kernels; only cuBLAS's batch count differs
        assert relative(got, reference(m, direction, x, mask, None)) < 1e-2
        got0 = m(x, None)                                            # no mask
        want0 = torch.cat([m(x[i:i + 1], None) for i in range(B)], 0)
        assert relative(got0, want0) < 1e-3
        again = m(x, mask)
        assert torch.equal(got, again)                               # bitwise run to run


def _train(m, x, mask, ds, dy, monkeypatch):
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    xx = x.detach().clone().requires_grad_()
    params = list(m.parameters())
    y = m(xx, mask)
    grads = torch.autograd.grad(y, [xx, *params], dy)
    return y.detach(), [g.detach().clone() for g in grads]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_training_batch_matches_samples(case, monkeypatch):
    L, B, direction = case
    m = make(direction).train()
    x, mask, ds, dy = inputs(B, L)
    calls = native_entry(monkeypatch, direction, True)
    assert trimul_h100.serves(m, x)
    y_b, g_b = _train(m, x, mask, ds, dy, monkeypatch)
    assert calls == [B], calls
    ys, gs = [], []
    for i in range(B):
        yi, gi = _train(m, x[i:i + 1], mask[i:i + 1], ds[i:i + 1], dy[i:i + 1], monkeypatch)
        ys.append(yi)
        gs.append(gi)
    y_s = torch.cat(ys, 0)
    assert relative(y_b, y_s) < 1e-3, ("y", relative(y_b, y_s))
    dx_s = torch.cat([g[0] for g in gs], 0)
    assert relative(g_b[0], dx_s) < 2e-3, ("dx", relative(g_b[0], dx_s))
    for k in range(1, len(g_b)):                                     # parameter gradients: the batch sums over samples
        want = sum(g[k].float() for g in gs)
        assert relative(g_b[k], want) < 2e-3, (k, relative(g_b[k], want))
    # the FP32 PyTorch reference, every gradient
    xr = x.detach().clone().requires_grad_()
    params = list(m.parameters())
    z = reference(m, direction, xr, mask, ds)
    h = torch.autograd.grad(z, [xr, *params], dy.float())
    errors = [relative(a, b) for a, b in zip((y_b, *g_b), (z, *h), strict=True)]
    assert errors[0] < 5e-3, errors
    assert max(errors[1:]) < 2e-2, errors
    # bitwise run to run
    y2, g2 = _train(m, x, mask, ds, dy, monkeypatch)
    assert torch.equal(y_b, y2)
    assert all(torch.equal(a, b) for a, b in zip(g_b, g2, strict=True))


@pytest.mark.parametrize("direction", [0, 1, 2])
def test_training_batch_graph_replay(direction, monkeypatch):
    """A captured batched forward + backward replays with live inputs, masks, dropout scales and weights."""
    L, B = 256, 4
    m = make(direction).train()
    x, mask, ds, dy = inputs(B, L)
    x = x.requires_grad_()
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    params = list(m.parameters())

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


@pytest.mark.parametrize("direction", [0, 1, 2])
def test_inference_batch_graph_replay(direction):
    L, B = 256, 4
    m = make(direction).eval()
    x, mask, _, _ = inputs(B, L)
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


def test_batch_limits():
    """D64 takes up to MAX_BATCH samples per call; wider widths and larger batches keep the one-sample contract."""
    bidir = make(0)
    x = torch.empty(trimul_h100.MAX_BATCH, 128, 128, D, device="cuda", dtype=torch.bfloat16)
    assert trimul_h100.serves(bidir, x)
    assert not trimul_h100.serves(bidir, torch.empty(trimul_h100.MAX_BATCH + 1, 128, 128, D, device="cuda", dtype=torch.bfloat16))
    with torch.no_grad():
        bidir.eval()
        assert trimul_h100.serves_inference(bidir, x, bidirectional=True)
        assert not trimul_h100.serves_inference(
            bidir, torch.empty(trimul_h100.MAX_BATCH + 1, 128, 128, D, device="cuda", dtype=torch.bfloat16), bidirectional=True)
    wide = BidirectionalTriangleMultiplication(128, implementation=ImplementationType.MINIWORLD).cuda()
    x128 = torch.empty(2, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    assert not trimul_h100.serves(wide, x128)
    with torch.no_grad():
        wide.eval()
        assert not trimul_h100.serves_inference(wide, x128, bidirectional=True)

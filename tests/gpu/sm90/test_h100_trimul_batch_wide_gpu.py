"""H100 TriMul D128 / D256 / D384 / D512, native batch: B samples in one call (tokens b-major, one launch per stage) against B calls
of one sample.

D128 runs the native K1 / K3 inference kernels and the B1 / B7 training kernels (bidirectional and single direction); D256 / D384 /
D512 run the B200-style port. What the batch changes is the slow token axis of K1 / K3 (rows = B L), the sample index of the token
mask (K1, B7, the owned pair mask), the per-sample row-dropout scale (K3, B1, the single-direction B1 and the port's K3 / gate
backward read row s L + j) and cuBLAS's batch dim (channels x B planes); everything else is token-flat. Each sample draws its own
dropout scale and mask, so an index mix-up between samples shows as a large error. The wide widths compare against B calls of the
same port (the preserved single-sample cells are switched off for the comparison: B > 1 always takes the port)."""

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.integrations import trimul_h100
from miniworld_engine.kernels.trimul_inproj.cuda import (
    h100_b200style,
    h100_inference,
    h100_single,
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
def hopper(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 (sm_90) required")
    if torch.cuda.get_device_properties(0).multi_processor_count != 132:
        pytest.skip("the cooperative D128 grids are qualified on full 132-SM H100s")
    torch.compiler.reset()
    # B = 1 at D256+ takes the same B200-style port as the batch (the comparison is B samples vs B calls of the same kernels)
    monkeypatch.setattr(trimul_h100, "_legacy_native_faster", lambda *a, **k: False)


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


def make(D, direction, seed=0):
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


def inputs(B, L, D, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, generator=g)
    mask = torch.rand(B, L, device="cuda", generator=g) > 0.15                                    # a different mask per sample
    ds = (torch.rand(B, 1, L, D, device="cuda", generator=g) > 0.25).to(torch.bfloat16) / 0.75    # a different dropout scale per sample
    dy = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, mask, ds, dy


def reference(m, direction, x, mask, ds):
    """FP32 statement of the module (the PyTorch reference of the residual update)."""
    p = dict(m.named_parameters())
    D = x.shape[-1]
    xn = F.layer_norm(x.float(), (D,), p["ln_pair.weight"], p["ln_pair.bias"], 1e-5)
    pm = (mask[:, :, None] & mask[:, None, :])[..., None].float()
    left = torch.sigmoid(F.linear(xn, p["to_left_gate.weight"])) * F.linear(xn, p["to_left.weight"]) * pm
    right = torch.sigmoid(F.linear(xn, p["to_right_gate.weight"])) * F.linear(xn, p["to_right.weight"]) * pm
    if direction == 0:
        h = left.shape[-1] // 2
        tri = torch.cat((torch.einsum("bikd,bjkd->bijd", left[..., :h], right[..., :h]),
                         torch.einsum("bkid,bkjd->bijd", left[..., h:], right[..., h:])), -1)
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
        first = a[0][0] if isinstance(a[0], (list, tuple)) else a[0] if hasattr(a[0], "shape") else a[1]
        calls.append(first.shape[0])
        return fn(*a, **k)

    monkeypatch.setattr(module, name, wrapped)
    return calls


def native_entry(monkeypatch, D, direction, training):
    if D >= 256:
        return counting(monkeypatch, h100_b200style, "trimul")
    if not training:
        return counting(monkeypatch, h100_inference, "inference")
    if direction == 0:
        return counting(monkeypatch, h100_training, "bidirectional_trimul")
    return counting(monkeypatch, h100_single, "single_trimul")


SHAPES = {   # (L, B); B 16 / 48 at L128 back max_batch = 48 (the full B 16 / 32 / 48 x L128 / 384 sweep is in the record)
    128: ((128, 2), (128, 8), (128, 16), (128, 48), (256, 4), (384, 4), (768, 2)),
    256: ((128, 2), (128, 16), (256, 4), (384, 2), (768, 2)),
    384: ((128, 2), (128, 16), (256, 4), (384, 2)),
    512: ((128, 2), (128, 16), (128, 48), (256, 4), (384, 2)),
}
CASES = [(D, L, B, direction) for D, shapes in SHAPES.items() for (L, B) in shapes for direction in (0, 1, 2)]
IDS = [f"D{D}-L{L}-B{B}-dir{d}" for D, L, B, d in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_inference_batch_matches_samples(case, monkeypatch):
    D, L, B, direction = case
    m = make(D, direction).eval()
    x, mask, _, _ = inputs(B, L, D)
    calls = native_entry(monkeypatch, D, direction, False)
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
    D, L, B, direction = case
    m = make(D, direction).train()
    x, mask, ds, dy = inputs(B, L, D)
    calls = native_entry(monkeypatch, D, direction, True)
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
    # run to run: bitwise at D128 (fixed-order reductions); the port's backward sums its LN_out row statistics and gate-affine
    # gradients with atomicAdd (B200's wide_bwd.cu), so at D256+ only its forward is bitwise
    y2, g2 = _train(m, x, mask, ds, dy, monkeypatch)
    assert torch.equal(y_b, y2)
    if D == 128:
        assert all(torch.equal(a, b) for a, b in zip(g_b, g2, strict=True))
    else:
        assert max(relative(a, b) for a, b in zip(g_b, g2, strict=True)) < 1e-4


GRAPH_CASES = [(D, direction) for D in (128, 256, 384, 512) for direction in (0, 1, 2)]
GRAPH_IDS = [f"D{D}-dir{d}" for D, d in GRAPH_CASES]


@pytest.mark.parametrize("case", GRAPH_CASES, ids=GRAPH_IDS)
def test_training_batch_graph_replay(case, monkeypatch):
    """A captured batched forward + backward replays with live inputs, masks, dropout scales and weights."""
    D, direction = case
    L, B = 256, 4
    m = make(D, direction).train()
    x, mask, ds, dy = inputs(B, L, D)
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
    if D == 128:
        assert all(torch.equal(a, b) for a, b in zip(grads, want_g, strict=True))
    else:   # the port's atomic backward sums (see test_training_batch_matches_samples)
        assert max(relative(a, b) for a, b in zip(grads, want_g, strict=True)) < 1e-4


@pytest.mark.parametrize("case", GRAPH_CASES, ids=GRAPH_IDS)
def test_inference_batch_graph_replay(case):
    D, direction = case
    L, B = 256, 4
    m = make(D, direction).eval()
    x, mask, _, _ = inputs(B, L, D)
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


PAD_CASES = [(D, L, direction) for D in (128, 256, 384, 512) for L in (200,) for direction in (0, 1, 2)]


@pytest.mark.parametrize("case", PAD_CASES, ids=[f"D{D}-L{L}-dir{d}" for D, L, d in PAD_CASES])
def test_inference_batch_padded_length(case):
    """B samples at an L the batched tiles do not divide: each sample is padded on the host (D128: to 16; D256+: to 128, the port's
    tile) with masked pad tokens and the pad rows dropped. Against the FP32 statement (B = 1 at D256+ is not served there)."""
    D, L, direction = case
    B = 2
    m = make(D, direction).eval()
    x, mask, _, _ = inputs(B, L, D)
    with torch.no_grad():
        assert trimul_h100.serves_inference(m, x, bidirectional=direction == 0)
        got = m(x, mask)
        assert got.shape == x.shape
        assert relative(got, reference(m, direction, x, mask, None)) < 1e-2
        if D == 128:
            want = torch.cat([m(x[i:i + 1], mask[i:i + 1]) for i in range(B)], 0)
            assert relative(got, want) < 1e-3


@pytest.mark.parametrize("D", [64, 128, 256, 384, 512])
def test_batch_limits(D):
    """Each width takes up to max_batch(D) samples in one launch; a larger batch is served in launches of at most max_batch(D)."""
    assert trimul_h100.max_batch(D) >= 1
    assert trimul_h100.max_batch(64) == trimul_h100.MAX_BATCH
    bidir = make(D, 0)
    limit = trimul_h100.max_batch(D)
    x = torch.empty(limit, 128, 128, D, device="cuda", dtype=torch.bfloat16)
    over = torch.empty(limit + 1, 128, 128, D, device="cuda", dtype=torch.bfloat16)
    assert trimul_h100.serves(bidir, x)
    assert trimul_h100._plan(bidir, x, False)[0] == limit
    assert trimul_h100.serves(bidir, over)
    assert trimul_h100._plan(bidir, over, False)[0] <= limit
    with torch.no_grad():
        bidir.eval()
        assert trimul_h100.serves_inference(bidir, x, bidirectional=True)
        assert trimul_h100._plan(bidir, x, True)[0] == limit
        assert trimul_h100.serves_inference(bidir, over, bidirectional=True)
        assert trimul_h100._plan(bidir, over, True)[0] <= limit


def _peak_below_device(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    assert peak < torch.cuda.get_device_properties(0).total_memory, peak
    return out


CHUNK_INFERENCE = [(512, 768, 48, 0), (128, 128, 64, 1)]   # one launch would need ~280 GB / B above max_batch(128)


@pytest.mark.parametrize("case", CHUNK_INFERENCE, ids=[f"D{D}-L{L}-B{B}-dir{d}" for D, L, B, d in CHUNK_INFERENCE])
def test_inference_chunks_a_batch_one_launch_cannot_take(case, monkeypatch):
    """A batch that does not fit one launch (memory or max_batch) runs in chunks (``trimul_h100._plan``), matches B calls of one
    sample bitwise and stays under the device's memory."""
    D, L, B, direction = case
    m = make(D, direction).eval()
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, generator=g)   # no dy / dropout scale: 29 GB at D512 L768 B48
    mask = torch.rand(B, L, device="cuda", generator=g) > 0.15
    chunk, _ = trimul_h100._plan(m, x, True)
    assert chunk < B
    calls = counting(monkeypatch, trimul_h100, "_update_inference_one")   # one launch per chunk
    with torch.no_grad():
        assert trimul_h100.serves_inference(m, x, bidirectional=direction == 0)
        got = _peak_below_device(lambda: m(x, mask))
        assert len(calls) == -(-B // chunk), (calls, chunk)
        assert max(calls) <= chunk, (calls, chunk)
        for i in range(B):
            assert torch.equal(got[i:i + 1], m(x[i:i + 1], mask[i:i + 1])), i


CHUNK_TRAINING = [(128, 768, 48, 0), (256, 768, 24, 1)]


@pytest.mark.parametrize("case", CHUNK_TRAINING, ids=[f"D{D}-L{L}-B{B}-dir{d}" for D, L, B, d in CHUNK_TRAINING])
def test_training_chunks_a_batch_one_launch_cannot_take(case, monkeypatch):
    """Training beyond one launch's memory: chunks, each checkpointed (its forward recomputed in the backward); y and dx match B
    calls of one sample, weight gradients their sum, peak under the device's memory."""
    D, L, B, direction = case
    m = make(D, direction).train()
    x, mask, ds, dy = inputs(B, L, D)
    chunk, checkpoint = trimul_h100._plan(m, x, False)
    assert chunk < B
    assert checkpoint
    assert trimul_h100.serves(m, x)
    y_b, g_b = _peak_below_device(lambda: _train(m, x, mask, ds, dy, monkeypatch))
    wsum = None
    for i in range(B):
        yi, gi = _train(m, x[i:i + 1], mask[i:i + 1], ds[i:i + 1], dy[i:i + 1], monkeypatch)
        assert torch.equal(y_b[i:i + 1], yi), i
        assert relative(g_b[0][i:i + 1], gi[0]) < 1e-5, i
        wsum = [g.float() for g in gi[1:]] if wsum is None else [a + g.float() for a, g in zip(wsum, gi[1:], strict=True)]
    for a, b in zip(g_b[1:], wsum, strict=True):
        assert relative(a, b) < 2e-3


CHUNK_GRAPH = [(128, 0, 1.0e9), (128, 1, 1.0e9), (256, 0, 2.5e9), (512, 2, 5.0e9)]   # checkpointed / plain / checkpointed / plain


@pytest.mark.parametrize("case", CHUNK_GRAPH, ids=[f"D{D}-dir{d}-mem{m:.1e}" for D, d, m in CHUNK_GRAPH])
def test_chunked_training_graph_replay(case, monkeypatch):
    """Chunked training (the device's memory shrunk so that B = 4 L256 is split; checkpointed or plain chunks) captures into a
    CUDA graph and replays with live inputs, masks, dropout scales and weights; it matches the unchunked call."""
    D, direction, memory = case
    L, B = 256, 4
    m = make(D, direction).train()
    x, mask, ds, dy = inputs(B, L, D)
    want_y, want_g = _train(m, x, mask, ds, dy, monkeypatch)                      # one launch
    monkeypatch.setattr(trimul_h100, "_total_memory", lambda index: int(memory))
    chunk, _ = trimul_h100._plan(m, x, False)
    assert chunk < B
    got_y, got_g = _train(m, x, mask, ds, dy, monkeypatch)                        # chunked, eager
    assert torch.equal(got_y, want_y)
    assert torch.equal(got_g[0], want_g[0])
    assert max(relative(a, b) for a, b in zip(got_g[1:], want_g[1:], strict=True)) < 2e-3
    x = x.requires_grad_()
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
    assert max(relative(a, b) for a, b in zip(grads, want_g, strict=True)) < 1e-4

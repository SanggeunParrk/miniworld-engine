"""B200 TriMul, batched samples at every width (D64 / 128 / 256 / 384 / 512, bidirectional and single direction, inference and
training): B samples in one call (tokens b-major, one launch per stage) against B calls of one sample.

The kernels are the same per token; what the batch changes is the sample index of the token mask (k1w, k1wb, the D64 / D128 pair
mask of b7m / b7g), the per-sample row-dropout scale (k3g, b1s / b1g, k3w, the wide gate backward read row s L + j) and cuBLAS's
batch dim (channels x B planes). Each sample draws its own dropout scale and mask, so an index mix-up between samples shows as a
large error. y and dx are compared bitwise with the B one-sample calls; the parameter gradients (a sum over the samples) within the
fp32 summation order. A batch larger than one launch takes (``max_batch``) or than fits in memory runs in chunks
(``trimul_b200._plan``): same bits, one autograd node, CUDA graphs and torch.compile."""

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.integrations import trimul_b200
from miniworld_engine.kernels.trimul_inproj.cuda import b200_infer, b200_train
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
def b200():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 (sm_100) required")
    old = settings.configure(force_portable_kernels=False)
    torch.compiler.reset()
    try:
        yield
    finally:
        settings.configure(**vars(old))


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


def make(D, direction, seed=0, hidden=None):
    torch.manual_seed(seed)
    if direction == 0:
        m = BidirectionalTriangleMultiplication(D, implementation=ImplementationType.MINIWORLD, p_drop=0.25)
    else:
        m = TriangleMultiplication(D, d_hidden=hidden, outgoing=direction == 1, implementation=ImplementationType.MINIWORLD,
                                   p_drop=0.25)
    m = m.cuda()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def inputs(B, L, D, seed=1, dy=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, generator=g)
    mask = torch.rand(B, L, device="cuda", generator=g) > 0.15                                    # a different mask per sample
    ds = (torch.rand(B, 1, L, D, device="cuda", generator=g) > 0.25).to(torch.bfloat16) / 0.75    # a different dropout scale per sample
    grad = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, generator=g) if dy else None
    return x, mask, ds, grad


def reference(m, direction, x, mask, ds):
    """FP32 statement of the module (the PyTorch reference of the residual update)."""
    p = {k: v.float() for k, v in m.named_parameters()}
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


def counting(monkeypatch, module, *names):
    """Samples per call of the named entry points (in call order, all names in one list)."""
    calls = []
    for name in names:
        fn = getattr(module, name)

        def wrapped(*a, fn=fn, **k):
            calls.append(a[0][0].shape[0])
            return fn(*a, **k)

        monkeypatch.setattr(module, name, wrapped)
    return calls


def launches(m, x, inference, direction):
    """The launch sizes ``trimul_b200._plan`` gives this call."""
    chunk, _ = trimul_b200._plan(m, x, inference, direction == 0)
    return [x.shape[0]] if chunk >= x.shape[0] else trimul_b200._split(x.shape[0], chunk)


def train(m, x, mask, ds, dy):
    m._make_drop_row_scale = lambda pair, p: ds
    try:
        xx = x.detach().clone().requires_grad_()
        params = list(m.parameters())
        y = m(xx, mask)
        grads = torch.autograd.grad(y, [xx, *params], dy)
    finally:
        del m._make_drop_row_scale
    return y.detach(), [g.detach().clone() for g in grads]


SHAPES = {   # (L, B): B 16 / 48 at L128 back max_batch = 48 per launch (the full B 1-48 x L128 / 384 / 768 / 1536 sweep is in the record)
    64: ((128, 2), (128, 16), (128, 48), (384, 4), (768, 2)),
    128: ((128, 2), (128, 16), (128, 48), (384, 4), (768, 2)),
    256: ((128, 2), (128, 16), (384, 2), (768, 2)),
    384: ((128, 2), (128, 16), (384, 2)),
    512: ((128, 2), (128, 16), (128, 48), (384, 2)),
}
CASES = [(D, L, B, direction) for D, shapes in SHAPES.items() for (L, B) in shapes for direction in (0, 1, 2)]
IDS = [f"D{D}-L{L}-B{B}-dir{d}" for D, L, B, d in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_inference_batch_matches_samples(case, monkeypatch):
    D, L, B, direction = case
    m = make(D, direction).eval()
    x, mask, _, _ = inputs(B, L, D, dy=False)
    with torch.no_grad():
        assert trimul_b200.serves_inference(m, x, bidirectional=direction == 0)
        calls = counting(monkeypatch, b200_infer, "inference", "inference_into")
        got = m(x, mask)
        assert calls == launches(m, x, True, direction), calls      # one launch per stage (or per chunk, ``_plan``)
        monkeypatch.undo()
        for i in range(B):
            assert torch.equal(got[i:i + 1], m(x[i:i + 1], mask[i:i + 1])), i
        k = min(B, 2)                                                # the FP32 statement on two samples
        assert relative(got[:k], reference(m, direction, x[:k], mask[:k], None)) < 1e-2
        got0 = m(x, None)                                            # no mask
        for i in (0, B - 1):
            assert torch.equal(got0[i:i + 1], m(x[i:i + 1], None)), i
        assert torch.equal(got, m(x, mask))                          # bitwise run to run


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_training_batch_matches_samples(case, monkeypatch):
    D, L, B, direction = case
    m = make(D, direction).train()
    x, mask, ds, dy = inputs(B, L, D)
    assert trimul_b200.serves_train(m, x, bidirectional=direction == 0)
    calls = counting(monkeypatch, b200_train, "trimul_train", "trimul_train_chunked")
    y_b, g_b = train(m, x, mask, ds, dy)
    assert calls == [B], calls                                       # one autograd node (one launch or its chunks)
    wsum = None
    for i in range(B):
        yi, gi = train(m, x[i:i + 1], mask[i:i + 1], ds[i:i + 1], dy[i:i + 1])
        assert torch.equal(y_b[i:i + 1], yi), ("y", i)
        assert torch.equal(g_b[0][i:i + 1], gi[0]), ("dx", i)
        wsum = [g.float() for g in gi[1:]] if wsum is None else [a + g.float() for a, g in zip(wsum, gi[1:], strict=True)]
    for k, (a, b) in enumerate(zip(g_b[1:], wsum, strict=True)):    # parameter gradients: the batch sums over the samples
        assert relative(a, b) < 2e-3, (k, relative(a, b))
    if B * L * L * D <= 2 * 384 * 384 * 128:                         # the FP32 PyTorch reference, every gradient
        xr = x.detach().clone().requires_grad_()
        params = list(m.parameters())
        z = reference(m, direction, xr, mask, ds)
        h = torch.autograd.grad(z, [xr, *params], dy.float())
        errors = [relative(a, b) for a, b in zip((y_b, *g_b), (z, *h), strict=True)]
        assert errors[0] < 5e-3, errors
        assert max(errors[1:]) < 2e-2, errors
    y2, g2 = train(m, x, mask, ds, dy)                                # run to run: y, dx bitwise
    assert torch.equal(y_b, y2)
    assert torch.equal(g_b[0], g2[0])
    assert max(relative(a, b) for a, b in zip(g_b[1:], g2[1:], strict=True)) < 1e-5


HIDDEN2 = [(D, B, direction) for D in (64, 128) for B in (2, 8) for direction in (1, 2)]


@pytest.mark.parametrize("case", HIDDEN2, ids=[f"D{D}-B{B}-dir{d}" for D, B, d in HIDDEN2])
def test_hidden_twice_width_batch(case):
    """One direction at hidden 2 D (the AF3 / Protenix template block) on the D64 / D128 kernels, batched."""
    D, B, direction = case
    L = 256
    m = make(D, direction, hidden=2 * D)
    x, mask, ds, dy = inputs(B, L, D)
    with torch.no_grad():
        m.eval()
        got = m(x, mask)
        for i in range(B):
            assert torch.equal(got[i:i + 1], m(x[i:i + 1], mask[i:i + 1])), i
    m.train()
    y_b, g_b = train(m, x, mask, ds, dy)
    for i in range(B):
        yi, gi = train(m, x[i:i + 1], mask[i:i + 1], ds[i:i + 1], dy[i:i + 1])
        assert torch.equal(y_b[i:i + 1], yi) and torch.equal(g_b[0][i:i + 1], gi[0]), i


def test_d64_takes_48_samples_at_l384(monkeypatch):
    """The D64 limit is 48 per launch (was 8): bidirectional, L384, one launch, bitwise against the samples."""
    D, L, B = 64, 384, 48
    m = make(D, 0).train()
    x, mask, ds, dy = inputs(B, L, D)
    assert trimul_b200._plan(m, x, False, True) == (B, False)
    y_b, g_b = train(m, x, mask, ds, dy)
    for i in (0, 17, B - 1):
        yi, gi = train(m, x[i:i + 1], mask[i:i + 1], ds[i:i + 1], dy[i:i + 1])
        assert torch.equal(y_b[i:i + 1], yi) and torch.equal(g_b[0][i:i + 1], gi[0]), i


GRAPH_CASES = [(D, direction) for D in (64, 128, 256, 384, 512) for direction in (0, 1, 2)]
GRAPH_IDS = [f"D{D}-dir{d}" for D, d in GRAPH_CASES]


def _graph_train_step(m, x, mask, ds, dy):
    m._make_drop_row_scale = lambda pair, p: ds
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
    assert torch.equal(grads[0], want_g[0])
    assert max(relative(a, b) for a, b in zip(grads[1:], want_g[1:], strict=True)) < 1e-5


@pytest.mark.parametrize("case", GRAPH_CASES, ids=GRAPH_IDS)
def test_training_batch_graph_replay(case):
    """A captured batched forward + backward replays with live inputs, masks, dropout scales and weights."""
    D, direction = case
    m = make(D, direction).train()
    x, mask, ds, dy = inputs(4, 256, D)
    _graph_train_step(m, x.requires_grad_(), mask, ds, dy)


@pytest.mark.parametrize("case", GRAPH_CASES, ids=GRAPH_IDS)
def test_inference_batch_graph_replay(case):
    D, direction = case
    m = make(D, direction).eval()
    x, mask, _, _ = inputs(4, 256, D, dy=False)
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


COMPILE_CASES = [(64, 0), (128, 1), (256, 0), (512, 2)]


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("case", COMPILE_CASES, ids=[f"D{D}-dir{d}" for D, d in COMPILE_CASES])
def test_torch_compile_batch(case, chunked, monkeypatch):
    """torch.compile (fullgraph) of the module on a batch: the same bits as eager, inference and training (chunked: the device's
    memory shrunk so that the batch is split; the split is decided from shapes at trace time)."""
    D, direction = case
    L, B = 256, 4
    m = make(D, direction)
    x, mask, ds, dy = inputs(B, L, D)
    if chunked:
        tokens = L * L * D
        for memory in range(B * 120 * tokens, B * 10 * tokens, -tokens):    # the largest device on which both modes split
            monkeypatch.setattr(trimul_b200, "_total_memory", lambda index, mem=memory: int(mem))
            if trimul_b200._plan(m, x, True, direction == 0)[0] < B and trimul_b200._plan(m, x, False, direction == 0)[0] < B:
                break
        else:
            pytest.fail("no memory size splits both modes")
    compiled = torch.compile(m, fullgraph=True)
    with torch.no_grad():
        m.eval()
        assert torch.equal(compiled(x, mask), m(x, mask))
    m.train()
    want_y, want_g = train(m, x, mask, ds, dy)
    got_y, got_g = _compiled_train(compiled, m, x, mask, ds, dy)
    assert torch.equal(got_y, want_y)
    assert torch.equal(got_g[0], want_g[0])
    assert max(relative(a, b) for a, b in zip(got_g[1:], want_g[1:], strict=True)) < 1e-5


def _compiled_train(compiled, m, x, mask, ds, dy):
    m._make_drop_row_scale = lambda pair, p: ds
    try:
        xx = x.detach().clone().requires_grad_()
        params = list(m.parameters())
        y = compiled(xx, mask)
        grads = torch.autograd.grad(y, [xx, *params], dy)
    finally:
        del m._make_drop_row_scale
    return y.detach(), [g.detach().clone() for g in grads]


@pytest.mark.parametrize("D", [64, 128, 256, 384, 512])
def test_batch_limits(D):
    """Each width takes up to max_batch(D) samples (and up to ``_LAUNCH_TOKENS`` pair tokens) in one launch; a larger batch is served in
    launches of at most that many."""
    assert trimul_b200.max_batch(D) == 48
    assert trimul_b200.max_batch(64) == trimul_b200.MAX_BATCH
    m = make(D, 0)
    limit = trimul_b200.max_batch(D)
    x = torch.empty(limit, 128, 128, D, device="cuda", dtype=torch.bfloat16)
    over = torch.empty(limit + 1, 128, 128, D, device="cuda", dtype=torch.bfloat16)
    for inference in (False, True):                                  # at most max_batch samples and _LAUNCH_TOKENS pair tokens
        cap = trimul_b200._LAUNCH_TOKENS[(not inference, D == 64)]
        want = limit if cap is None else min(limit, cap // 128 ** 2)
        assert trimul_b200._plan(m, x, inference, True)[0] == want
        assert trimul_b200._plan(m, over, inference, True)[0] <= want
    assert trimul_b200.serves_train(m, x, bidirectional=True)
    assert trimul_b200.serves_train(m, over, bidirectional=True)
    with torch.no_grad():
        assert trimul_b200.serves_inference(m, x, bidirectional=True)
        assert trimul_b200.serves_inference(m, over, bidirectional=True)


def _peak_below_device(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    assert peak < torch.cuda.get_device_properties(0).total_memory, peak
    return out


CHUNK_INFERENCE = [(512, 768, 48, 0), (128, 128, 64, 1), (64, 1536, 49, 2)]   # memory / max_batch / max_batch


@pytest.mark.parametrize("case", CHUNK_INFERENCE, ids=[f"D{D}-L{L}-B{B}-dir{d}" for D, L, B, d in CHUNK_INFERENCE])
def test_inference_chunks_a_batch_one_launch_cannot_take(case, monkeypatch):
    """A batch that does not fit one launch (memory or max_batch) runs in chunks written in place into the output, matches B calls
    of one sample bitwise and stays under the device's memory."""
    D, L, B, direction = case
    m = make(D, direction).eval()
    x, mask, _, _ = inputs(B, L, D, dy=False)
    chunk, _ = trimul_b200._plan(m, x, True, direction == 0)
    assert chunk < B
    calls = counting(monkeypatch, b200_infer, "inference", "inference_into")
    with torch.no_grad():
        assert trimul_b200.serves_inference(m, x, bidirectional=direction == 0)
        got = _peak_below_device(lambda: m(x, mask))
        assert len(calls) == -(-B // chunk) and max(calls) <= chunk, (calls, chunk)
        for i in range(B):
            assert torch.equal(got[i:i + 1], m(x[i:i + 1], mask[i:i + 1])), i


CHUNK_TRAINING = [(256, 768, 48, 0, True), (512, 384, 48, 0, False), (384, 768, 32, 1, False), (64, 128, 60, 2, False)]


@pytest.mark.parametrize("case", CHUNK_TRAINING, ids=[f"D{D}-L{L}-B{B}-dir{d}-{'ckpt' if c else 'plain'}" for D, L, B, d, c in CHUNK_TRAINING])
def test_training_chunks_a_batch_one_launch_cannot_take(case, monkeypatch):
    """Training beyond one launch (memory, or max_batch at D64 B60): chunks of one autograd node, plain or checkpointed (``_plan``,
    180 GB B200); y and dx match B calls of one sample bitwise, weight gradients their sum, peak under the device's memory."""
    D, L, B, direction, checkpoint = case
    m = make(D, direction).train()
    x, mask, ds, dy = inputs(B, L, D)
    chunk, ck = trimul_b200._plan(m, x, False, direction == 0)
    assert chunk < B and ck == checkpoint, (chunk, ck)
    assert trimul_b200.serves_train(m, x, bidirectional=direction == 0)
    calls = counting(monkeypatch, b200_train, "trimul_train_chunked")
    y_b, g_b = _peak_below_device(lambda: train(m, x, mask, ds, dy))
    assert calls == [B]
    wsum = None
    for i in range(B):
        yi, gi = train(m, x[i:i + 1], mask[i:i + 1], ds[i:i + 1], dy[i:i + 1])
        assert torch.equal(y_b[i:i + 1], yi), i
        assert torch.equal(g_b[0][i:i + 1], gi[0]), i
        wsum = [g.float() for g in gi[1:]] if wsum is None else [a + g.float() for a, g in zip(wsum, gi[1:], strict=True)]
    for a, b in zip(g_b[1:], wsum, strict=True):
        assert relative(a, b) < 2e-3


CHUNK_GRAPH = [(64, 0), (128, 1), (256, 0), (512, 2)]


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("case", CHUNK_GRAPH, ids=[f"D{D}-dir{d}" for D, d in CHUNK_GRAPH])
def test_chunked_training_graph_replay(case, checkpoint, monkeypatch):
    """Chunked training (the device's memory shrunk so that B = 4 L256 is split into plain or checkpointed chunks) matches the
    one-launch call bitwise (y, dx), captures into a CUDA graph and replays with live inputs, masks, dropout scales and weights."""
    D, direction = case
    L, B = 256, 4
    m = make(D, direction).train()
    x, mask, ds, dy = inputs(B, L, D)
    want_y, want_g = train(m, x, mask, ds, dy)                                  # one launch
    four = direction == 0
    tokens = L * L * D
    for memory in range(B * 20 * tokens, B * 120 * tokens, tokens):              # the smallest device that gives this chunk kind
        monkeypatch.setattr(trimul_b200, "_total_memory", lambda index, mem=memory: int(mem))
        chunk, ck = trimul_b200._plan(m, x, False, four)
        if chunk < B and ck == checkpoint:
            break
    else:
        pytest.fail("no memory size gives this chunk kind")
    got_y, got_g = train(m, x, mask, ds, dy)                                    # chunked, eager
    assert torch.equal(got_y, want_y)
    assert torch.equal(got_g[0], want_g[0])
    assert max(relative(a, b) for a, b in zip(got_g[1:], want_g[1:], strict=True)) < 2e-3
    _graph_train_step(m, x.requires_grad_(), mask, ds, dy)


@pytest.mark.parametrize("case", [(384, 4, 0), (384, 4, 1), (384, 4, 2), (768, 2, 0), (128, 16, 0)],
                         ids=["L384-B4-dir0", "L384-B4-dir1", "L384-B4-dir2", "L768-B2-dir0", "L128-B16-dir0"])
def test_d64_sample_major_layout_matches_plane_major(case, monkeypatch):
    """D64 training keeps planes / t / dt / the plane gradient sample-major ([P / 64, B, 64, L, L], ``b200_train._sample_major``):
    k1w, k3g, b1s and b7m address them through 4-D maps. Same products and per-token reductions as the [P, B, L, L] layout, so y, dx
    and the LayerNorm gradients are bitwise equal to it and the weight gradients within the atomic flush order; the saved planes
    are 5-D."""
    L, B, direction = case
    m = make(64, direction)
    x, mask, ds, dy = inputs(B, L, 64)
    saved = []
    fwd = b200_train.forward_small

    def spy(*a, **k):
        out = fwd(*a, **k)
        saved.append(out[1].dim())
        return out

    monkeypatch.setattr(b200_train, "forward_small", spy)
    y_sm, g_sm = train(m, x, mask, ds, dy)
    monkeypatch.setattr(b200_train, "_sample_major", lambda *a: False)
    y_pm, g_pm = train(m, x, mask, ds, dy)
    assert saved == [5, 4]
    assert torch.equal(y_sm, y_pm) and torch.equal(g_sm[0], g_pm[0])
    names = [n for n, _ in m.named_parameters()]
    for n, a, b in zip(names, g_sm[1:], g_pm[1:], strict=True):
        if n.startswith("ln_"):
            assert torch.equal(a, b), n
        else:
            assert relative(a, b) < 1e-5, (n, relative(a, b))

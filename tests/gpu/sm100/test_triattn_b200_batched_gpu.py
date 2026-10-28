"""B200 TriangleAttention on B samples in one call (templates, structures or augments folded into the batch), at any length.

One launch per kernel takes up to ``max_batch`` samples; a call with more, or with more than fit in memory, runs in launches of
``plan``'s chunk one after another; a length that is not a multiple of 128 is padded (pad keys masked).  Checked here: B samples in one
call against B calls of one sample (each with its own key mask and dropout scale) and against the fp32 PyTorch statements, every
gradient, both nodes, ragged lengths, bitwise run-to-run determinism (NaN-poisoned allocator), CUDA-graph capture and torch.compile, the
forced chunk modes (saved tensors kept and recomputed), long lengths, and calls that do not fit in one launch.
"""

import gc

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import triattn_b200 as tri
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention.module import TriangleAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# (d_pair, d_hidden, heads): AF3 / Protenix-v1 / Protenix-v2 template stacks, MiniWorld (the fused path), the wide trunks
CONFIGS = {
    "C64-H4x16": (64, 64, 4),
    "C64-H4x32": (64, 128, 4),
    "C64-H2x32": (64, 64, 2),
    "C128": (128, 128, 4),
    "C256": (256, 256, 8),
    "C384": (384, 384, 12),
    "C512": (512, 512, 16),
}
BATCHES = (1, 2, 4, 8, 16, 32, 48)
LENGTHS = (128, 300, 384, 768)          # 300: a ragged length (computed at 384, the pad keys masked)
#: what the test itself holds next to the call (x, dy, the batched y and dx): cases beyond it are not run ("where memory allows")
TEST_BYTES = 64 << 30


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


def make(cfg, impl=ImplementationType.MINIWORLD, starting=True, p_drop=0.0, dtype=torch.bfloat16, seed=0):
    c, hd, h = cfg
    torch.manual_seed(seed)
    m = TriangleAttention(c, h, d_hidden=hd, starting=starting, p_drop=p_drop, implementation=impl)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m.cuda().to(dtype)


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 (sm_100) required")
    old = settings.configure(force_portable_kernels=False)
    try:
        yield
    finally:
        settings.configure(**vars(old))
        gc.collect()
        torch.cuda.empty_cache()


def inputs(B, L, C, seed=1, grad=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, L, L, C, device="cuda", dtype=torch.bfloat16, generator=g)
    # every sample its own key mask (sample 0 unmasked, the others drop 5-40 % of the keys)
    keep = torch.rand(B, L, device="cuda", generator=g) > torch.linspace(0.0, 0.4, B, device="cuda")[:, None]
    keep[0] = True
    keep[:, 0] = True
    dy = 0.1 * torch.randn(B, L, L, C, device="cuda", dtype=torch.bfloat16, generator=g) if grad else None
    return x, keep, dy


class FixedDrop:
    """The module's dropout scale replaced by a fixed draw [B, 1, L, C] (starting) / [B, L, 1, C] (ending), sliced per call."""

    def __init__(self, module, scale):
        self.scale, self.sel = scale, slice(None)
        module._make_drop_scale = lambda pair, p: self.scale[self.sel]


def drop_scale(module, B, L, C, seed=7):
    shape = [B, L, L, C]
    shape[module._drop_dim] = 1
    g = torch.Generator(device="cuda").manual_seed(seed)
    keep = torch.rand(shape, device="cuda", generator=g) > module.p_drop
    return keep.to(torch.bfloat16) / (1.0 - module.p_drop)


def run(m, x, mask, dy, train):
    """-> (y, dx, parameter gradients) of one call (dx and the gradients None in inference)."""
    if not train:
        with torch.no_grad():
            return m.eval()(x, mask), None, None
    m.zero_grad(set_to_none=True)
    xi = x.detach().requires_grad_()
    y = m.train()(xi, mask)
    y.backward(dy)
    return y.detach(), xi.grad, [p.grad.float().clone() for p in m.parameters()]


def against_per_sample(m, x, mask, dy, train, fixed=None):
    """B samples in one call against B calls of one sample: max relative difference of y - x and dx per sample, and of the parameter
    gradients against the fp32 sum of the per-sample ones."""
    y, dx, grads = run(m, x, mask, dy, train)
    assert all(torch.isfinite(t).all() for t in y)  # per sample: a bool copy of a 48-sample y is large
    worst = {"y": 0.0, "dx": 0.0, "params": 0.0}
    equal = True
    sums = None
    for i in range(x.shape[0]):
        if fixed is not None:
            fixed.sel = slice(i, i + 1)
        yi, dxi, gi = run(m, x[i:i + 1], mask[i:i + 1], None if dy is None else dy[i:i + 1], train)
        equal &= torch.equal(yi[0], y[i])
        worst["y"] = max(worst["y"], relative(y[i].float() - x[i].float(), yi[0].float() - x[i].float()))
        if train:
            worst["dx"] = max(worst["dx"], relative(dx[i], dxi[0]))
            sums = gi if sums is None else [a + b for a, b in zip(sums, gi, strict=True)]
    if fixed is not None:
        fixed.sel = slice(None)
    if train:
        worst["params"] = max(relative(a, b) for a, b in zip(grads, sums, strict=True))
    return worst, equal, (y, dx, grads)


def check_reference(cfg, ours, x, mask, dy, train, got, starting, scale=None, samples=None):
    """The fp32 PyTorch statements on ``samples`` (all when None, then also the parameter gradients): within 1.25x the bf16 statements'
    own error."""
    idx = list(range(x.shape[0])) if samples is None else samples
    y, dx, grads = got
    ref16 = make(cfg, ImplementationType.PYTORCH, starting, ours.p_drop)
    ref16.load_state_dict(ours.state_dict())
    ref32 = make(cfg, ImplementationType.PYTORCH, starting, ours.p_drop, dtype=torch.float32)
    ref32.load_state_dict({k: v.float() for k, v in ours.state_dict().items()})
    xs, ms, dys = x[idx], mask[idx], dy[idx] if train else None
    out = {}
    for name, m, xin in (("ref16", ref16, xs), ("ref32", ref32, xs.float())):
        xi = xin.clone().requires_grad_(train)
        with torch.set_grad_enabled(train):
            if scale is None:
                yr = m.train(train)(xi, ms)
            else:
                yr = xi + m.train()._attention(xi, ms) * scale[idx].to(xin.dtype)
            if train:
                yr.backward(dys.to(yr.dtype))
        out[name] = (yr.detach(), xi.grad if train else None, [p.grad for p in m.parameters()] if train else None)
        del xi, yr
    e = relative(y[idx].float() - xs.float(), out["ref32"][0] - xs.float())
    e16 = relative(out["ref16"][0].float() - xs.float(), out["ref32"][0] - xs.float())
    assert e <= 1.25 * e16 + 1e-3, ("y", e, e16)
    if train:
        e = relative(dx[idx], out["ref32"][1])
        e16 = relative(out["ref16"][1], out["ref32"][1])
        assert e <= 1.25 * e16 + 2e-3, ("dx", e, e16)
        if samples is None:
            for n, a, b16, b32 in zip([n for n, _ in ours.named_parameters()], grads, out["ref16"][2], out["ref32"][2], strict=True):
                e, e16 = relative(a, b32), relative(b16, b32)
                assert e <= 1.25 * e16 + 2e-3, (n, e, e16)


def _bytes(B, L, C, train):
    return (4 if train else 2) * B * L * L * C * 2


#: the fp32 PyTorch statements materialise the [B, H, L, L, L] logits (training: about four such tensors)
REFERENCE_BYTES = 40 << 30


def reference_samples(batch, length, heads, train):
    """The samples the fp32 statements run on: all (None: also the parameter gradients), the first and last, the last, or none."""
    per = heads * length**3 * 4 * (4 if train else 2)
    if batch * per <= REFERENCE_BYTES:
        return None
    if 2 * per <= REFERENCE_BYTES:
        return [0, batch - 1]
    return [batch - 1] if per <= REFERENCE_BYTES else []


@pytest.mark.parametrize("train", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_batched_matches_per_sample_calls(name, batch, length, train):
    """B samples in one call (or in chunks) = B calls of one sample; the fp32 statements on every sample where they fit (then also the
    parameter gradients), else on the first and last, or the last (``reference_samples``)."""
    cfg = CONFIGS[name]
    C = cfg[0]
    if _bytes(batch, length, C, train) > TEST_BYTES:
        pytest.skip("the test's own tensors would not fit")
    m = make(cfg, p_drop=0.25 if train else 0.0)
    x, mask, dy = inputs(batch, length, C, grad=train)
    assert tri.kind_of(m.train(train), x, mask) == (tri.FUSED if name == "C128" else tri.GENERIC)
    fixed = FixedDrop(m, drop_scale(m, batch, length, C)) if train else None
    worst, equal, got = against_per_sample(m, x, mask, dy, train, fixed)
    print(f"\n{name} B{batch} L{length} {'train' if train else 'infer'} {worst} y-bitwise={equal}")
    assert equal, "y not bitwise equal to B one-sample calls"
    assert worst["y"] < 1e-3, worst
    if train:
        assert worst["dx"] < 5e-3, worst
        assert worst["params"] < 5e-3, worst
    samples = reference_samples(batch, length, cfg[2], train)
    if samples != []:
        check_reference(cfg, m, x, mask, dy, train, got, True, fixed.scale if train else None, samples)


@pytest.mark.parametrize("train", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("length", [128, 300, 384])
@pytest.mark.parametrize("batch", [4, 48])
@pytest.mark.parametrize("name", ["C64-H4x16", "C128", "C256"])
def test_ending_node(name, batch, length, train):
    cfg = CONFIGS[name]
    C = cfg[0]
    m = make(cfg, starting=False, p_drop=0.25 if train else 0.0)
    x, mask, dy = inputs(batch, length, C, grad=train)
    assert tri.kind_of(m.train(train), x, mask) is not None
    fixed = FixedDrop(m, drop_scale(m, batch, length, C)) if train else None
    worst, equal, got = against_per_sample(m, x, mask, dy, train, fixed)
    assert equal, "y not bitwise equal to B one-sample calls"
    assert worst["y"] < 1e-3, worst
    if train:
        assert worst["dx"] < 5e-3, worst
        assert worst["params"] < 5e-3, worst
    check_reference(cfg, m, x, mask, dy, train, got, False, fixed.scale if train else None, [0, batch - 1])


@pytest.mark.parametrize("length", [16, 17, 33, 100, 129, 250, 520])
@pytest.mark.parametrize("name", ["C64-H4x16", "C128", "C256"])
def test_ragged_lengths(name, length):
    """Any L >= 16: padded to whole 128-token tiles (pad keys masked, pad rows / columns cut); inference and training against the fp32
    statements and B one-sample calls, with and without a key mask."""
    cfg = CONFIGS[name]
    C, B = cfg[0], 3
    m = make(cfg, p_drop=0.25)
    x, mask, dy = inputs(B, length, C)
    fixed = FixedDrop(m, drop_scale(m, B, length, C))
    assert tri.kind_of(m, x, mask) is not None
    for train in (False, True):
        worst, equal, got = against_per_sample(m, x, mask, dy, train, fixed if train else None)
        assert equal
        assert worst["dx"] < 5e-3 and worst["params"] < 5e-3, worst
        check_reference(cfg, m, x, mask, dy, train, got, True, fixed.scale if train else None)
    with torch.no_grad():                                             # no mask: the pad keys are still masked
        y = m.eval()(x)
        ref = make(cfg, ImplementationType.PYTORCH)
        ref.load_state_dict(m.state_dict())
        ref32 = make(cfg, ImplementationType.PYTORCH, dtype=torch.float32)
        ref32.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
        want = ref32.eval()(x.float())
        e, e16 = relative(y.float() - x.float(), want - x.float()), relative(ref.eval()(x).float() - x.float(), want - x.float())
    assert e <= 1.25 * e16 + 1e-3, (e, e16)


def test_shortest_lengths_not_served():
    m = make(CONFIGS["C128"])
    assert tri.kind_of(m, torch.zeros(1, 15, 15, 128, device="cuda", dtype=torch.bfloat16)) is None
    assert tri.kind_of(m, torch.zeros(1, 16, 16, 128, device="cuda", dtype=torch.bfloat16)) == tri.FUSED


def poisoned(fn):
    """Run fn with every new allocation of the caching allocator NaN-filled (fresh blocks and reused ones)."""
    torch.cuda.empty_cache()
    hog = []
    try:  # fill the free cached memory with NaN blocks, then release them: the call's allocations reuse NaN-filled memory
        for _ in range(64):
            hog.append(torch.full((64 << 20,), float("nan"), device="cuda", dtype=torch.float32))
    except torch.OutOfMemoryError:
        pass
    del hog
    return fn()


@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
@pytest.mark.parametrize("det", [True, False], ids=["deterministic-wgrad", "default"])
@pytest.mark.parametrize("name", ["C128", "C256", "C64-H4x16"])
def test_deterministic(name, det, starting):
    """Three runs of one batched training call (NaN-poisoned allocator): the output bit for bit, and the q / k / v / gate / out weight
    gradients bit for bit on the wide path always and on the fused path with ``settings.triattn_b200_deterministic_wgrad`` (by default
    its weight-gradient partials are added with fp32 atomics); the LayerNorm and the bias projection gradients carry the core's fp32
    reduce-add of dbias (L2 add order)."""
    settings.configure(triattn_b200_deterministic_wgrad=det)
    cfg = CONFIGS[name]
    m = make(cfg, starting=starting, p_drop=0.25)
    x, mask, dy = inputs(8, 384, cfg[0])
    FixedDrop(m, drop_scale(m, 8, 384, cfg[0]))
    runs = [poisoned(lambda: run(m, x, mask, dy, True)) for _ in range(3)]
    names = ["y", "x", *(n for n, _ in m.named_parameters())]
    must = BITWISE if det or name != "C128" else ("y",)
    bitwise = []
    for r in runs[1:]:
        for i, n in enumerate(names):
            a = runs[0][0] if i == 0 else runs[0][1] if i == 1 else runs[0][2][i - 2]
            b = r[0] if i == 0 else r[1] if i == 1 else r[2][i - 2]
            if torch.equal(a, b):
                bitwise.append(n)
            elif n in must:
                raise AssertionError(f"{n} not bitwise")
            else:
                assert relative(a, b) < 1e-2, n
    print(f"\n{name} det={det} starting={starting}: bitwise {sorted(set(bitwise))}")


def test_deterministic_wgrad_matches_default():
    """The fused path's fixed-order weight-gradient sum against the default atomics: the same output, gradients within fp32 rounding."""
    cfg = CONFIGS["C128"]
    m = make(cfg, p_drop=0.25)
    x, mask, dy = inputs(4, 256, cfg[0])
    FixedDrop(m, drop_scale(m, 4, 256, cfg[0]))
    got = {}
    for det in (False, True):
        settings.configure(triattn_b200_deterministic_wgrad=det)
        got[det] = run(m, x, mask, dy, True)
    assert torch.equal(got[False][0], got[True][0])
    assert relative(got[False][1], got[True][1]) < 1e-4
    for a, b in zip(got[False][2], got[True][2], strict=True):
        assert relative(a, b) < 1e-3


#: what test_deterministic requires bit for bit run to run (the rest within 1e-2: dbias's L2 reduce order)
BITWISE = ("y", "to_query.weight", "to_key.weight", "to_value.weight", "to_gate.weight", "to_out.weight")


@pytest.mark.parametrize("length", [256, 208], ids=["L256", "L208-padded"])
@pytest.mark.parametrize("mode", ["inference", "plain", "checkpoint"])
@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
@pytest.mark.parametrize("name", ["C128", "C64-H4x16", "C256"])
def test_forced_chunks_match_one_launch(monkeypatch, name, starting, mode, length):
    """Chunks of 2 of a 5-sample call (sizes 2 / 2 / 1) against one launch: chunks keeping their saved tensors, chunks rerunning their
    forward in the backward (on the same dropout draw), inference chunks written in place / copied into one output.  The output is the
    same bits; dx and the parameter gradients (added up over the chunks in fp32) within the bf16 rounding."""
    cfg = CONFIGS[name]
    C, B, L = cfg[0], 5, length
    train = mode != "inference"
    m = make(cfg, starting=starting, p_drop=0.25 if train else 0.0)
    x, mask, dy = inputs(B, L, C)
    if train:
        FixedDrop(m, drop_scale(m, B, L, C))
    one = run(m, x, mask, dy, train)
    calls = []
    real = tri.plan

    def forced(*args, **kwargs):
        calls.append((args, kwargs))
        return 2, mode == "checkpoint"

    monkeypatch.setattr(tri, "plan", forced)
    assert tri._split(B, 2) == [2, 2, 1]
    chunked = run(m, x, mask, dy, train)
    assert calls
    assert real(*calls[0][0], **calls[0][1])[0] >= B
    assert torch.equal(chunked[0], one[0])
    if train:
        assert relative(chunked[1], one[1]) < 5e-3
        for a, b in zip(chunked[2], one[2], strict=True):
            assert relative(a, b) < 5e-3


#: (name, training, L, B): calls whose one launch would not fit in 0.85 of a B200 (179 GiB), so ``plan`` splits them
SPLIT_CASES = [
    ("C128", True, 768, 96),
    ("C256", True, 768, 32),
    ("C512", False, 768, 40),
    ("C512", True, 768, 16),
    ("C512", True, 768, 32),        # the 32 samples' saved tensors do not fit next to x / y / dy / dx: chunks rerun their forward
    ("C64-H4x16", True, 768, 96),
]


@pytest.mark.parametrize("case", SPLIT_CASES, ids=[f"{n}-{'train' if t else 'infer'}-L{L}-B{B}" for n, t, L, B in SPLIT_CASES])
def test_calls_that_do_not_fit_in_one_launch(case):
    """Split into chunks: they run, stay under the device's memory and match B calls of one sample."""
    name, train, L, B = case
    cfg = CONFIGS[name]
    C = cfg[0]
    total = torch.cuda.get_device_properties(0).total_memory
    m = make(cfg, p_drop=0.25 if train else 0.0)
    probe = torch.empty(1, L, L, C, device="cuda", dtype=torch.bfloat16)
    chunk, checkpoint = tri.plan(tri.kind_of(m, probe), B, L, C, cfg[2], train, total)
    assert chunk < B
    x, mask, dy = inputs(B, L, C, grad=train)
    fixed = FixedDrop(m, drop_scale(m, B, L, C)) if train else None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    worst, equal, _ = against_per_sample(m, x, mask, dy, train, fixed)
    peak = torch.cuda.max_memory_allocated()
    print(f"\n{case}: chunk {chunk} checkpoint {checkpoint} peak {peak / 2**30:.1f} GiB of {total / 2**30:.1f} {worst}")
    assert peak < 0.95 * total
    assert equal
    assert worst["y"] < 1e-3, worst
    if train:
        assert worst["dx"] < 5e-3, worst
        assert worst["params"] < 5e-3, worst


def rows_reference(m, x, mask, rows, dtype=torch.float32):
    """The module's statements of the starting node in ``dtype`` for the pair rows ``rows`` alone (the bias term still from the whole
    pair): the attention update y - x of those rows."""
    p = {k: v.to(dtype) for k, v in m.state_dict().items()}
    H = m.n_head
    with torch.no_grad():
        y = torch.nn.functional.layer_norm(x.to(dtype), (x.shape[-1],), p["ln_pair.weight"], p["ln_pair.bias"], m.ln_pair.eps)
        bias = torch.einsum("bjkc,hc->bhjk", y, p["to_bias.weight"])
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(dtype).min)
        yr = y[:, rows]
        del y
        q, k, v, g = ((yr @ p[f"to_{n}.weight"].t()).unflatten(-1, (H, -1)) for n in ("query", "key", "value", "gate"))
        a = torch.einsum("brjhd,brkhd->brhjk", q * q.shape[-1] ** -0.5, k) + bias[:, None]
        o = torch.einsum("brhjk,brkhd->brjhd", a.softmax(-1), v)
        return (torch.sigmoid(g) * o).flatten(-2) @ p["to_out.weight"].t()


@pytest.mark.parametrize("train", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("length", [1536, 2048, 2560, 1000])
@pytest.mark.parametrize("name", ["C128", "C64-H4x16", "C256"])
def test_long_lengths(name, length, train):
    """The MiniWorld crop lengths beyond the registered 768 (and a ragged 1000): one sample, finite output and gradients, and pair
    rows 0-7 and the last 8 against the fp32 statements of those rows (the full statements would materialise [H, L, L, L] logits),
    within 1.25x the bf16 statements' own error."""
    cfg = CONFIGS[name]
    C = cfg[0]
    m = make(cfg)
    x, mask, dy = inputs(1, length, C, grad=train)
    y, dx, grads = run(m, x, mask, dy, train)
    assert torch.isfinite(y).all()
    if train:
        assert torch.isfinite(dx).all() and all(torch.isfinite(g).all() for g in grads)
    for rows in (slice(0, 8), slice(length - 8, length)):
        want = rows_reference(m, x, mask, rows)
        xr = x[:, rows]
        e = relative(y[:, rows].float() - xr.float(), want)
        e16 = relative((xr + rows_reference(m, x, mask, rows, torch.bfloat16)).float() - xr.float(), want)   # the bf16 module: y in bf16
        print(f"\n{name} L{length} rows {rows}: {e:.2e} (bf16 statements {e16:.2e})")
        assert e <= 1.25 * e16 + 1e-3, (e, e16)


@pytest.mark.parametrize("chunked", [False, True], ids=["one-launch", "chunks"])
@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("name", ["C128", "C256"])
def test_cuda_graph_capture(monkeypatch, name, grad, chunked):
    cfg = CONFIGS[name]
    m = make(cfg)
    x, mask, dy = inputs(5, 256, cfg[0])
    x.requires_grad_(grad)
    if chunked:
        monkeypatch.setattr(tri, "plan", lambda *a, **k: (2, False))
    params = [x, *m.parameters()] if grad else []

    def step():
        with torch.set_grad_enabled(grad):
            y = m(x, mask)
            return torch.autograd.grad(y, params, dy) if grad else (y,)

    eager = [t.clone() for t in step()]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[0], eager[0]) if not grad else True
    for a, b in zip(out, eager, strict=True):
        assert relative(a, b) < 1e-3


@pytest.mark.parametrize("chunked", [False, True], ids=["one-launch", "chunks"])
@pytest.mark.parametrize("length", [256, 300], ids=["L256", "L300-padded"])
def test_compiled_module(monkeypatch, chunked, length):
    """torch.compile of the module with B samples: the plan is a constant of the trace (shapes and the device's total memory); one
    graph (the launches are opaque ops)."""
    import torch._dynamo as dynamo

    dynamo.reset()
    if chunked:
        monkeypatch.setattr(tri, "plan", lambda *a, **k: (2, False))
    m = make(CONFIGS["C128"]).eval()
    mc = torch.compile(m, dynamic=False, fullgraph=True)
    x, mask, _ = inputs(5, length, 128)
    with torch.no_grad():
        eager = m(x, mask)
        y = mc(x, mask)
    assert torch.equal(y, eager)


def test_compiled_training_step():
    """torch.compile of a training step (wide path, dropout, ending node) matches eager: forward bitwise, gradients within rounding."""
    import torch._dynamo as dynamo

    dynamo.reset()
    cfg = CONFIGS["C256"]
    m = make(cfg, starting=False, p_drop=0.25)
    x, mask, dy = inputs(3, 256, cfg[0])
    FixedDrop(m, drop_scale(m, 3, 256, cfg[0]))
    eager = run(m, x, mask, dy, True)
    mc = torch.compile(m, dynamic=False, fullgraph=True)
    m.zero_grad(set_to_none=True)
    xi = x.detach().requires_grad_()
    y = mc.train()(xi, mask)
    y.backward(dy)
    assert torch.equal(y.detach(), eager[0])
    assert relative(xi.grad, eager[1]) < 1e-3
    for p, g in zip(m.parameters(), eager[2], strict=True):
        assert relative(p.grad.float(), g) < 1e-3

"""H100 AttentionPairBias in fp32 (TF32): ``kernels/augmented_attention/cuda/apb_tf32.py`` through the module, against an fp64 run of the same module.

The yardstick is the repo's: the engine's relative-Frobenius error against fp64 must not exceed 1.5x the error of the PyTorch module run in the
same precision (fp32 with TF32 matmuls), for the output and every gradient. The route is asserted to be taken (the tests would pass on the
PyTorch leg too, so a silent fallback is a failure); training is repeatable bit for bit; the module runs under a CUDA graph and torch.compile.
"""
import pytest
import torch

pytestmark = pytest.mark.gpu

HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
needs_hopper = pytest.mark.skipif(not HOPPER, reason="sm_90a kernels")

#: (n_head, d_single, d_pair, L)
CASES = [
    (8, 384, 128, 384),
    (8, 384, 128, 768),
    (16, 384, 128, 384),
    (8, 384, 256, 384),
    (12, 384, 128, 256),
]
IDS = [f"h{h}_p{p}_L{L}" for h, _, p, L in CASES]


def _rel(a, b):
    return float((a.double() - b).norm() / b.norm())


def _make(n_head, d_single, d_pair, gate_bias=0.0):
    from miniworld_engine.modules import AttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    def make(impl):
        return AttentionPairBias(d_single, d_pair, n_head, gate_bias=gate_bias, implementation=impl)

    torch.manual_seed(3)
    mw = make(IT.MINIWORLD).cuda()
    with torch.no_grad():
        for p in mw.parameters():
            p.normal_(0, p.shape[-1] ** -0.5) if p.dim() > 1 else p.normal_(0, 0.1)
        mw.ln_pair.weight.add_(1.0)
    pt = make(IT.PYTORCH).cuda()
    pt.load_state_dict(mw.state_dict())
    ref = make(IT.PYTORCH).cuda().double()
    ref.load_state_dict({k: v.double() for k, v in mw.state_dict().items()})
    return mw, pt, ref


def _inputs(L, d_single, d_pair, masked=True):
    g = torch.Generator(device="cuda").manual_seed(5)
    single = torch.randn(1, L, d_single, device="cuda", generator=g)
    pair = torch.randn(1, L, L, d_pair, device="cuda", generator=g)
    mask = (torch.rand(1, L, device="cuda", generator=g) > 0.2) if masked else None
    dy = torch.randn(1, L, d_single, device="cuda", generator=g)
    return single, pair, mask, dy


def _run(m, single, pair, mask, dy, dtype):
    s, p = (t.detach().to(dtype).clone().requires_grad_() for t in (single, pair))
    m.zero_grad()
    o = m(s, p, mask)
    o.backward(dy.to(o.dtype))
    named = dict(m.named_parameters())
    g = {"out": o, "dsingle": s.grad, "dpair": p.grad}
    for k, v in named.items():
        g["grad:" + k] = v.grad
    return {k: v.detach().double() for k, v in g.items()}


@pytest.fixture(autouse=True)
def _tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev


def _spy():
    from miniworld_engine.kernels.augmented_attention.cuda import apb_tf32

    taken = []
    orig = apb_tf32.attention_pair_tf32
    apb_tf32.attention_pair_tf32 = lambda *a, **k: (taken.append(1), orig(*a, **k))[1]
    return apb_tf32, orig, taken


@needs_hopper
@pytest.mark.parametrize(("n_head", "d_single", "d_pair", "L"), CASES, ids=IDS)
@pytest.mark.parametrize(("masked", "gate_bias"), [(True, 0.0), (False, 1.0)], ids=["mask", "gatebias"])
def test_route_is_taken_and_holds_the_yardstick(n_head, d_single, d_pair, L, masked, gate_bias):
    mw, pt, ref = _make(n_head, d_single, d_pair, gate_bias)
    single, pair, mask, dy = _inputs(L, d_single, d_pair, masked)
    mod, orig, taken = _spy()
    try:
        r = _run(ref, single, pair, mask, dy, torch.float64)
        e = _run(mw, single, pair, mask, dy, torch.float32)
    finally:
        mod.attention_pair_tf32 = orig
    b = _run(pt, single, pair, mask, dy, torch.float32)
    assert taken, "the CUDA core was not used"
    bad = [(k, _rel(e[k], r[k]), _rel(b[k], r[k])) for k in r if _rel(e[k], r[k]) > 1.5 * _rel(b[k], r[k])]
    print({k: (round(_rel(e[k], r[k]) / _rel(b[k], r[k]), 2)) for k in r})
    assert not bad, bad


@needs_hopper
@pytest.mark.parametrize(("n_head", "d_single", "d_pair", "L"), CASES[:1] + CASES[2:4], ids=[IDS[0], IDS[2], IDS[3]])
def test_inference_holds_the_yardstick(n_head, d_single, d_pair, L):
    mw, pt, ref = _make(n_head, d_single, d_pair)
    single, pair, mask, _ = _inputs(L, d_single, d_pair)
    mod, orig, taken = _spy()
    try:
        with torch.no_grad():
            e = mw(single, pair, mask)
            r = ref(single.double(), pair.double(), mask)
            b = pt(single, pair, mask)
    finally:
        mod.attention_pair_tf32 = orig
    assert taken
    assert _rel(e, r) <= 1.5 * _rel(b, r), (_rel(e, r), _rel(b, r))


@needs_hopper
@pytest.mark.parametrize(("n_head", "d_single", "d_pair", "L"), CASES[:1] + CASES[2:4], ids=[IDS[0], IDS[2], IDS[3]])
def test_training_step_is_bitwise_repeatable(n_head, d_single, d_pair, L):
    mw, _, _ = _make(n_head, d_single, d_pair)
    single, pair, mask, dy = _inputs(L, d_single, d_pair)
    a = _run(mw, single, pair, mask, dy, torch.float32)
    b = _run(mw, single, pair, mask, dy, torch.float32)
    for k in a:
        assert torch.equal(a[k], b[k]), k


@needs_hopper
def test_fully_masked_row_and_unsupported_calls_fall_back():
    """A ragged length (L % 128 != 0) and an fp32 call with TF32 off run the PyTorch reference, with a correct result."""
    mw, pt, _ = _make(8, 384, 128)
    single, pair, mask, _ = _inputs(200, 384, 128)
    with torch.no_grad():
        assert _rel(mw(single, pair, mask), pt(single, pair, mask).double()) < 1e-4
    prev_prec = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        single, pair, mask, _ = _inputs(384, 384, 128)
        mod, orig, taken = _spy()
        try:
            with torch.no_grad():
                mw(single, pair, mask)
        finally:
            mod.attention_pair_tf32 = orig
        assert not taken
    finally:
        torch.set_float32_matmul_precision(prev_prec)


@needs_hopper
@pytest.mark.parametrize("train", [False, True])
def test_cuda_graph_replay_matches_eager(train):
    mw, _, _ = _make(8, 384, 128)
    single, pair, mask, dy = _inputs(384, 384, 128)
    s, p = single.clone().requires_grad_(train), pair.clone().requires_grad_(train)

    def step():
        if train:
            o = mw(s, p, mask)
            o.backward(dy)
            return o
        with torch.no_grad():
            return mw(s, p, mask)

    eager = step().detach().clone()
    if train:
        mw.zero_grad(set_to_none=False)
        s.grad = p.grad = None
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(st)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.detach(), eager)


@needs_hopper
def test_runs_under_torch_compile():
    """The core is two opaque ops: a compiled module still launches the hand-CUDA TF32 core and matches the eager route."""
    from torch.profiler import ProfilerActivity, profile

    mw, _, _ = _make(8, 384, 128)
    single, pair, mask, dy = _inputs(384, 384, 128)
    s, p = single.clone().requires_grad_(), pair.clone().requires_grad_()

    def run(m):
        o = m(s, p, mask)
        o.backward(dy)
        return o.detach(), s.grad.clone()

    eager = run(mw)
    s.grad = None
    mc = torch.compile(mw, dynamic=False)
    run(mc)
    s.grad = None
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = run(mc)
        torch.cuda.synchronize()
    names = " ".join(e.key for e in prof.key_averages())
    assert "attn_tf32_fwd_kernel" in names, "the compiled module did not launch the CUDA core"
    assert _rel(out[0], eager[0].double()) < 3e-3
    assert _rel(out[1], eager[1].double()) < 3e-3

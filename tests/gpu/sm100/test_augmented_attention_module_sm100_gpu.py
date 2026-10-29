"""AugmentedAttentionPairBias's B200 module route (kernels/augmented_attention/cuda/sm100/module_route.py): the whole module after the
AdaLN as one node -- packed q|k|v|g GEMM, the pair pass, the tcgen05 cores, the gates / residual / gradient adds in the ``aa_rows.cu``
row kernels -- against an fp64 run of the module, with the PyTorch module in the same precision as the yardstick (1.5x its error, as
the pair-fused core's tests). Covers what the per-call core routes did not serve: ``extra_pair_bias``, L not a multiple of 128 (and
not of 16), key masks per sample (the cores built -DKMASK=1), odd A in inference (attn_inf / attn_inf_tf32, no pad sample), bitwise
repeatable gradients, CUDA-graph replay and torch.compile.
"""
import pytest
import torch

CUDA = torch.cuda.is_available()
BLACKWELL = CUDA and torch.cuda.get_device_capability() == (10, 0)
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not BLACKWELL, reason="the kernels are sm_100a only")]

PARAMS = ("to_bias.weight", "ln_pair.weight", "to_query.weight", "to_query.bias", "to_key.weight", "to_value.weight", "to_gate.weight",
          "to_out.weight", "to_scale.weight", "to_scale.bias", "ada_ln_in.to_scale.weight", "ada_ln_in.ln_cond.weight")


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _modules(d, dc, dp, H, qk=False):
    from miniworld_engine.modules import AugmentedAttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    torch.manual_seed(3)
    mw = AugmentedAttentionPairBias(d, dc, dp, H, use_qk_norm=qk, implementation=IT.MINIWORLD).cuda()
    with torch.no_grad():
        for p in mw.parameters():
            p.normal_(0, p.shape[-1] ** -0.5) if p.dim() > 1 else p.normal_(0, 0.1)
        mw.ln_pair.weight.add_(1.0)
        if qk:
            mw.norm_query.weight.add_(1.0)
            mw.norm_key.weight.add_(1.0)
    pt = AugmentedAttentionPairBias(d, dc, dp, H, use_qk_norm=qk, implementation=IT.PYTORCH).cuda()
    pt.load_state_dict(mw.state_dict())
    ref = AugmentedAttentionPairBias(d, dc, dp, H, use_qk_norm=qk, implementation=IT.PYTORCH).cuda().double()
    ref.load_state_dict({k: v.double() for k, v in mw.state_dict().items()})
    return mw, pt, ref


def _case(A, L, d, dc, dp, masked, extra, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed + 1000 * A + L)
    single = torch.randn(A, 1, L, d, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, dc, device="cuda", generator=g)
    pair = torch.randn(1, L, L, dp, device="cuda", generator=g)
    if masked == "ps":   # a key mask per sample (materialized [A, 1, L]: the cores' -DKMASK path)
        mask = torch.rand(A, 1, L, device="cuda", generator=g) > 0.2
    else:
        mask = (torch.rand(1, L, device="cuda", generator=g) > 0.1) if masked else None
    x = torch.randn(L, L, device="cuda", generator=g) if extra else None
    dy = torch.randn(A, 1, L, d, device="cuda", generator=g)
    return single, cond, pair, mask, x, dy


def _run(m, case, dtype, autocast, train, residual=True):
    single, cond, pair, mask, extra, dy = case
    s, c, p = (t.detach().to(dtype).clone().requires_grad_(train) for t in (single, cond, pair))
    x = None if extra is None else extra.detach().to(dtype).clone().requires_grad_(train)
    m.zero_grad(set_to_none=True)
    kw = {} if x is None else {"extra_pair_bias": x}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast), torch.set_grad_enabled(train):
        o = (m if residual else m.delta)(s, c, p, mask, **kw)
    g = {"out": o}
    if train:
        o.backward(dy.to(o.dtype))
        named = dict(m.named_parameters())
        g.update({"dsingle": s.grad, "dcond": c.grad, "dpair": p.grad, **{n: named[n].grad for n in PARAMS if n in named}})
        if "norm_query.weight" in named:
            g.update({"norm_query.weight": named["norm_query.weight"].grad, "norm_key.weight": named["norm_key.weight"].grad})
        if x is not None:
            g["dextra"] = x.grad
    return {k: v.detach().double() for k, v in g.items()}


def _spied(fn):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import module_route

    taken = []
    orig = module_route.module_update
    module_route.module_update = lambda *a, **k: (taken.append(1), orig(*a, **k))[1]  # ty: ignore[invalid-assignment] -- spy
    try:
        out = fn()
    finally:
        module_route.module_update = orig
    return out, bool(taken)


def _check(d, dc, dp, H, prec, train, A, L, masked, extra, residual=True, qk=False):
    torch.backends.cuda.matmul.allow_tf32 = True
    mw, pt, ref = _modules(d, dc, dp, H, qk)
    case = _case(A, L, d, dc, dp, masked, extra)
    dt, ac = (torch.bfloat16, True) if prec == "bf16" else (torch.float32, False)
    e, taken = _spied(lambda: _run(mw, case, dt, ac, train, residual))
    assert taken, "the module route did not serve the call"
    r = _run(ref, case, torch.float64, False, train, residual)
    b = _run(pt, case, dt, ac, train, residual)
    for k in r:
        assert _rel(e[k], r[k]) <= 1.5 * _rel(b[k], r[k]) + 1e-6, (k, _rel(e[k], r[k]), _rel(b[k], r[k]))


# (A, L, key mask, extra_pair_bias): odd A, L % 128 != 0 (200, 136), L % 16 != 0 (100, 37), the extra term with a gradient, a key mask
# per sample ("ps")
CASES = [(3, 256, True, False), (4, 256, False, True), (3, 200, True, True), (2, 384, True, False), (5, 136, False, False),
         (3, 100, True, True), (1, 37, True, False), (3, 256, "ps", False), (4, 200, "ps", True), (5, 100, "ps", False)]


@pytest.mark.parametrize("prec", ["bf16", "fp32"])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize(("A", "L", "masked", "extra"), CASES)
def test_module_route_holds_the_yardstick(prec, train, A, L, masked, extra):
    _check(768, 384, 128, 16, prec, train, A, L, masked, extra)


@pytest.mark.parametrize(("d", "dc", "dp", "H"), [(768, 384, 128, 24), (768, 384, 128, 12), (1024, 384, 128, 16), (768, 768, 256, 16)])
@pytest.mark.parametrize("prec", ["bf16", "fp32"])
@pytest.mark.parametrize("train", [True, False])
def test_module_route_layouts(d, dc, dp, H, prec, train):
    if prec == "fp32" and dp != 128:
        pytest.skip("the fp32 pair pass is d_pair 128")
    _check(d, dc, dp, H, prec, train, 3, 200, True, True)
    _check(d, dc, dp, H, prec, train, 3, 136, "ps", False)


@pytest.mark.parametrize("prec", ["bf16", "fp32"])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize(("A", "L", "masked", "extra"), [(3, 256, True, False), (3, 200, "ps", True), (5, 100, True, True)])
def test_module_route_qk_norm(prec, train, A, L, masked, extra):
    """QK-norm (per-head RMSNorm of q and k) inside the route: same yardstick, the norm weights' gradients included."""
    _check(768, 384, 128, 16, prec, train, A, L, masked, extra, qk=True)


@pytest.mark.parametrize("prec", ["bf16", "fp32"])
def test_module_route_delta(prec):
    """``delta`` (no residual) through the route, same yardstick."""
    _check(768, 384, 128, 16, prec, True, 3, 200, True, True, residual=False)


@pytest.mark.parametrize("prec", ["bf16", "fp32"])
@pytest.mark.parametrize(("A", "L", "masked"), [(48, 384, True), (3, 200, True), (6, 256, "ps")])
def test_module_route_training_step_is_bitwise_repeatable(prec, A, L, masked):
    torch.backends.cuda.matmul.allow_tf32 = True
    mw, _, _ = _modules(768, 384, 128, 16)
    case = _case(A, L, 768, 384, 128, masked, True)
    dt, ac = (torch.bfloat16, True) if prec == "bf16" else (torch.float32, False)
    a = _run(mw, case, dt, ac, True)
    b = _run(mw, case, dt, ac, True)
    for k in a:
        assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("prec", ["bf16", "fp32"])
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("masked", [True, "ps"])
def test_module_route_graph_replay_and_compile(prec, train, masked):
    """CUDA-graph capture replays the eager result on new inputs, and torch.compile keeps the route (opaque ops) with the eager numbers
    (odd A, L 200, a key mask shared or per sample, and the extra term)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    mw, _, _ = _modules(768, 384, 128, 16)
    single, cond, pair, mask, extra, dy = _case(5, 200, 768, 384, 128, masked, True)
    dt, ac = (torch.bfloat16, True) if prec == "bf16" else (torch.float32, False)
    s, c, p, x = (t.to(dt).requires_grad_(train) for t in (single, cond, pair, extra))

    def step(m):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac), torch.set_grad_enabled(train):
            o = m(s, c, p, mask, extra_pair_bias=x)
        if train:
            o.backward(dy.to(o.dtype))
        return o.detach()

    eager = step(mw).clone()
    comp, taken = _spied(lambda: step(torch.compile(mw)))
    assert taken, "the compiled module left the module route"
    assert _rel(comp.double(), eager.double()) < 1e-2, _rel(comp.double(), eager.double())
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            step(mw)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cap = step(mw)
    with torch.no_grad():
        s.mul_(0.5)
        p.mul_(1.5)
        x.mul_(-1.0)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(cap, step(mw))

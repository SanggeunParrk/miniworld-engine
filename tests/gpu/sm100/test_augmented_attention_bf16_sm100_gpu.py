"""The sm_100 bf16 pair-bias attention core (kernels/augmented_attention/cuda/sm100) against an fp64 truth, and its
module wiring.

As for the sm_90 core: the kernels round their operands to bf16, so the bar is "no worse than the same math on the
bf16-rounded inputs", and the gate matters as much as the numbers (the tile shapes and the sample pairing are literals).
"""
import math

import pytest
import torch

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
BLACKWELL = CUDA and torch.cuda.get_device_capability() == (10, 0)
needs_blackwell = pytest.mark.skipif(not BLACKWELL, reason="the kernels are sm_100a only")


def _inputs(A, L, seed=0, bias_scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(A, 1, L, 16, 48, device="cuda", generator=g) for _ in range(3))
    bias = torch.randn(1, L, L, 16, device="cuda", generator=g) * bias_scale
    return q, k, v, bias


def _truth(q, k, v, bias, do):
    """fp64 autograd: (O, dq, dk, dv, dbias) in the op's layouts."""
    leaves = [t.detach().double().requires_grad_() for t in (q, k, v, bias)]
    qd, kd, vd, bd = leaves
    s = torch.einsum("aihd,ajhd->ahij", qd[:, 0], kd[:, 0]) / math.sqrt(48) + bd[0].permute(2, 0, 1)[None]
    o = torch.einsum("ahij,ajhd->aihd", torch.softmax(s, -1), vd[:, 0])[:, None]
    o.backward(do.double())
    return (o.detach(), *(t.grad for t in leaves))


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_gate_rejects_what_the_tiles_cannot_take():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, _, _, bias = _inputs(4, 128)
    assert sm100.supported(q, bias) is BLACKWELL
    assert sm100.supported(q, bias.permute(3, 0, 1, 2).contiguous(), bias_head_major=True) is BLACKWELL
    assert not sm100.supported(q, bias.permute(3, 0, 1, 2).contiguous())               # layout flag disagrees
    assert not sm100.supported(q[:3], bias)                                            # odd A: samples run in pairs
    assert not sm100.supported(q, bias, mask=torch.ones(4, 1, 128, device="cuda", dtype=torch.bool))  # no key mask
    q96, _, _, b96 = _inputs(4, 96)
    assert not sm100.supported(q96, b96)                                              # L not a multiple of 128
    assert not sm100.supported(q[..., :32], bias)                                     # head dim 32
    assert not sm100.supported(torch.randn(4, 2, 128, 16, 48, device="cuda"), bias)   # B == 2
    assert not sm100.supported(q.cpu(), bias.cpu())


@needs_blackwell
@pytest.mark.parametrize(("A", "L", "bias_scale", "head_major"), [
    (48, 384, 1.0, False),     # the token DiT's training shape
    (2, 768, 1.0, True),       # one sample pair, head-major bias (as the DiT hoists it)
    (6, 256, 4.0, False),      # wide logits
    (4, 128, 1.0, True),       # one query tile per head
])
def test_matches_fp64_at_the_bf16_input_floor(A, L, bias_scale, head_major):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, k, v, bias = _inputs(A, L, seed=A * 1000 + L, bias_scale=bias_scale)
    do = torch.randn(A, 1, L, 16, 48, device="cuda")
    truth = _truth(q, k, v, bias, do)
    r = lambda t: t.bfloat16().float()
    floor = _truth(r(q), r(k), r(v), r(bias), r(do))

    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    b = (bias.permute(3, 0, 1, 2).contiguous() if head_major else bias.clone()).requires_grad_()
    assert sm100.available(leaves[0], b, bias_head_major=head_major)
    o = sm100.augmented_attention_bf16_sm100(leaves[0], leaves[1], leaves[2], b, bias_head_major=head_major)
    o.backward(do)
    db = b.grad.permute(1, 2, 3, 0) if head_major else b.grad
    got = (o.detach(), leaves[0].grad, leaves[1].grad, leaves[2].grad, db)
    for name, x, t, f in zip(("O", "dq", "dk", "dv", "dbias"), got, truth, floor, strict=False):
        e, ef = _rel(x, t), _rel(f, t)
        assert math.isfinite(e), f"{name}: not finite"
        assert e < 1.3 * ef + 1e-4, f"{name}: {e:.2e} against a floor of {ef:.2e}"


@needs_blackwell
def test_graph_capture_replays_the_same_forward():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, k, v, bias = _inputs(4, 256)
    with torch.no_grad():
        eager = sm100.augmented_attention_bf16_sm100(q, k, v, bias)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cap = sm100.augmented_attention_bf16_sm100(q, k, v, bias)
        q.mul_(0.5)
        g.replay()
        assert torch.equal(cap, sm100.augmented_attention_bf16_sm100(q, k, v, bias))
        assert not torch.equal(cap, eager)


@needs_blackwell
def test_module_bf16_core_takes_the_kernels_and_keeps_the_triton_error():
    """AugmentedAttentionPairBias with compute_dtype=bf16 on a supported shape runs the sm_100 kernels (since 2026-10-29 the
    pair-fused core, ``sm100.fused_pair``, which serves this call before the plain bf16 core); its error against the fp32 PyTorch
    module is no worse than the bf16 Triton core's (the switch is off in the second run)."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import fused_pair
    from miniworld_engine.modules import AugmentedAttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType

    torch.manual_seed(0)
    A, L = 4, 256
    ref = AugmentedAttentionPairBias(768, 384, 128, 16, use_qk_norm=True, implementation=ImplementationType.PYTORCH).cuda()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(std=p.shape[-1] ** -0.5 if p.dim() > 1 else 0.3).add_(1.0 if p.dim() == 1 else 0.0)
    eng = AugmentedAttentionPairBias(768, 384, 128, 16, use_qk_norm=True,
                                     implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    single = torch.randn(A, 1, L, 768, device="cuda")
    cond = torch.randn(A, 1, L, 384, device="cuda")
    pair = torch.randn(1, L, L, 128, device="cuda")
    w = torch.randn(A, 1, L, 768, device="cuda")

    def run(m, **kw):
        ins = [t.clone().requires_grad_() for t in (single, cond, pair)]
        out = m(*ins, None, **kw)
        (out * w).sum().backward()
        grads = [t.grad for t in ins]
        m.zero_grad(set_to_none=True)
        return [out.detach(), *grads]

    truth = run(ref)
    calls = []
    orig, orig_f = sm100.augmented_attention_bf16_sm100, fused_pair.attention_pair_sm100
    sm100.augmented_attention_bf16_sm100 = lambda *a, **k: calls.append(1) or orig(*a, **k)  # ty: ignore[invalid-assignment] -- deliberate spy
    fused_pair.attention_pair_sm100 = lambda *a, **k: calls.append(2) or orig_f(*a, **k)  # ty: ignore[invalid-assignment] -- deliberate spy
    try:
        got = run(eng, compute_dtype=torch.bfloat16)
    finally:
        sm100.augmented_attention_bf16_sm100, fused_pair.attention_pair_sm100 = orig, orig_f
    assert calls, "the bf16 core did not take the sm_100 kernels"
    prev = settings.configure(augmented_attention_bf16_sm100=False)
    try:
        tri = run(eng, compute_dtype=torch.bfloat16)
    finally:
        settings.configure(augmented_attention_bf16_sm100=prev.augmented_attention_bf16_sm100)
    for name, a, b, t in zip(("out", "dsingle", "dcond", "dpair"), got, tri, truth, strict=False):
        es, et = _rel(a, t), _rel(b, t)
        assert es < 1.5 * et + 1e-4, f"{name}: sm100 {es:.2e} vs triton bf16 {et:.2e}"


@needs_blackwell
@pytest.mark.parametrize(("dtype", "tol"), [(torch.bfloat16, 1.2e-2), (torch.float32, 2e-3)])
@pytest.mark.parametrize(("S", "L"), [(5, 384), (2, 256), (3, 128), (3, 200), (2, 520), (5, 136)])
def test_gated_inference_core_matches_fp64(dtype, tol, S, L):
    """The step's gated core alone (bf16 attn_inf on q | k | v | g / fp32 TF32 attn_inf_tf32 on q | k | g and v^T) against
    fp64: o = sigmoid(g) softmax_2(q k^T + bias) v written over q, logits in exp2 units; odd S exercises the clamped partner
    sample, L not a multiple of 128 the zero-filled / clipped tile tails and the masked keys of the last block."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    g = torch.Generator(device="cuda").manual_seed(S * 100 + L)
    M, D = S * L, 768
    qkvg = torch.randn(M, 4 * D, device="cuda", generator=g)
    qkvg[:, :D] *= 48 ** -0.5 * 1.4426950408889634              # the runner folds sm_scale log2 e into Wq
    bias = torch.randn(2 * 16, L, L, device="cuda", generator=g) * 1.4426950408889634
    ref_in = qkvg.to(dtype).double()
    q, k, v, gt = (ref_in[:, i * D:(i + 1) * D].view(S, L, 16, 48) for i in range(4))
    b = bias.to(dtype).double()[16:32]                           # block 1 of a two-block hoist
    s = torch.einsum("slhd,sjhd->shlj", q, k) + b[None]
    p = torch.softmax(s * math.log(2.0), -1)
    want = (torch.einsum("shlj,sjhd->slhd", p, v) * torch.sigmoid(gt)).reshape(M, D)
    core = sm100.GatedInferenceCore(torch.cuda.current_device(), dtype)
    x = qkvg.to(dtype)
    if dtype is torch.float32:                                   # the TF32 core: q | k | g and v^T
        buf = torch.cat([x[:, :2 * D], x[:, 3 * D:]], 1).contiguous()
        vt = x[:, 2 * D:3 * D].t().contiguous()
        core(buf, bias.to(dtype).contiguous(), 1, S, vt)
        assert torch.equal(vt, x[:, 2 * D:3 * D].t()), "v^T is read only"
    else:
        buf = x.contiguous()
        core(buf, bias.to(dtype).contiguous(), 1, S)
    torch.cuda.synchronize()
    e = float((buf[:, :D].double() - want).norm() / want.norm())
    assert math.isfinite(e), f"{dtype} S{S} L{L}: {e}"
    assert e < tol, f"{dtype} S{S} L{L}: {e:.2e}"
    rest = torch.cat([x[:, D:2 * D], x[:, 3 * D:]], 1) if dtype is torch.float32 else x[:, D:]
    assert torch.equal(buf[:, D:], rest), "only q's columns may be written"


# ------------------------------------------------------------------------------------------------ the pair-fused core (fused_pair.py)
def _pair_case(A, L, H, DH, dp, dtype, masked, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v, do = (torch.randn(A, 1, L, H, DH, device="cuda", generator=g) for _ in range(4))
    pair = torch.randn(1, L, L, dp, device="cuda", generator=g) * 1.5 + 0.3
    lnw = 1.0 + 0.3 * torch.randn(dp, device="cuda", generator=g)
    wb = torch.randn(H, dp, device="cuda", generator=g) * dp ** -0.5 * 2.0
    mask = (torch.rand(L, device="cuda", generator=g) > 0.2) if masked else None
    q, k, v, pair = (t.to(dtype) for t in (q, k, v, pair))
    return q, k, v, pair, lnw, wb, mask, do


def _pair_truth(q, k, v, pair, lnw, wb, mask, do, rnd=None, eps=1e-5):
    """fp64 autograd of softmax(q k^T / sqrt d + LN(pair) (wb lnw)^T) v: (O, dq, dk, dv, dpair, dlnw, dwb). ``rnd``: rounding applied
    (straight through) to Wf and to the bias, as the kernels round them (the floor of the same math on rounded operands)."""
    st = (lambda t: t + (rnd(t) - t).detach()) if rnd else (lambda t: t)
    leaves = [t.detach().double().requires_grad_() for t in (q, k, v, pair, lnw, wb)]
    qd, kd, vd, pd, lw, wbd = leaves
    DH = q.shape[-1]
    mu = pd.mean(-1, keepdim=True)
    xh = (pd - mu) / torch.sqrt(((pd - mu) ** 2).mean(-1, keepdim=True) + eps)
    bias = st(torch.einsum("bijc,hc->bhij", xh, st(wbd * lw[None])))[0]           # [H, L, L]
    s = torch.einsum("aihd,ajhd->ahij", qd[:, 0], kd[:, 0]) / DH ** 0.5 + bias[None]
    if mask is not None:
        s = s.masked_fill(~mask[None, None, None, :], torch.finfo(torch.float64).min)
    o = torch.einsum("ahij,ajhd->aihd", torch.softmax(s, -1), vd[:, 0])[:, None]
    o.backward(do.double())
    return (o.detach(), *(t.grad for t in leaves))


PAIR_LAYOUTS = [(16, 48, 128), (24, 32, 128), (12, 64, 128), (16, 64, 128), (16, 48, 256)]


@needs_blackwell
@pytest.mark.parametrize(("H", "DH", "dp"), PAIR_LAYOUTS)
@pytest.mark.parametrize(("A", "L", "masked"), [(3, 256, True), (4, 128, False)])
def test_pair_fused_core_matches_fp64_at_the_bf16_floor(H, DH, dp, A, L, masked):
    """The pair-fused core (pair LayerNorm + head projection + the deterministic bf16 core; odd A runs a zero pad sample) against fp64,
    no worse than the same math on the bf16-rounded operands (q, k, v, pair, dO, Wf and the bias rounded once, as the kernels do)."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import fused_pair

    q, k, v, pair, lnw, wb, mask, do = _pair_case(A, L, H, DH, dp, torch.bfloat16, masked, H * 100 + DH + L + A)
    truth = _pair_truth(q, k, v, pair, lnw, wb, mask, do)
    r = lambda t: t.bfloat16().double()  # noqa: E731
    floor = _pair_truth(q, k, v, pair, lnw, wb, mask, r(do), rnd=r)
    leaves = [t.clone().requires_grad_() for t in (q, k, v, pair, lnw, wb)]
    assert fused_pair.available(leaves[0], leaves[3], leaves[4], leaves[5], mask)
    o = fused_pair.attention_pair_sm100(*leaves, mask)
    o.backward(do)
    got = (o.detach(), *(t.grad for t in leaves))
    for name, x, t, f in zip(("O", "dq", "dk", "dv", "dpair", "dlnw", "dwb"), got, truth, floor, strict=True):
        e, ef = _rel(x, t), _rel(f, t)
        assert math.isfinite(e), f"{name}: not finite"
        assert e < 1.3 * ef + 1e-4, f"{name}: {e:.2e} against a floor of {ef:.2e}"


@needs_blackwell
@pytest.mark.parametrize(("H", "DH"), [(16, 48), (12, 64)])
def test_pair_fused_backward_is_bitwise_repeatable(H, DH):
    """A = 48 (the training shape): dq, dk, dv, dpair, dlnw, dwb bitwise equal over repeated backward passes (the -DDET / -DDQPART
    builds; the plain cores' two MMA issuers and dQ reductions are not repeatable at this shape)."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import fused_pair

    q, k, v, pair, lnw, wb, mask, do = _pair_case(48, 384, H, DH, 128, torch.bfloat16, True, 11)
    runs = []
    for _ in range(3):
        leaves = [t.clone().requires_grad_() for t in (q, k, v, pair, lnw, wb)]
        fused_pair.attention_pair_sm100(*leaves, mask).backward(do)
        runs.append([t.grad for t in leaves])
    for a, b in zip(runs, runs[1:]):
        for name, x, y in zip(("dq", "dk", "dv", "dpair", "dlnw", "dwb"), a, b, strict=True):
            assert torch.equal(x, y), f"{name} differs between runs"


def _aa_modules(d, dc, dp, H, qk=False):
    from miniworld_engine.modules import AugmentedAttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    torch.manual_seed(3)
    mw = AugmentedAttentionPairBias(d, dc, dp, H, use_qk_norm=qk, implementation=IT.MINIWORLD).cuda()
    with torch.no_grad():
        for p in mw.parameters():
            p.normal_(0, p.shape[-1] ** -0.5) if p.dim() > 1 else p.normal_(0, 0.1)
        mw.ln_pair.weight.add_(1.0)
    pt = AugmentedAttentionPairBias(d, dc, dp, H, use_qk_norm=qk, implementation=IT.PYTORCH).cuda()
    pt.load_state_dict(mw.state_dict())
    ref = AugmentedAttentionPairBias(d, dc, dp, H, use_qk_norm=qk, implementation=IT.PYTORCH).cuda().double()
    ref.load_state_dict({k: v.double() for k, v in mw.state_dict().items()})
    return mw, pt, ref


def _aa_run(m, single, cond, pair, mask, dy, dtype, autocast):
    s, c, p = (t.detach().to(dtype).clone().requires_grad_() for t in (single, cond, pair))
    m.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        o = m.delta(s, c, p, mask)
    o.backward(dy.to(o.dtype))
    named = dict(m.named_parameters())
    g = {"out": o, "dsingle": s.grad, "dcond": c.grad, "dpair": p.grad, "to_bias.weight": named["to_bias.weight"].grad,
         "ln_pair.weight": named["ln_pair.weight"].grad, "to_query.weight": named["to_query.weight"].grad,
         "to_key.weight": named["to_key.weight"].grad}
    return {k: v.detach().double() for k, v in g.items()}


def _aa_case(A, L, d, dc, dp, pair_dtype, seed=5):
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(A, 1, L, d, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, dc, device="cuda", generator=g)
    pair = torch.randn(1, L, L, dp, device="cuda", generator=g).to(pair_dtype)
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.1
    dy = torch.randn(A, 1, L, d, device="cuda", generator=g)
    return single, cond, pair, mask, dy


AA_LAYOUTS = [(768, 384, 128, 16), (768, 384, 128, 24), (768, 384, 128, 12), (1024, 384, 128, 16), (768, 768, 256, 16)]


def _route_spy(qk):
    """(the module attribute the call must reach, its name): the module route (``sm100.module_route``, the whole module as one node
    since 2026-10-29, with or without QK-norm). The pair-fused core (``sm100.fused_pair``) keeps the calls with a ``compute_dtype``
    other than the activations' precision (test_module_bf16_core_takes_the_kernels_and_keeps_the_triton_error)."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import module_route

    return module_route, "module_update"


@needs_blackwell
@pytest.mark.parametrize(("d", "dc", "dp", "H"), AA_LAYOUTS)
@pytest.mark.parametrize("pair_fp32", [False, True])
@pytest.mark.parametrize("qk", [False, True])
def test_module_bf16_takes_the_pair_fused_core_and_holds_the_yardstick(d, dc, dp, H, pair_fp32, qk):
    """AugmentedAttentionPairBias (MINIWORLD) under bf16 autocast on B200 runs the sm_100a module route, QK-norm on or off (bf16 inputs; or fp32 activations, whose fp32 pair takes the TF32 pair pass) and its error against an
    fp64 run of the module is within 1.5x the PyTorch module's in the same precision (out, every input gradient, the pair-side and
    q / k weight gradients)."""
    if pair_fp32 and dp != 128:
        pytest.skip("the fp32 pair pass is d_pair 128")
    mw, pt, ref = _aa_modules(d, dc, dp, H, qk)
    dt = torch.float32 if pair_fp32 else torch.bfloat16
    single, cond, pair, mask, dy = _aa_case(3, 256, d, dc, dp, dt)
    taken = []
    mod, name = _route_spy(qk)
    orig = getattr(mod, name)
    setattr(mod, name, lambda *a, **k: (taken.append(1), orig(*a, **k))[1])
    try:
        e = _aa_run(mw, single, cond, pair, mask, dy, dt, True)
    finally:
        setattr(mod, name, orig)
    r = _aa_run(ref, single, cond, pair, mask, dy, torch.float64, False)
    b = _aa_run(pt, single, cond, pair, mask, dy, dt, True)
    assert taken, f"the sm100 {name} route was not used"
    for k in r:
        assert _rel(e[k], r[k]) <= 1.5 * _rel(b[k], r[k]) + 1e-6, (k, _rel(e[k], r[k]), _rel(b[k], r[k]))


@needs_blackwell
@pytest.mark.parametrize("qk", [False, True])
def test_module_bf16_training_step_is_bitwise_repeatable(qk):
    mw, _, _ = _aa_modules(768, 384, 128, 16, qk)
    single, cond, pair, mask, dy = _aa_case(48, 384, 768, 384, 128, torch.bfloat16)
    a = _aa_run(mw, single, cond, pair, mask, dy, torch.bfloat16, True)
    b = _aa_run(mw, single, cond, pair, mask, dy, torch.bfloat16, True)
    for k in a:
        assert torch.equal(a[k], b[k]), k


@needs_blackwell
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("qk", [False, True])
def test_module_pair_fused_graph_replay_and_compile(train, qk):
    """CUDA-graph capture replays the eager result on new inputs, and torch.compile keeps the route (opaque ops) with the eager
    numbers (bf16 autocast, A = 5, L 256; the module route, QK-norm on or off)."""
    mw, _, _ = _aa_modules(768, 384, 128, 16, qk)
    single, cond, pair, mask, dy = _aa_case(5, 256, 768, 384, 128, torch.bfloat16)
    s, c, p = (t.to(torch.bfloat16).requires_grad_(train) for t in (single, cond, pair))

    def step(m):
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(train):
            o = m(s, c, p, mask)
        if train:
            o.backward(dy.to(o.dtype))
        return o.detach()

    for t in (s, c, p):
        t.grad = None
    eager = step(mw).clone()
    taken = []
    mod, name = _route_spy(qk)
    orig = getattr(mod, name)
    setattr(mod, name, lambda *a, **k: (taken.append(1), orig(*a, **k))[1])
    try:
        comp = step(torch.compile(mw))
    finally:
        setattr(mod, name, orig)
    assert taken, f"the compiled module left the {name} route"
    assert _rel(comp, eager) < 1e-2, _rel(comp, eager)
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
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(cap, step(mw))

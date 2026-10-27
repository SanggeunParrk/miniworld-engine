"""H100 pair-fused attention (kernels/augmented_attention/cuda/fused_pair*.py) and the CUDA adaLN rows (kernels/adaln/cuda), through the
modules that route to them, against an fp64 PyTorch run of the same module.

The yardstick is the repo's: the engine's relative-Frobenius error against fp64 must not exceed 1.5x the error of the PyTorch module run in
the same precision (bf16 under autocast, or fp32 with TF32 matmuls). The routes are asserted to be taken (the tests would pass on the PyTorch
leg too, so a silent fallback is a failure), and the bf16 backward is repeatable bit for bit.
"""
import pytest
import torch

pytestmark = pytest.mark.gpu

HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
needs_hopper = pytest.mark.skipif(not HOPPER, reason="sm_90a kernels")


def _rel(a, b):
    return float((a.double() - b).norm() / b.norm())


def _modules(kind):
    from miniworld_engine.modules import AttentionPairBias, AugmentedAttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    def make(impl):
        if kind == "token256":   # the ESMFold2 token DiT: cond 768, d_pair 256
            return AugmentedAttentionPairBias(768, 768, 256, 16, implementation=impl)
        if kind == "token":
            return AugmentedAttentionPairBias(768, 384, 128, 16, implementation=impl)
        if kind == "atom":
            return AugmentedAttentionPairBias(128, 128, 16, 4, implementation=impl)
        return AttentionPairBias(384, 256, 8, implementation=impl)

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


def _run(m, kind, single, cond, pair, mask, dy, dtype, autocast):
    s, c, p = (t.detach().to(dtype).clone().requires_grad_() for t in (single, cond, pair))
    m.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        o = m(s, p, mask) if kind == "apb" else m.delta(s, c, p, mask)
    o.backward(dy.to(o.dtype))
    named = dict(m.named_parameters())
    g = {"out": o, "dsingle": s.grad, "dpair": p.grad, "to_bias.weight": named["to_bias.weight"].grad,
         "ln_pair.weight": named["ln_pair.weight"].grad, "to_query.weight": named["to_query.weight"].grad}
    if kind != "apb":
        g["dcond"] = c.grad
    return {k: v.detach().double() for k, v in g.items()}


def _case(kind, L, A, fp32=False):
    n = 8 * L if kind == "atom" else L
    dS, dC, dP = {"token": (768, 384, 128), "token256": (768, 768, 256), "atom": (128, 128, 16), "apb": (384, 384, 256)}[kind]
    a = 1 if kind == "apb" else A
    shape = (1, n, dS) if kind == "apb" else (a, 1, n, dS)
    cshape = (a, 1, n, dC)
    dt = torch.float32 if fp32 else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(5)
    single = torch.randn(*shape, device="cuda", generator=g).to(dt)
    cond = torch.randn(*cshape, device="cuda", generator=g).to(dt)
    pair = torch.randn(1, n, n, dP, device="cuda", generator=g).to(dt)
    mask = torch.rand(1, n, device="cuda", generator=g) > 0.1
    dy = torch.randn(*shape, device="cuda", generator=g)
    return single, cond, pair, mask, dy, dt


@needs_hopper
@pytest.mark.parametrize(("kind", "L", "A"), [("token", 384, 3), ("token", 384, 5), ("atom", 128, 3), ("apb", 384, 1)])
def test_bf16_routes_are_taken_and_hold_the_yardstick(kind, L, A):
    from miniworld_engine.kernels.augmented_attention.cuda import fused_pair

    mw, pt, ref = _modules(kind)
    single, cond, pair, mask, dy, dt = _case(kind, L, A)
    taken = []
    orig = fused_pair.attention_pair_fused
    fused_pair.attention_pair_fused = lambda *a, **k: (taken.append(1), orig(*a, **k))[1]
    try:
        r = _run(ref, kind, single, cond, pair, mask, dy, torch.float64, False)
        e = _run(mw, kind, single, cond, pair, mask, dy, dt, True)
    finally:
        fused_pair.attention_pair_fused = orig
    b = _run(pt, kind, single, cond, pair, mask, dy, dt, True)
    assert taken, "the pair-fused CUDA core was not used"
    for k in r:
        assert _rel(e[k], r[k]) <= 1.5 * _rel(b[k], r[k]), (k, _rel(e[k], r[k]), _rel(b[k], r[k]))


@needs_hopper
@pytest.mark.parametrize(("kind", "L"), [("token", 384), ("token256", 384), ("atom", 128)])
def test_fp32_token_route_is_taken_and_holds_the_yardstick(kind, L):
    from miniworld_engine.kernels.augmented_attention.cuda import fused_pair_tf32

    torch.backends.cuda.matmul.allow_tf32 = True
    mw, pt, ref = _modules(kind)
    single, cond, pair, mask, dy, dt = _case(kind, L, 3, fp32=True)
    taken = []
    orig = fused_pair_tf32.attention_pair_tf32
    fused_pair_tf32.attention_pair_tf32 = lambda *a, **k: (taken.append(1), orig(*a, **k))[1]
    try:
        r = _run(ref, kind, single, cond, pair, mask, dy, torch.float64, False)
        e = _run(mw, kind, single, cond, pair, mask, dy, dt, False)
    finally:
        fused_pair_tf32.attention_pair_tf32 = orig
    b = _run(pt, kind, single, cond, pair, mask, dy, dt, False)
    assert taken
    for k in r:
        assert _rel(e[k], r[k]) <= 1.5 * _rel(b[k], r[k]), (k, _rel(e[k], r[k]), _rel(b[k], r[k]))


@needs_hopper
def test_bf16_training_step_is_bitwise_repeatable():
    mw, _, _ = _modules("token")
    single, cond, pair, mask, dy, dt = _case("token", 384, 6)
    a = _run(mw, "token", single, cond, pair, mask, dy, dt, True)
    b = _run(mw, "token", single, cond, pair, mask, dy, dt, True)
    for k in a:
        assert torch.equal(a[k], b[k]), k


@needs_hopper
@pytest.mark.parametrize("fp32", [False, True])
def test_adaln_rows_match_the_pytorch_leg(fp32):
    from miniworld_engine.kernels.adaln import cuda as adaln_cuda
    from miniworld_engine.modules import AdaptiveLayerNorm
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    torch.backends.cuda.matmul.allow_tf32 = True
    dt = torch.float32 if fp32 else torch.bfloat16
    torch.manual_seed(1)
    m = AdaptiveLayerNorm(768, 384, implementation=IT.MINIWORLD).cuda()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05) if p.dim() > 1 else p.normal_(0, 0.1)
        m.ln_cond.weight.add_(1.0)
    ref = AdaptiveLayerNorm(768, 384, implementation=IT.PYTORCH).cuda().double()
    ref.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    x, c = torch.randn(3, 1, 256, 768, device="cuda").to(dt), torch.randn(3, 1, 256, 384, device="cuda").to(dt)
    dy = torch.randn(3, 1, 256, 768, device="cuda")
    assert adaln_cuda.available(x, c)

    def run(mod, dtype):
        xs, cs = x.to(dtype).clone().requires_grad_(), c.to(dtype).clone().requires_grad_()
        mod.zero_grad()
        o = mod(xs, cs)
        o.backward(dy.to(o.dtype))
        return o.detach().double(), xs.grad.double(), cs.grad.double(), mod.to_scale.weight.grad.double(), mod.ln_cond.weight.grad.double()

    r, e = run(ref, torch.float64), run(m, dt)
    bar = 3e-3 if fp32 else 2e-2
    for name, a, b in zip(("out", "dx", "dcond", "dWs", "dwc"), e, r, strict=True):
        assert _rel(a, b) < bar, (name, _rel(a, b))


@needs_hopper
@pytest.mark.parametrize("fp32", [False, True])
def test_routes_run_under_torch_compile(fp32):
    """The routes are opaque ops: a compiled module still launches the hand-CUDA cores and matches the eager route."""
    from torch.profiler import ProfilerActivity, profile

    torch.backends.cuda.matmul.allow_tf32 = True
    mw, _, _ = _modules("token")
    single, cond, pair, mask, dy, _dt = _case("token", 384, 3, fp32=fp32)
    s, c, p = (t.clone().requires_grad_() for t in (single, cond, pair))

    def run(m):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
            o = m(s, c, p, mask)
        o.backward(dy.to(o.dtype))
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
    assert "attn_tf32_fwd_kernel" in names if fp32 else "attn_fwd_kernel" in names, "the compiled module did not launch the CUDA core"
    # the one AdaLN entry (kernels.adaln.cuda.adaln) picks the row kernels (adaln_fwd_k) or the sm_90a wgmma route (ln_rows_k + adaln_gemm_k)
    # by mode, dtype and shape: either one is "the CUDA adaLN"
    assert any(k in names for k in ("adaln_fwd_k", "adaln_gemm_k", "ln_rows_k")), "the compiled module did not launch the CUDA adaLN"
    assert _rel(out[0], eager[0].double()) < (3e-3 if fp32 else 3e-2)
    assert _rel(out[1], eager[1].double()) < (3e-3 if fp32 else 3e-2)


@needs_hopper
@pytest.mark.parametrize("masked", [False, True])
def test_the_generic_fp32_pair_bias_pass_matches_the_tensor_core_one_at_d_pair_128(masked):
    """``pair_bias_rows32`` (any d_pair; the d_pair 256 route) reproduces ``pair_bias_tok_cuda`` (d_pair 128 only): same key-permuted layout, mask and units."""
    from miniworld_engine.kernels.augmented_attention.cuda.fused_pair import _rows_ext
    from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

    L, dp = 256, 128
    g = torch.Generator(device="cuda").manual_seed(1)
    pair = torch.randn(L * L, dp, device="cuda", generator=g) * 2 + 0.5
    wf = torch.randn(16, dp, device="cuda", generator=g) * 0.1
    mask = (torch.rand(L, device="cuda", generator=g) > 0.2) if masked else None
    a = torch.empty(16, L, L, device="cuda")
    b = torch.empty_like(a)
    rows32().pair_bias_tok_cuda(pair, wf, a, mask, L, 1e-5, True, -1e30)
    _rows_ext().pair_bias_rows32(pair, wf, b, mask, L, 1e-5, -1e30)
    # the tensor-core pass rounds its operands to TF32, this one is fp32 on CUDA cores: the layout agrees exactly, the values to TF32 precision
    assert float((b - a).norm() / a.norm()) < 1e-3, float((b - a).norm() / a.norm())
    assert torch.equal(b[..., :8] <= -1e29, a[..., :8] <= -1e29)

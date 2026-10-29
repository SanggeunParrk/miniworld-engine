"""The fp32 (TF32 tensor core) training kernels of the augmented pair-bias attention core on B200
(kernels/augmented_attention/cuda/sm100: attn_fwd_tf32, attn_dkv_tf32, attn_dqb_tf32) against fp64."""

import math

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
H, DH = 16, 48


@pytest.fixture(autouse=True)
def blackwell():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _inputs(A, L, masked, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(A * L, H * DH, device="cuda", generator=g) for _ in range(3))
    bias = torch.randn(H, L, L, device="cuda", generator=g)
    if masked:
        keep = torch.rand(L, device="cuda", generator=g) > 0.2
        bias[:, :, ~keep] = float("-inf")
    return q, k, v, bias


def _reference(q, k, v, bias, A, L):
    qd, kd, vd = (t.double().view(A, L, H, DH) for t in (q, k, v))
    s = torch.einsum("alhd,ajhd->ahlj", qd, kd) / math.sqrt(DH) + bias.double()[None]
    lse = torch.logsumexp(s, -1)
    o = torch.einsum("ahlj,ajhd->alhd", torch.softmax(s, -1), vd).reshape(A * L, H * DH)
    return o, lse * (1 / math.log(2.0)), s


@pytest.mark.parametrize(("A", "L"), [(2, 128), (4, 256), (2, 384)])
@pytest.mark.parametrize("masked", [False, True])
def test_forward_tf32_matches_fp64(A, L, masked):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, k, v, bias = _inputs(A, L, masked, A * 1000 + L + masked)
    O, LSE = sm100.forward_tf32(q, k, v, bias, A, L)
    torch.cuda.synchronize()
    o_ref, lse_ref, _ = _reference(q, k, v, bias, A, L)
    assert _rel(O, o_ref) < 2e-3, _rel(O, o_ref)
    # TF32 drops the operands' low mantissa bits: the logits (and LSE) shift by ~1e-3; the backward recomputes P from the same
    # operands, so the shift cancels there
    assert float((LSE.double() - lse_ref).abs().max()) < 5e-3


@pytest.mark.parametrize(("A", "L"), [(2, 128), (4, 256), (2, 384)])
@pytest.mark.parametrize("masked", [False, True])
def test_backward_tf32_matches_fp64(A, L, masked):
    """dQ, dK, dV and dbias (summed over the samples) of the fp32 kernels against fp64 autograd, from the kernels' own O / LSE."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, k, v, bias = _inputs(A, L, masked, A * 1000 + L + masked + 7)
    do = torch.randn(A * L, H * DH, device="cuda", generator=torch.Generator(device="cuda").manual_seed(L))
    O, LSE = sm100.forward_tf32(q, k, v, bias, A, L)
    Dd = (do * O).view(A, L, H, DH).sum(-1).permute(0, 2, 1).contiguous()
    DQ, DK, DV, DB = sm100.backward_tf32(q, k, v, do, bias, LSE, Dd, A, L)
    torch.cuda.synchronize()
    leaves = [t.double().requires_grad_() for t in (q, k, v)]
    bd = bias.double().requires_grad_()
    o, _, _ = _reference(leaves[0], leaves[1], leaves[2], bd, A, L)
    o.backward(do.double())
    for name, got, want in (("dq", DQ, leaves[0].grad), ("dk", DK, leaves[1].grad), ("dv", DV, leaves[2].grad), ("dbias", DB, bd.grad)):
        assert _rel(got, want) < 3e-3, f"{name}: {_rel(got, want):.2e}"


@pytest.mark.parametrize(("A", "L"), [(8, 128), (6, 256)])
def test_backward_tf32_into_column_blocks_and_sample_groups(A, L):
    """dQ / dK / dV written into column blocks of a wider buffer (the fused training block's q|k|v|g gradient: rows 4 W floats
    apart) match the fresh-buffer call to fp32 reassociation (not bit for bit: attn_dkv_tf32's two MMA warps accumulate their
    alternate query blocks onto one dK / dV accumulator in the order they get there, so two identical calls differ in the last
    bits), and fp64 everywhere. A = 8, L = 128 takes attn_dqb_tf32's sample groups (32 items x NS = 4 on the B200's SMs: dbias as
    reductions into a zeroed buffer)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    q, k, v, bias = _inputs(A, L, True, A * 100 + L)
    do = torch.randn(A * L, H * DH, device="cuda", generator=torch.Generator(device="cuda").manual_seed(L + 1))
    O, LSE = sm100.forward_tf32(q, k, v, bias, A, L)
    Dd = (do * O).view(A, L, H, DH).sum(-1).permute(0, 2, 1).contiguous()
    W = H * DH
    buf = torch.full((A * L, 4 * W), float("nan"), device="cuda")
    DQ, DK, DV, DB = sm100.backward_tf32(q, k, v, do, bias, LSE, Dd, A, L, dq=buf[:, :W], dk=buf[:, W:2 * W], dv=buf[:, 2 * W:3 * W])
    DQ0, DK0, DV0, DB0 = sm100.backward_tf32(q, k, v, do, bias, LSE, Dd, A, L)
    torch.cuda.synchronize()
    assert DQ.data_ptr() == buf.data_ptr(), "outputs left their column blocks"
    assert torch.isnan(buf[:, 3 * W:]).all(), "outputs left their column blocks"
    assert _rel(DK, DK0) < 1e-5, (_rel(DK, DK0), _rel(DV, DV0))
    assert _rel(DV, DV0) < 1e-5, (_rel(DK, DK0), _rel(DV, DV0))
    leaves = [t.double().requires_grad_() for t in (q, k, v)]
    bd = bias.double().requires_grad_()
    o, _, _ = _reference(leaves[0], leaves[1], leaves[2], bd, A, L)
    o.backward(do.double())
    for name, got, want in (("dq", DQ, leaves[0].grad), ("dk", DK, leaves[1].grad), ("dv", DV, leaves[2].grad), ("dbias", DB, bd.grad),
                            ("dq fresh", DQ0, leaves[0].grad), ("dbias fresh", DB0, bd.grad)):
        assert _rel(got, want) < 3e-3, f"{name}: {_rel(got, want):.2e}"


# ------------------------------------------------------------------------------------------------ the pair-fused core (fused_pair.py), fp32
def _pair_case(A, L, H, DH, masked, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v, do = (torch.randn(A, 1, L, H, DH, device="cuda", generator=g) for _ in range(4))
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g) * 1.5 + 0.3
    lnw = 1.0 + 0.3 * torch.randn(128, device="cuda", generator=g)
    wb = torch.randn(H, 128, device="cuda", generator=g) * 128 ** -0.5 * 2.0
    mask = (torch.rand(L, device="cuda", generator=g) > 0.2) if masked else None
    return q, k, v, pair, lnw, wb, mask, do


def _pair_truth(q, k, v, pair, lnw, wb, mask, do, eps=1e-5):
    """fp64 autograd of softmax(q k^T / sqrt d + LN(pair) (wb lnw)^T) v: (O, dq, dk, dv, dpair, dlnw, dwb)."""
    leaves = [t.detach().double().requires_grad_() for t in (q, k, v, pair, lnw, wb)]
    qd, kd, vd, pd, lw, wbd = leaves
    DHd = q.shape[-1]
    mu = pd.mean(-1, keepdim=True)
    xh = (pd - mu) / torch.sqrt(((pd - mu) ** 2).mean(-1, keepdim=True) + eps)
    bias = torch.einsum("bijc,hc->bhij", xh, wbd * lw[None])[0]
    s = torch.einsum("aihd,ajhd->ahij", qd[:, 0], kd[:, 0]) / DHd ** 0.5 + bias[None]
    if mask is not None:
        s = s.masked_fill(~mask[None, None, None, :], torch.finfo(torch.float64).min)
    o = torch.einsum("ahij,ajhd->aihd", torch.softmax(s, -1), vd[:, 0])[:, None]
    o.backward(do.double())
    return (o.detach(), *(t.grad for t in leaves))


@pytest.mark.parametrize(("H", "DH"), [(16, 48), (24, 32), (12, 64), (16, 64)])
@pytest.mark.parametrize(("A", "L", "masked"), [(3, 256, True), (2, 128, False)])
def test_pair_fused_tf32_matches_fp64(H, DH, A, L, masked):
    """The fp32 pair-fused core (TF32 pair pass + the deterministic TF32 core; odd A runs a zero pad sample) against fp64: every
    output and gradient within 3e-3 (the bar of the core tests above)."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import fused_pair

    torch.backends.cuda.matmul.allow_tf32 = True
    q, k, v, pair, lnw, wb, mask, do = _pair_case(A, L, H, DH, masked, H * 100 + DH + L + A)
    truth = _pair_truth(q, k, v, pair, lnw, wb, mask, do)
    leaves = [t.clone().requires_grad_() for t in (q, k, v, pair, lnw, wb)]
    assert fused_pair.available(leaves[0], leaves[3], leaves[4], leaves[5], mask)
    o = fused_pair.attention_pair_sm100(*leaves, mask)
    o.backward(do)
    torch.cuda.synchronize()
    got = (o.detach(), *(t.grad for t in leaves))
    for name, x, t in zip(("O", "dq", "dk", "dv", "dpair", "dlnw", "dwb"), got, truth, strict=True):
        assert _rel(x, t) < 3e-3, f"{name}: {_rel(x, t):.2e}"
    with torch.no_grad():   # inference takes the other pair pass (pair_bias_tok: mask folded in, no statistics)
        oi = fused_pair.attention_pair_sm100(q, k, v, pair, lnw, wb, mask)
    assert _rel(oi, truth[0]) < 3e-3, _rel(oi, truth[0])


@pytest.mark.parametrize(("H", "DH"), [(16, 48), (16, 64)])
def test_pair_fused_tf32_backward_is_bitwise_repeatable(H, DH):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import fused_pair

    torch.backends.cuda.matmul.allow_tf32 = True
    q, k, v, pair, lnw, wb, mask, do = _pair_case(48, 384, H, DH, True, 13)
    runs = []
    for _ in range(3):
        leaves = [t.clone().requires_grad_() for t in (q, k, v, pair, lnw, wb)]
        fused_pair.attention_pair_sm100(*leaves, mask).backward(do)
        runs.append([t.grad for t in leaves])
    for a, b in zip(runs, runs[1:]):
        for name, x, y in zip(("dq", "dk", "dv", "dpair", "dlnw", "dwb"), a, b, strict=True):
            assert torch.equal(x, y), f"{name} differs between runs"


def _aa_modules(d, dc, dp, H):
    from miniworld_engine.modules import AugmentedAttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    torch.manual_seed(3)
    mw = AugmentedAttentionPairBias(d, dc, dp, H, implementation=IT.MINIWORLD).cuda()
    with torch.no_grad():
        for p in mw.parameters():
            p.normal_(0, p.shape[-1] ** -0.5) if p.dim() > 1 else p.normal_(0, 0.1)
        mw.ln_pair.weight.add_(1.0)
    pt = AugmentedAttentionPairBias(d, dc, dp, H, implementation=IT.PYTORCH).cuda()
    pt.load_state_dict(mw.state_dict())
    ref = AugmentedAttentionPairBias(d, dc, dp, H, implementation=IT.PYTORCH).cuda().double()
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


@pytest.mark.parametrize(("d", "H"), [(768, 16), (768, 24), (768, 12), (1024, 16)])
def test_module_fp32_takes_the_pair_fused_core_and_holds_the_yardstick(d, H):
    """AugmentedAttentionPairBias (MINIWORLD) in fp32 on B200 (TF32 matmuls allowed) runs the TF32 pair-fused core, within 1.5x the
    PyTorch module's error (fp32, TF32 matmuls) against an fp64 run, for the output and every gradient checked."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import fused_pair

    torch.backends.cuda.matmul.allow_tf32 = True
    mw, pt, ref = _aa_modules(d, 384, 128, H)
    single, cond, pair, mask, dy = _aa_case(3, 256, d, 384, 128, torch.float32)
    taken = []
    orig = fused_pair.attention_pair_sm100
    fused_pair.attention_pair_sm100 = lambda *a, **k: (taken.append(1), orig(*a, **k))[1]  # ty: ignore[invalid-assignment] -- spy
    try:
        e = _aa_run(mw, single, cond, pair, mask, dy, torch.float32, False)
    finally:
        fused_pair.attention_pair_sm100 = orig
    r = _aa_run(ref, single, cond, pair, mask, dy, torch.float64, False)
    b = _aa_run(pt, single, cond, pair, mask, dy, torch.float32, False)
    assert taken, "the TF32 pair-fused sm100 core was not used"
    for k in r:
        assert _rel(e[k], r[k]) <= 1.5 * _rel(b[k], r[k]) + 1e-6, (k, _rel(e[k], r[k]), _rel(b[k], r[k]))

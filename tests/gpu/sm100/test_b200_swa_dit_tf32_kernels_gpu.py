"""The hand-written TF32 kernels of the fp32 SWA atom DiT training step on B200 (``kernels/swa_dit/cuda/sm100``: ``tf32_fwd`` / ``tf32_bwd``), each
chain against the fp64 meaning of its outputs on random operands: relative error TF32-level (< 1e-3), every output finite with a NaN-poisoned
buffer (the kernel wrote all of it), a repeated launch bit-identical (no atomics, fixed summation orders), no register spills.
The whole block against fp64 and against the Triton fp32 path is ``test_b200_swa_dit_tf32_gpu.py``."""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_bwd as TB
from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_fwd as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100)"),
]
C, H, D, NH, HW, EPS = 128, 4, 32, 256, 64, 1e-5
TOL = 1e-3
SHAPES = [(3, 2, 256), (5, 1, 384), (48, 1, 512)]                         # (A, B, S)


def _rel(x, e):
    return float((x.double() - e.double()).norm() / e.double().norm().clamp_min(1e-30))


def _gen(seed):
    g = torch.Generator().manual_seed(seed)
    return lambda *shape, sc=1.0: (torch.randn(*shape, generator=g) * sc).cuda()


def _kernels():
    return TB.kernels_bwd(torch.cuda.current_device())


def _row_maps(m, a, b, s):
    """(b index, atom index) of every row of an [A B S, ...] row-major tensor."""
    idx = torch.arange(m, device="cuda")
    n = idx // s
    return n % b, idx % s


def _round(t):
    return F._round_tf32(t.contiguous())


# ----------------------------------------------------------------------------------------------------------------- spills
def test_no_register_spills():
    k = _kernels()
    objs = {"ffn_bwd_gate": k.ffn_gate.k, "adaln_bwd": k.adaln.k, "oproj_bwd_fused": k.oproj_fused.k, "attn_dq": k.attn.kq,
            "attn_dkv": k.attn.kk, "mod_silu": k.mod.ks, "mod_dsilu": k.mod.kd, "dw": k.dw.kd, "dw_reduce": k.dw.kr}
    for name, kern in objs.items():
        assert kern.lmem == 0, f"{name} spills {kern.lmem} B of local memory per thread at {kern.regs} registers"
    for name, (regs, lmem) in F.kernels_tf32(torch.cuda.current_device()).stats().items():
        assert lmem == 0, f"fwd {name} spills {lmem} B at {regs} registers"


# ---------------------------------------------------------------------------------------------------------- weight forms
def test_backward_weight_forms_are_the_rounded_transposes():
    r = _gen(11)
    wd, wo = r(C, NH, sc=0.1), r(C, C, sc=0.1)
    wdt, wot = TB._bwd_forms(wd, wo)
    assert torch.equal(wdt, _round(wd.t())) and torch.equal(wot, _round(wo.t()))


# ---------------------------------------------------------------------------------------------------------------- FFN backward
@pytest.mark.parametrize(("a", "b", "s"), SHAPES)
def test_ffn_backward_chain(a, b, s):
    """gate kernel -> dy = [dA dB] Wu (cuBLAS TF32) -> adaLN backward: dffn, h, dA | dB, dq1 and the FFN's modulation gradient."""
    r = _gen(1)
    m = a * b * s
    y, q1, dq2, ffn, att = (r(m, C) for _ in range(5))
    mod, wu, wd = r(b * s, 6 * C, sc=0.5), r(2 * NH, C, sc=C ** -0.5), r(C, NH, sc=NH ** -0.5)
    wab = _round(torch.cat([torch.cat([wu[32 * j:32 * j + 32], wu[NH + 32 * j:NH + 32 * j + 32]]) for j in range(8)]))
    wdt = _round(wd.t())
    k = _kernels()
    dmod = torch.full((b * s, 6 * C), float("nan"), device="cuda")
    rg, (dffn, hh, dab) = k.ffn_gate.bind(y, dq2, mod, wab, wdt, a, b)
    dy = torch.empty(m, C, device="cuda")
    ra, dq1 = k.adaln.bind(q1, dy, dq2, ffn, mod, dmod, a, b, EPS, 3 * C, att, 2 * C)

    def go():
        rg()
        dy.copy_(TB.mm_tf32(dab, wu))
        ra()

    for t in (dffn, hh, dab):
        t.fill_(float("nan"))
    go()
    first = [t.clone() for t in (dffn, hh, dab, dq1, dmod[:, 2 * C:])]
    dmod.fill_(float("nan"))
    go()
    assert all(torch.equal(x, t) for x, t in zip(first, (dffn, hh, dab, dq1, dmod[:, 2 * C:])))
    bi, si = _row_maps(m, a, b, s)
    md = mod[bi * s + si].double()
    Y, Q1, DQ2, FF, AT, WU, WD = (t.double() for t in (y, q1, dq2, ffn, att, wu, wd))
    dffn_r = DQ2 * md[:, 5 * C:6 * C]
    ab = Y @ WU.t()
    av, bv = ab[:, :NH], ab[:, NH:]
    dh = dffn_r @ WD
    sa = torch.sigmoid(av)
    da, db = dh * bv * sa * (1 + av * (1 - sa)), dh * av * sa
    dy_r = torch.cat([da, db], 1) @ WU
    rstd = 1 / torch.sqrt((Q1 ** 2).mean(1, keepdim=True) + EPS)
    xh = Q1 * rstd
    dxh = dy_r * (1 + md[:, 4 * C:5 * C])
    dq1_r = DQ2 + rstd * (dxh - xh * (dxh * xh).mean(1, keepdim=True))

    def aug(x):
        return x.view(a, b, s, C).sum(0).reshape(b * s, C)

    assert _rel(dffn, dffn_r) < TOL and _rel(hh, av * sa * bv) < TOL and _rel(dab, torch.cat([da, db], 1)) < TOL
    assert _rel(dq1, dq1_r) < TOL
    assert _rel(dmod[:, 3 * C:4 * C], aug(dy_r)) < TOL and _rel(dmod[:, 4 * C:5 * C], aug(dy_r * xh)) < TOL
    assert _rel(dmod[:, 5 * C:], aug(DQ2 * FF)) < TOL and _rel(dmod[:, 2 * C:3 * C], aug(dq1_r * AT)) < TOL   # gate_f, gate_a
    assert bool(torch.isnan(dmod[:, :2 * C]).all())                        # the kernels write nothing outside their columns
    for t in (dffn, hh, dab, dq1):
        assert bool(torch.isfinite(t).all())


# ------------------------------------------------------------------------------------------------- out-projection backward
@pytest.mark.parametrize(("a", "b", "s"), SHAPES)
def test_oproj_backward_fused(a, b, s):
    r = _gen(2)
    m = a * b * s
    dq1, o, gt = r(m, C), r(m, C), r(m, C)
    mod, wo = r(b * s, 6 * C, sc=0.5), r(C, C, sc=C ** -0.5)
    _, wot = TB._bwd_forms(r(C, NH), wo)
    dp = torch.full((m, 4 * C), float("nan"), device="cuda")
    run, (datt, gated, do, dv) = _kernels().oproj_fused.bind(dq1, o, gt, mod, wot, dp, a, b)
    run()
    first = [t.clone() for t in (datt, gated, do, dv, dp[:, 3 * C:])]
    for t in (datt, gated, do, dv, dp):
        t.fill_(float("nan"))
    run()
    assert all(torch.equal(x, t) for x, t in zip(first, (datt, gated, do, dv, dp[:, 3 * C:])))
    bi, si = _row_maps(m, a, b, s)
    ga = mod[bi * s + si, 2 * C:3 * C].double()
    Dq, O, G = dq1.double(), o.double(), gt.double()
    wor = _round(wo.t()).t().double()
    datt_r = Dq * ga
    so = torch.sigmoid(G)
    dgated = datt_r @ wor
    do_r = dgated * so
    dv_r = (do_r * O).view(m // s, s, H, D).sum(-1).permute(0, 2, 1)
    assert _rel(datt, datt_r) < TOL and _rel(gated, so * O) < TOL and _rel(do, do_r) < TOL
    assert _rel(dp[:, 3 * C:], dgated * O * so * (1 - so)) < TOL and _rel(dv, dv_r) < TOL
    assert bool(torch.isnan(dp[:, :3 * C]).all())                           # dP[:, :3C] is the attention kernels'


# --------------------------------------------------------------------------------------------------------- attention backward
@pytest.mark.parametrize(("n", "bn", "s", "lens"), [(3, 1, 256, [256, 239, 5]), (3, 3, 384, [384, 1, 129]), (4, 2, 1024, [1024, 700, 65, 64])])
def test_attention_backward_with_qkvg_tail(n, bn, s, lens):
    """dQ / dK / dV with the head-wise qkvg tail (unrope + per-head RMS backward) in the epilogues: dP[:, :3C] = [dpq | dpk | dpv]."""
    r = _gen(n * 1000 + s)
    q, k_, v = (_round(r(n, H, s, D)) for _ in range(3))
    do = _round(r(n * s, H * D))
    sq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    pq, pk = r(n * s, C), r(n * s, C)
    ang = r(bn, s, D // 2, sc=3.0)
    cos, sin = ang.cos().reshape(bn * s, -1).contiguous(), ang.sin().reshape(bn * s, -1).contiguous()
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k_, v))
    idx = torch.arange(s, device="cuda")
    band = (idx[:, None] - idx[None, :]).abs() <= HW
    ki = idx[None, :] < sq.view(n, 1)
    mask = band[None, None] & ki[:, None, None, :] & ki[:, None, :, None]
    sc = ((qd @ kd.transpose(-1, -2)) * D ** -0.5).masked_fill(~mask, -1e30)
    lse = torch.logsumexp(sc, -1)
    o = torch.where(mask, torch.exp(sc - lse[..., None]), torch.zeros_like(sc)) @ vd
    do4 = do.double().view(n, s, H, D).permute(0, 2, 1, 3)
    (o * do4).sum().backward()
    dvv = (do4 * o).sum(-1).detach().float().contiguous()
    lse_f = torch.where(lse < -1e29, torch.zeros_like(lse), lse).detach().float().contiguous()
    nn_, ss = torch.arange(n * s, device="cuda") // s, torch.arange(n * s, device="cuda") % s
    cs, sn = cos[(nn_ % bn) * s + ss].double()[:, None, :], sin[(nn_ % bn) * s + ss].double()[:, None, :]

    def heads(t):
        return t.permute(0, 2, 1, 3).reshape(n * s, H, D)

    def unrope(t):
        t1, t2 = t[..., :D // 2], t[..., D // 2:]
        return torch.cat([t1 * cs + t2 * sn, t2 * cs - t1 * sn], -1)

    def hrb(p, du):
        p = p.double().view(n * s, H, D)
        rr = 1 / torch.sqrt((p ** 2).mean(-1, keepdim=True) + EPS)
        yh = p * rr
        return (rr * (du - yh * (du * yh).mean(-1, keepdim=True))).reshape(n * s, C)

    dp = torch.full((n * s, 4 * C), float("nan"), device="cuda")
    run = _kernels().attn.bind(q, k_, v, do, lse_f, dvv, sq, pq, pk, cos, sin, dp, bn)
    run()
    first = dp[:, :3 * C].clone()
    dp.fill_(float("nan"))
    run()
    assert torch.equal(first, dp[:, :3 * C]) and bool(torch.isfinite(first).all())
    assert _rel(dp[:, :C], hrb(pq, unrope(heads(qd.grad)))) < TOL
    assert _rel(dp[:, C:2 * C], hrb(pk, unrope(heads(kd.grad)))) < TOL
    assert _rel(dp[:, 2 * C:3 * C], heads(vd.grad).reshape(n * s, C)) < TOL
    assert bool(torch.isnan(dp[:, 3 * C:]).all())


# ------------------------------------------------------------------------------------------------------------------- forward
@pytest.mark.parametrize(("a", "b", "s"), SHAPES)
@pytest.mark.parametrize("save", [True, False])
def test_qkvg_forward(a, b, s, save):
    from test_b200_swa_dit_tf32_gpu import _case, _mod32, _ref64

    leaves, cos, sin, seqused = _case(a, b, s, seed=5)
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    mod = _mod32(c_base, wmod)
    w, _, _, _ = F._tf32_block_forms(wqkv, wg, wo, wu, wd)
    run, (qh, kh, vh, g, x, pq, pk) = _kernels_fwd().qkvg.bind(q, mod, cos.reshape(-1, cos.shape[-1]).contiguous(),
                                                                 sin.reshape(-1, sin.shape[-1]).contiguous(), w, a, b, save=save)
    outs = [qh, kh, vh, g] + ([x, pq, pk] if save else [])
    for t in outs:
        t.fill_(float("nan"))
    run()
    first = [t.clone() for t in outs]
    for t in outs:
        t.fill_(float("nan"))
    run()
    assert all(torch.equal(u, t) for u, t in zip(first, outs)) and all(bool(torch.isfinite(t).all()) for t in outs)
    ref = _ref64(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, b)
    for name, got in zip(("Qh", "Kh", "Vh", "G"), (qh, kh, vh, g)):
        assert _rel(got, ref[name]) < TOL, name
    if save:
        for name, got in zip(("X", "PQ", "PK"), (x, pq, pk)):
            assert _rel(got, ref[name]) < TOL, name


def _kernels_fwd():
    return F.kernels_tf32(torch.cuda.current_device())


@pytest.mark.parametrize(("n", "s", "lens"), [(3, 256, [256, 239, 5]), (3, 384, [384, 1, 129]), (4, 1024, [1024, 700, 65, 64])])
def test_attention_forward(n, s, lens):
    r = _gen(n * 7 + s)
    q, k_, v = (_round(r(n, H, s, D)) for _ in range(3))
    sq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    run, o, lse = _kernels_fwd().attn.bind(q, k_, v, sq)
    o.fill_(float("nan"))
    lse.fill_(float("nan"))
    run()
    a_, b_ = o.clone(), lse.clone()
    o.fill_(float("nan"))
    lse.fill_(float("nan"))
    run()
    assert torch.equal(a_, o) and torch.equal(b_, lse) and bool(torch.isfinite(o).all())
    qd, kd, vd = q.double(), k_.double(), v.double()
    idx = torch.arange(s, device="cuda")
    band = (idx[:, None] - idx[None, :]).abs() <= HW
    ki = idx[None, :] < sq.view(n, 1)
    mask = band[None, None] & ki[:, None, None, :] & ki[:, None, :, None]
    sc = ((qd @ kd.transpose(-1, -2)) * D ** -0.5).masked_fill(~mask, float("-inf"))
    mx = sc.amax(-1, keepdim=True)
    has = torch.isfinite(mx)
    p = torch.where(has, torch.exp(sc - torch.where(has, mx, torch.zeros_like(mx))), torch.zeros_like(sc))
    ll = p.sum(-1, keepdim=True)
    o_ref = torch.where(ll > 0, (p / ll.clamp_min(1e-300)) @ vd, torch.zeros_like(qd)).permute(0, 2, 1, 3).reshape(n * s, H * D)
    lse_ref = torch.where(ll[..., 0] > 0, torch.where(has, mx, torch.zeros_like(mx))[..., 0] + torch.log(ll[..., 0].clamp_min(1e-300)), torch.zeros_like(ll[..., 0]))
    valid = (idx[None, :] < sq.view(n, 1))[:, None, :].expand(n, H, s)       # padding query rows carry a convention value, not checked
    assert _rel(o, o_ref) < TOL and _rel(lse[valid], lse_ref[valid]) < 1e-4


# ----------------------------------------------------------------------------------------------------------- modulation backward
def test_modulation_backward():
    r = _gen(9)
    c, wmod, g = r(256, C), r(6 * C, C, sc=0.1), r(256, 6 * C)
    dc, dw = TB.mod_bwd_tf32(g, c, wmod)
    c64 = c.double().requires_grad_()
    w64 = wmod.double().requires_grad_()
    (torch.nn.functional.silu(c64) @ w64.t() * g.double()).sum().backward()
    assert _rel(dc, c64.grad) < TOL and _rel(dw, w64.grad) < TOL


# ------------------------------------------------------------------------------------------------------------ weight gradients
@pytest.mark.parametrize("a,b,s", SHAPES)
def test_weight_gradients(a, b, s):
    """dw_tf32: the four split-K weight-gradient problems (dWqkvg = dP^T x, dWo, dWu, dWd) against fp64, NaN-poisoned outputs, repeat identical."""
    r = _gen(31 + a)
    m = a * b * s
    dp, dab, hh = (_round(r(m, w)) for w in (4 * C, 2 * NH, NH))
    x, datt, gated, y, dffn = (_round(r(m, C)) for _ in range(5))
    run, outs = _kernels().dw.bind(dp, x, datt, gated, dab, y, dffn, hh)
    refs = [p.double().t() @ q.double() for p, q in ((dp, x), (datt, gated), (dab, y), (dffn, hh))]
    for o in outs:
        o.fill_(float("nan"))
    run()
    first = [o.clone() for o in outs]
    for o in outs:
        o.fill_(float("nan"))
    run()
    for o, f, ref in zip(outs, first, refs):
        assert o.shape == ref.shape and torch.isfinite(o).all()
        assert _rel(o, ref) < 1e-4
        assert torch.equal(o, f), "the fixed-order slice reduction is bit-reproducible"

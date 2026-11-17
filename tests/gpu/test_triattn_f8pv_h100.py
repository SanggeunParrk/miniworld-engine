"""The opt-in FP8-PV TriangleAttention core (``h100_sm90.FP8_PV``, ``triattn_h16_sm90.cu`` built with ``-DTA_F8PV=1``).

- the default route is untouched: ``core_h16()`` is the loaded fp16 core while the switch is off;
- the FP8 core against an fp32 attention of the same fp16 operands (P and V on e4m3, per-(batch, head) V scale, the row sums from the
  quantized P): relative error inside the e4m3 band (a 3-bit mantissa: a few percent, against 0.2 % of the fp16 core), head widths 32 and 16,
  with a padded key mask, 4 and 8 heads;
- ``vt_quant`` (the V^T pass) equals the torch reference except at exact rounding ties (one e4m3 step).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


pytestmark.append(pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)"))


def _reference(q, k, v, sbias, scale):
    out = torch.empty_like(q, dtype=torch.float32)
    for n0 in range(0, q.shape[1], 32):
        lg = torch.einsum("bnjhd,bnkhd->bnhjk", q[:, n0 : n0 + 32].float(), k[:, n0 : n0 + 32].float()) + sbias.float()[:, None]
        out[:, n0 : n0 + 32] = torch.einsum("bnhjk,bnkhd->bnjhd", torch.softmax(scale * lg, -1), v[:, n0 : n0 + 32].float())
    return out


def test_default_route_is_the_fp16_core():
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    assert hs.FP8_PV is False
    assert hs.core_h16() is hs._core_h16()


@pytest.mark.parametrize(("heads", "dh", "length"), [(4, 32, 256), (8, 32, 512), (4, 16, 256), (12, 32, 256)])
def test_f8pv_core_error_band(monkeypatch, heads, dh, length):
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    torch.manual_seed(length + heads)
    q, k, v = (torch.randn(1, length, length, heads, dh, device="cuda").mul(s).half() for s in (1.5, 1.5, 3.0))
    sbias = torch.randn(1, heads, length, length, device="cuda").half()
    keep = torch.ones(1, length, dtype=torch.bool, device="cuda")
    keep[:, length - 40 :] = False
    sbias = sbias.masked_fill(~keep[:, None, None, :], float("-inf")).contiguous()
    scale = dh**-0.5
    monkeypatch.setattr(hs, "FP8_PV", True)
    got = hs.core_h16().triattn_fwd_h16(q, k, v, sbias, scale, False, key_mask=keep)[0]
    monkeypatch.setattr(hs, "FP8_PV", False)
    fp16 = hs.core_h16().triattn_fwd_h16(q, k, v, sbias, scale, False, key_mask=keep)[0]
    ref = _reference(q, k, v, sbias, scale)
    err8 = ((got.float() - ref).norm() / ref.norm()).item()
    err16 = ((fp16.float() - ref).norm() / ref.norm()).item()
    assert err16 < 0.005
    assert err8 < 0.05, err8
    assert err8 > 3 * err16  # (the variant really runs e4m3)


def test_vt_quant_matches_the_torch_reference():
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    for dh, nh in ((32, 4), (16, 4), (32, 12), (32, 16)):
        torch.manual_seed(dh + nh)
        v = torch.randn(2, 40, 256, nh, dh, device="cuda").mul(3).half()
        a, sa = hs.quantize_v(v)
        b, sb = hs.quantize_v_torch(v)
        ne = a.view(torch.uint8) != b.view(torch.uint8)  # (only exact rounding ties differ: the pass and torch break them differently)
        assert ne.float().mean().item() < 1e-3
        x, y = a.float()[ne], b.float()[ne]
        assert ((x - y).abs() <= 0.13 * torch.maximum(x.abs(), y.abs())).all()
        assert torch.equal(sa, sb)


# ---------------------------------------------------------------------------------------------- hi / lo V planes (FP8_PV_HL) and the fused front


@pytest.mark.parametrize(("heads", "dh", "length"), [(4, 32, 256), (8, 32, 512), (4, 16, 256), (12, 32, 256)])
def test_f8pv_hl_core_error_band(monkeypatch, heads, dh, length):
    """V as two e4m3 planes (hi, residual x16): the V rounding error drops, what is left is the e4m3 probabilities."""
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    torch.manual_seed(length + heads)
    q, k, v = (torch.randn(1, length, length, heads, dh, device="cuda").mul(s).half() for s in (1.5, 1.5, 3.0))
    sbias = torch.randn(1, heads, length, length, device="cuda").half()
    keep = torch.ones(1, length, dtype=torch.bool, device="cuda")
    keep[:, length - 40 :] = False
    sbias = sbias.masked_fill(~keep[:, None, None, :], float("-inf")).contiguous()
    scale = dh**-0.5
    ref = _reference(q, k, v, sbias, scale)
    errs = {}
    for name, hl in (("hi", False), ("hl", True)):
        monkeypatch.setattr(hs, "FP8_PV", True)
        monkeypatch.setattr(hs, "FP8_PV_HL", hl)
        got = hs.core_h16().triattn_fwd_h16(q, k, v, sbias, scale, False, key_mask=keep)[0]
        errs[name] = ((got.float() - ref).norm() / ref.norm()).item()
    assert errs["hl"] < 0.035, errs  # (the e4m3 probabilities are what is left: 2-3 % on this noisy synthetic attention)
    assert errs["hl"] < 0.8 * errs["hi"], errs


def test_vt_quant_hl_matches_the_torch_reference(monkeypatch):
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    monkeypatch.setattr(hs, "FP8_PV_HL", True)
    for dh, nh in ((32, 4), (16, 4), (32, 12), (32, 16)):
        torch.manual_seed(dh + nh)
        v = torch.randn(2, 40, 256, nh, dh, device="cuda").mul(3).half()
        a, sa = hs.quantize_v(v)
        b, sb = hs.quantize_v_torch(v)
        assert a.shape == b.shape == (2, 40, nh, 2, dh, 256)
        assert torch.equal(sa, sb)
        da = a.float()[:, :, :, 0] + a.float()[:, :, :, 1] / 16
        db = b.float()[:, :, :, 0] + b.float()[:, :, :, 1] / 16
        assert ((da - db).abs() > 1e-2 * da.abs().clamp_min(1.0)).float().mean().item() < 1e-3  # (rounding ties only)


@pytest.mark.parametrize("planes", [1, 2])
def test_fused_front_writes_the_v_planes(planes):
    """tri_front_vt: q / k / bias bit-identical to tri_front, V^T planes = the static-scale e4m3 hi (/ lo) of the projection."""
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    c, nh, d, length, batch = 128, 4, 32, 256, 2
    torch.manual_seed(planes)
    w = lambda *s: torch.randn(*s, device="cuda") * 0.08
    w4, wb16 = hs.pack_front(w(c, c), w(c, c), w(c, c), w(c, c), w(nh, c))
    lnw, lnb = torch.ones(c, device="cuda"), torch.zeros(c, device="cuda")
    x = torch.randn(batch * length * length, c, device="cuda").bfloat16()
    mask = torch.ones(batch, length, dtype=torch.bool, device="cuda")
    mod = hs._mod()
    q, k, v, _, sb = mod.tri_front(x, lnw, lnb, 1e-5, w4, wb16, mask, batch, length, d**-0.5, False, True, False)
    vs = hs.vscale_from_amax(v.view(-1, nh, d).abs().amax(dim=(0, 2)))
    ovf = torch.zeros(1, dtype=torch.int32, device="cuda")
    q2, k2, _, _, sb2, vt = mod.tri_front_vt(
        x, lnw, lnb, 1e-5, w4, wb16, mask, batch, length, d**-0.5, False, (1.0 / vs).contiguous(), ovf, planes
    )
    assert torch.equal(q, q2) and torch.equal(k, k2) and torch.equal(sb, sb2)
    assert vt.shape == (batch, length, nh, planes, d, length) and int(ovf.item()) == 0
    vtf = vt.float().reshape(batch, length, nh, planes, d, length // 32, 32)
    deq = vtf[:, :, :, 0] + (vtf[:, :, :, 1] / 16 if planes == 2 else 0)
    deq = deq * vs[None, None, :, None, None, None]
    out = torch.empty_like(deq)
    out[..., hs.vt_perm(torch.device("cuda"))] = deq
    ref = v.view(batch, length, length, nh, d).float().permute(0, 1, 3, 4, 2).reshape(batch, length, nh, d, length // 32, 32)
    err = ((out - ref).norm() / ref.norm()).item()
    assert err < (2e-3 if planes == 2 else 0.04), err
    ovf.zero_()  # a scale far too small: values saturate and the flag is set
    mod.tri_front_vt(x, lnw, lnb, 1e-5, w4, wb16, mask, batch, length, d**-0.5, False, (1.0 / (vs / 64)).contiguous(), ovf, planes)
    assert int(ovf.item()) == 1


def test_whitelist_gates_the_route():
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    try:
        assert hs.enable_fp8_pv_for("boltz2") is False  # 9c90 bistable: not whitelisted
        assert (hs.FP8_PV, hs.FP8_PV_FUSED) == (False, False)
        assert hs.enable_fp8_pv_for("protenix1") is True
        assert hs.FP8_PV and hs.FP8_PV_FUSED and hs.FP8_VSCALE_AUTO
    finally:
        hs.disable_fp8_pv()
    assert hs.core_h16() is hs._core_h16()


@pytest.mark.parametrize(("heads", "dh"), [(4, 32), (12, 32), (4, 16)])
def test_vt_quant_static_scale_and_overflow_flag(heads, dh):
    """The single-pass vt_quant with a static per-head scale equals the dynamic pass at the same scale (rounding ties aside), writes the scale it was given,
    and sets the flag when a value saturates."""
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    torch.manual_seed(heads + dh)
    v = torch.randn(2, 24, 256, heads, dh, device="cuda").mul(3).half()
    v[1] = v[0]  # (the same amax in both batch elements: the static scale equals the dynamic one)
    _dyn, vs = hs.quantize_v(v)
    amax = vs * 448.0  # [B, H]: amax per (batch, head)
    one = amax.amax(dim=0)  # one scale for both batch elements: the larger
    vinv = (448.0 / one).contiguous()
    ovf = torch.zeros(1, dtype=torch.int32, device="cuda")
    got, gs = hs._core_f8pv().vt_quant(v, 0.0, vinv, ovf)
    # (the largest value maps to exactly 448 +-ulp: whether it flags is a rounding coin toss, so the flag is checked at 2 % headroom below)
    assert torch.allclose(gs, (one / 448.0).expand_as(gs), rtol=1e-6)
    ref, _ = hs._core_f8pv().vt_quant(
        v, 0.0
    )  # per-(batch, head) amax (dynamic): equal to the static one for the batch element with the larger amax
    b = 0
    ne = got[b].view(torch.uint8) != ref[b].view(torch.uint8)
    assert ne.float().mean().item() < 1e-3
    ovf.zero_()
    hs._core_f8pv().vt_quant(v, 0.0, (vinv * 0.98).contiguous(), ovf)
    assert int(ovf.item()) == 0
    hs._core_f8pv().vt_quant(v, 0.0, (vinv * 64).contiguous(), ovf)
    assert int(ovf.item()) == 1


def _module_args(dp, heads, dh, length, wv_scale=1.0, seed=0):
    torch.manual_seed(seed)
    dhid = heads * dh
    w = lambda *s: torch.randn(*s, device="cuda") * 0.08
    wq, wk, wv, wg = w(dhid, dp), w(dhid, dp), w(dhid, dp) * wv_scale, w(dhid, dp)
    empty = torch.empty(0, device="cuda")
    pair = torch.randn(1, length, length, dp, device="cuda").bfloat16()
    mask = torch.ones(1, length, dtype=torch.bool, device="cuda")
    return [
        pair,
        torch.ones(dp, device="cuda"),
        torch.zeros(dp, device="cuda"),
        wq,
        wk,
        wv,
        wg,
        w(heads, dp),
        w(dp, dhid),
        mask,
        empty.bfloat16(),
        empty,
        empty,
    ]


def test_unfused_sites_are_gated_static_and_guarded(monkeypatch):
    """The whitelisted unfused route (d_pair 384, 12 heads: the generic front + the standalone V^T pass): below the S gate the default core runs (bit-identical),
    above it the FP8 core with a per-site static scale (calibrated on the first call); a V that outgrows the scale sets the overflow flag."""
    from miniworld_engine.kernels.integrations import triattn_h100_sm90 as integ
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    monkeypatch.setitem(hs.FP8_PV_WHITELIST, "unfused_probe", frozenset({"unfused"}))
    run = lambda args: integ._infer(integ.GENERIC, args, 1e-5, 12, 32, False)
    short, long_ = _module_args(384, 12, 32, 256), _module_args(384, 12, 32, 1024)
    ref_short, ref_long = run(short), run(long_)
    try:
        assert hs.enable_fp8_pv_for("unfused_probe") is True
        assert hs.FP8_PV_UNFUSED and not hs.FP8_PV_FUSED and hs.FP8_PV_MIN_S == hs.FP8_PV_UNFUSED_MIN_S
        assert torch.equal(run(short), ref_short)  # below the gate: the fp16 core
        run(long_)  # calibration call (dynamic scales)
        got = run(long_)  # the single-pass static-scale route
        assert not hs.take_overflow()
        err = ((got.float() - ref_long.float()).norm() / (ref_long.float() - long_[0].float()).norm()).item()  # of the module's update
        assert 0.002 < err < 0.06, err
        hot = [long_[0], long_[1] * 50.0, *long_[2:]]  # the LayerNorm gain x 50: V outgrows the calibrated scale
        run(hot)
        assert hs.take_overflow()
        sites = {k for k in hs._VS_AUTO if isinstance(k, tuple) and k[0] == "vq"}
        assert len(sites) == 1
    finally:
        hs.disable_fp8_pv()
        for k in [k for k in hs._VS_AUTO if isinstance(k, tuple) and k[0] == "vq"]:
            del hs._VS_AUTO[k]
    assert torch.equal(run(long_), ref_long)  # off again: the default route


@pytest.mark.parametrize(
    ("dp", "heads", "dh", "length", "hl"),
    [
        (64, 4, 32, 256, False),
        (64, 4, 32, 256, True),
        (64, 2, 32, 384, False),
        (64, 4, 16, 256, False),
        (64, 4, 16, 384, True),
        (256, 8, 32, 256, False),
    ],
)
def test_front_made_v_planes_equal_the_standalone_pass(monkeypatch, dp, heads, dh, length, hl):
    """The fronts that write the e4m3 V^T planes themselves (``FP8_FRONT_VT``: the d_pair-64 one-kernel fronts, the wide d_pair-256 front's V CTA) give the module output of the
    standalone pass bit for bit (same fp16-rounded V, same static scale, same cvt); the key mask is padded."""
    from miniworld_engine.kernels.integrations import triattn_h100_sm90 as integ
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    monkeypatch.setitem(hs.FP8_PV_WHITELIST, "front_probe", frozenset({"unfused"}))
    kind = integ.FUSED if (dp == 256 and heads == 8) else integ.GENERIC
    args = _module_args(dp, heads, dh, length)
    args[9][:, -37:] = False
    run = lambda: integ._infer(kind, args, 1e-5, heads, dh, False)
    try:
        assert hs.enable_fp8_pv_for("front_probe", hl=hl) is True
        hs.FP8_PV_MIN_S, hs.FP8_PV_FRONT_MIN_S = {}, {}  # no gate: the short lengths take the FP8 core too
        hs.FP8_FRONT_VT = False
        run()  # calibration (dynamic scales)
        ref = run().clone()
        hs.FP8_FRONT_VT = True
        got = run().clone()
        assert not hs.take_overflow()
        assert torch.equal(ref.view(torch.int16), got.view(torch.int16))
    finally:
        hs.FP8_FRONT_VT = True
        hs.disable_fp8_pv()
        for k in [k for k in hs._VS_AUTO if isinstance(k, tuple) and k[0] == "vq"]:
            del hs._VS_AUTO[k]
        hs._VINV.clear()
        hs._VSB.clear()


def test_front_gates_are_below_the_standalone_gates():
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    for (_dp, _h, ch), gate in hs.FP8_PV_UNFUSED_FRONT_MIN_S.items():
        assert gate <= hs.FP8_PV_UNFUSED_MIN_S[ch]


def test_vscale_floor_and_zero_heads():
    """The static V scale is never below 2^-30: a head whose calibration absmax is ~0 (all zero, or the dynamic pass's 1e-30 clamp) takes 1, not
    2^-99 .. 2^-107 (which saturated the e4m3 cast on any later V of that head); every other head keeps the power of two above headroom x amax / 448."""
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    amax = torch.tensor([0.0, 1e-30 * 448.0, 1e-12, 3.0, 448.0])
    got = hs.vscale_from_amax(amax, 4.0)
    assert got.tolist() == [1.0, 1.0, 1.0, 2.0**-5, 4.0]
    assert bool((got >= hs.VSCALE_FLOOR).all())
    t = torch.tensor([3.0, 0.07, 200.0]) * 4.0 / 448.0
    assert torch.equal(hs.vscale_from_amax(torch.tensor([3.0, 0.07, 200.0]), 4.0), torch.exp2(torch.ceil(torch.log2(t))))


def test_static_scale_of_a_zero_head_does_not_saturate_later():
    """quantize_v_static calibrated on a V with one all-zero head: a later V with ordinary values in that head runs without the overflow flag."""
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    torch.manual_seed(5)
    v = torch.randn(1, 8, 256, 4, 32, device="cuda").half()
    v[..., 2, :] = 0
    site = ("test-zero-head", 4, 32)
    try:
        hs.quantize_v_static(v, site)  # first call: calibrates
        key = ("vq", site, 4, 32)
        assert float(hs._VS_AUTO[key][2]) == 1.0
        ovf = hs._OVF.setdefault(v.device, torch.zeros(1, dtype=torch.int32, device=v.device))
        ovf.zero_()
        v2 = v.clone()
        v2[..., 2, :] = torch.randn_like(v2[..., 2, :]) * 3
        vt, vs = hs.quantize_v_static(v2, site)
        torch.cuda.synchronize()
        assert int(ovf.item()) == 0
        assert float(vs.view(-1)[2]) == 1.0
    finally:
        hs._VS_AUTO.pop(("vq", site, 4, 32), None)
        hs._drop_derived(("vq", site, 4, 32))

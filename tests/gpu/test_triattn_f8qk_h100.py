"""EXPERIMENTAL q / k on e4m3 for the FP8 TriangleAttention route (``h100_sm90.FP8_QK``, ``triattn_h16_sm90.cu`` with ``-DTA_F8QK=1``,
``tri_front_sm90<3, VTW, true>``): opt-in, default off.

- the switch is off by default and changes nothing while the FP8 PV route is off;
- the F8QK core (S += Q K^T as one m64n128k32 e4m3 wgmma, fp16 accumulator) on e4m3 q / k equals the F8 PV core on the same values dequantized to
  fp16 up to the score accumulation order (mean |difference| tiny; single elements move by one e4m3 step of P);
- the product-1 per-head scale (q / sq, k * sq) and its refusal when q / k do not both fit;
- the module: front-made e4m3 q / k agree with the torch pass, the update error stays inside the e4m3 band, no NaN, no overflow on in-range inputs.
"""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


pytestmark.append(pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)"))


def test_switch_is_off_by_default():
    from foldjet_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    assert hs.FP8_QK is False


def test_qk_scale_is_product_one_and_refuses_out_of_range():
    from foldjet_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    sq = hs.qk_scale_from_amax(torch.tensor([16.0, 1.0, 4.0, 0.0]), torch.tensor([1.0, 16.0, 4.0, 3.0]))
    assert sq is not None
    assert torch.equal(sq, torch.exp2(torch.round(torch.log2(sq))))  # powers of two
    assert sq[0] == 4.0 and sq[1] == 0.25 and sq[2] == 1.0
    assert hs.qk_scale_from_amax(torch.tensor([400.0]), torch.tensor([400.0])) is None  # 400 * 400 > (448 / 2)^2


@pytest.mark.parametrize(("heads", "length"), [(4, 256), (8, 512), (12, 256)])
def test_f8qk_core_matches_the_f8pv_core_on_dequantized_qk(heads, length):
    from foldjet_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

    torch.manual_seed(heads + length)
    n = 4
    q = (torch.randn(1, n, length, heads, 32, device="cuda") * 1.5).half()
    k = (torch.randn(1, n, length, heads, 32, device="cuda") * 0.7).half()
    v = torch.randn(1, n, length, heads, 32, device="cuda").half()
    sb = (torch.randn(1, heads, length, length, device="cuda") * 2).half()
    sq = hs.qk_scale_from_amax(q.view(-1, heads, 32).abs().amax((0, 2)).float(), k.view(-1, heads, 32).abs().amax((0, 2)).float())
    assert sq is not None
    q8, k8 = hs.quantize_qk_torch(q, k, sq)
    s5 = sq.view(1, 1, 1, heads, 1)
    qd, kd = (q8.float() * s5).half(), (k8.float() / s5).half()  # exact (powers of two)
    vt, vs = hs.quantize_v(v)
    ref = hs._core_f8pv().triattn_fwd_h16(qd, kd, kd, sb, 32**-0.5, False, None, True, None, None, False, vt, vs)[0].float()
    got = hs._core_f8qk().triattn_fwd_h16(q8, k8, k8, sb, 32**-0.5, False, None, True, None, None, False, vt, vs)[0].float()
    d = (got - ref).abs()
    assert not torch.isnan(got).any()
    # a layout error would be O(1); the e4m3 probabilities turn a one-ulp score difference into one P step (~6 %) on single elements
    assert d.mean().item() < 1e-2 * ref.abs().mean().item()
    assert d.max().item() < 0.1 * ref.abs().max().item()
    # against an fp32 attention of the same (dequantized) q / k and V: the F8QK core is as close as the F8 PV core
    lg = (torch.einsum("bnjhd,bnkhd->bnhjk", qd.float(), kd.float()) + sb.float()[:, None]) * 32**-0.5
    f32 = torch.einsum("bnhjk,bnkhd->bnjhd", torch.softmax(lg, -1), v.float())
    e_qk, e_pv = ((got - f32).norm() / f32.norm()).item(), ((ref - f32).norm() / f32.norm()).item()
    assert e_qk < 1.1 * e_pv + 1e-3, (e_qk, e_pv)


def test_module_route_front_and_torch_pass(monkeypatch):
    from foldjet_engine.kernels.triangle_attention.cuda import h100_sm90 as hs
    from foldjet_engine.modules.triangle import TriangleAttention

    torch.manual_seed(0)
    L = 512
    m = TriangleAttention(128, 4, d_hidden=128, starting=True, master_weights=True, backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 128**-0.5 if p.dim() > 1 else 1 + 0.1 * torch.randn_like(p))
    x = torch.randn(1, L, L, 128, device="cuda").bfloat16()
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    mask[:, L - 3 :] = False
    ref = copy.deepcopy(m).double()
    ref.set_backend("pytorch")
    with torch.no_grad():
        y64 = ref(x.double(), mask)
        base = m(x, mask)
        monkeypatch.setattr(hs, "FP8_QK", True)  # FP8 PV off: nothing changes
        assert torch.equal(m(x, mask), base)
        for name, val in (("FP8_PV", True), ("FP8_PV_FUSED", True), ("FP8_VSCALE_AUTO", True), ("FP8_PV_FUSED_MIN_S", 0)):
            monkeypatch.setattr(hs, name, val)
        m(x, mask)  # V scale calibration + q / k scale calibration
        m(x, mask)
        front = m(x, mask)
        monkeypatch.setattr(hs, "FP8_QK_FRONT", False)
        torch_pass = m(x, mask)
        torch.cuda.synchronize()
        assert not hs.take_overflow()
    upd = (y64 - x.double()).pow(2).mean().sqrt()
    e_front = ((front.double() - y64).pow(2).mean().sqrt() / upd).item()
    e_torch = ((torch_pass.double() - y64).pow(2).mean().sqrt() / upd).item()
    e_def = ((base.double() - y64).pow(2).mean().sqrt() / upd).item()
    assert not torch.isnan(front).any()
    assert e_def < 0.03
    assert e_front < 0.07, e_front
    assert abs(e_front - e_torch) < 0.1 * e_torch  # the front's q / k bytes are the pass's up to the fp16 rounding before the pass
    assert e_front > 1.5 * e_def  # (the variant really runs e4m3)

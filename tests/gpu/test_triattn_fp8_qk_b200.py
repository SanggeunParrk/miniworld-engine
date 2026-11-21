"""B200 (sm_100a) TriangleAttention FP8: e4m3 q / k on top of the e4m3 P / V route (``b200_fp8``, ``-DTF_Q8=1`` build of ``b200_sources/triattn_f8_sm100.cu``).

OPT-IN, default OFF (the shipped S gates ``b200_fp8.QK_MINS_SHIPPED`` are set from the measurements of docs/b200/fp8.md "TriAttn e4m3 q / k"; the tests open them).
FP8 off stays bit-identical: no q / k pass, no e4m3 call, counters 0.

- Kernel: e4m3 q / k core against the fp64 gold (random q / k / v / bias, masked keys, large logits): finite, error within a small multiple of the P / V-only core's.
- Module: accuracy against the fp32 PyTorch statements at the head-dim-32 geometries (fused d_pair 128 / unfused 64 / 256 / 384), mask, batch, ragged length.
- Saturation: q / k far above the calibration absmax set the guard flag; ``fp8.check()`` widens the scales in place (``qv``) and the redone call is clean.
- CUDA-graph replay == eager (calibrated site); an uncalibrated capture declines the q / k part; graph == eager.
- Gates / status: a closed gate means not served (``status()["triattn"]["qk"]`` False, counters 0); route stats keys.
"""

import copy

import pytest
import torch

from miniworld_engine import fp8
from miniworld_engine.kernels.triangle_attention.cuda import b200_fp8
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as bt
from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs
from miniworld_engine.measure.config import randomize
from miniworld_engine.modules.triangle import TriangleAttention, TriangleAttentionAF3

pytestmark = [pytest.mark.gpu]

L2E = 1.4426950408889634
_SHIPPED = dict(b200_fp8.QK_MINS_SHIPPED)

#: judge form max|y - y64| / max|y64| of the module (the H100 FP8 PV tolerance 8e-3; the e4m3 q / k crosses it on the H100 too: 0.85-0.95 %): the QK route's bound here
TOL_QK = 1.5e-2  # measured judge errors of the q / k route: 0.0074-0.0131 over every geometry at L384 / L1024 (docs/b200/fp8.md "TriAttn e4m3 q / k"); the P / V-only route 0.0047-0.0087


def _needs_b200() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("needs a B200 (sm_100)")


@pytest.fixture(autouse=True)
def _switches():
    hs.disable_fp8_pv()
    gates = b200_fp8.MIN_S, b200_fp8.MIN_S64, dict(b200_fp8.QK_MINS_SHIPPED), b200_fp8._QK_KINDS
    b200_fp8.MIN_S = b200_fp8.MIN_S64 = 256
    for k in b200_fp8.QK_MINS_SHIPPED:
        b200_fp8.QK_MINS_SHIPPED[k] = 256
    yield
    b200_fp8.MIN_S, b200_fp8.MIN_S64 = gates[0], gates[1]
    b200_fp8.QK_MINS_SHIPPED.update(gates[2])
    b200_fp8._QK_KINDS = gates[3]
    hs.disable_fp8_pv()


def _on(qk: bool, kinds=("unfused",)) -> None:
    hs.FP8_PV, hs.FP8_PV_FUSED, hs.FP8_PV_UNFUSED, hs.FP8_VSCALE_AUTO = True, True, True, True
    hs.FP8_QK = qk
    b200_fp8._QK_KINDS = frozenset(kinds) if qk else frozenset()


def _module(cls, d, h, kw, starting):
    torch.manual_seed(0)
    m = cls(d, h, starting=starting, master_weights=True, backend="cuda", **kw).cuda().eval()
    with torch.no_grad():
        randomize(m, torch.Generator(device="cuda").manual_seed(1))
    return m


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))


def _zero() -> None:
    for k in b200_fp8.STATS:
        b200_fp8.STATS[k] = 0


L = 384
#: (class, d_pair, n_head, kwargs, the front writes e4m3 q / k): the wide fronts (d_pair 256 / 384) write bf16 q / k, the standalone pass is not served there
CASES = [
    (TriangleAttention, 128, 4, {}, True),
    (TriangleAttentionAF3, 128, 4, {}, True),
    (TriangleAttention, 64, 4, {"d_hidden": 128}, True),
    (TriangleAttention, 64, 2, {}, True),
    (TriangleAttention, 256, 8, {}, False),
    (TriangleAttention, 384, 12, {}, False),
]


# ------------------------------------------------------------------------------------------------------------------------------ kernel level
def _make(N, S, H, sq=1.0, sb=1.0, seed=0, ntail=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    q, k, v = ((r(1, N, S, H, 32) * c).bfloat16() for c in (sq, sq, 1.0))
    bias = r(1, H, S, S) * sb
    if ntail:
        bias[..., S - ntail :] = -float("inf")
    return q, k, v, bias


def _ref64(q, k, v, bias, scale):
    lg = torch.einsum("bnqhc,bnkhc->bnhqk", q.double(), k.double()) * scale + bias.double()[:, None]
    return torch.einsum("bnhqk,bnkhc->bnqhc", torch.softmax(lg, -1), v.double())


def _scales(q, k):
    aq = torch.maximum(q.amax(dim=(0, 1, 2, 4)), -q.amin(dim=(0, 1, 2, 4))).float()
    ak = torch.maximum(k.amax(dim=(0, 1, 2, 4)), -k.amin(dim=(0, 1, 2, 4))).float()
    sq = hs.qk_scale_from_amax(aq, ak, b200_fp8.QK_HEADROOM)
    assert sq is not None
    one = torch.ones_like(sq)
    return (1.0 / sq).contiguous(), sq.contiguous(), one


def _core(q, k, v, bias, scale, with_qk):
    amax = torch.maximum(v.amax(dim=(0, 1, 2, 4)), -v.amin(dim=(0, 1, 2, 4))).float()
    vinv = torch.exp2(torch.floor(torch.log2(448.0 / (4 * amax)))).float().contiguous()
    guard = torch.zeros(2, dtype=torch.int32, device="cuda")
    bhb = (bias * L2E).bfloat16().contiguous()
    qk = None
    if with_qk:
        qa0, ka0, one = _scales(q, k)
        qk = (qa0, ka0, one, one, one)
    o, stat = b200_fp8.core_forward(q, k, v, bhb, scale, vinv, guard, qk=qk)
    return o, stat, guard


@pytest.mark.parametrize("S,H,ntail", [(256, 4, 0), (512, 4, 0), (384, 4, 70), (256, 12, 0)])
def test_core_accuracy(S, H, ntail) -> None:
    """e4m3 q / k core against the fp64 gold; the P / V-only core is the reference of the error budget."""
    _needs_b200()
    q, k, v, bias = _make(8, S, H, ntail=ntail)
    scale = 32**-0.5
    o64 = _ref64(q, k, v, bias, scale)
    o_pv, _, _ = _core(q, k, v, bias, scale, False)
    o_qk, stat, guard = _core(q, k, v, bias, scale, True)
    e_pv = (o_pv.double() - o64).pow(2).mean().sqrt() / o64.pow(2).mean().sqrt()
    e_qk = (o_qk.double() - o64).pow(2).mean().sqrt() / o64.pow(2).mean().sqrt()
    assert torch.isfinite(o_qk).all()
    assert float(e_qk) <= 3.0 * float(e_pv) + 5e-3, (float(e_qk), float(e_pv))
    assert int(guard[1]) == 0  # nothing saturated at the calibrated scales
    assert int(stat[0]) == 0  # no row needed the exact recompute


def test_core_stress_logits_and_masks() -> None:
    """Large logits (q, k x6), a 130-key mask, rows with few live keys: finite, close to the gold in max norm."""
    _needs_b200()
    for sq, sb, ntail in ((6.0, 1.0, 0), (1.0, 25.0, 130), (3.0, 4.0, 250)):
        q, k, v, bias = _make(8, 256, 4, sq=sq, sb=sb, ntail=ntail)
        scale = 32**-0.5
        o64 = _ref64(q, k, v, bias, scale)
        o_qk, stat, _ = _core(q, k, v, bias, scale, True)
        assert torch.isfinite(o_qk).all(), (sq, sb, ntail)
        err = float((o_qk.double() - o64).abs().max())
        if sq == 6.0:
            # q, k x 6: logits ~36 sigma, a softmax that is one key per row: the 6 % e4m3 rounding of q / k moves the logit by ~2 and flips the winning key on some rows (measured max
            # error 4.49 = 1.04 x max|o|: a wrong V row, not a blow-up); only "finite and bounded by the size of o" is asserted there
            assert err <= 1.2 * float(o64.abs().max()), (sq, sb, ntail, err)
        else:
            assert err <= 0.35 * float(o64.abs().max()) + 0.2, (sq, sb, ntail, err)


# ----------------------------------------------------------------------------------------------------------------------------- module level
@pytest.mark.parametrize("cls,d,h,kw,front", CASES, ids=lambda v: getattr(v, "__name__", str(v)))
@pytest.mark.parametrize("starting", [True, False])
def test_module_accuracy(cls, d, h, kw, front, starting) -> None:
    _needs_b200()
    m = _module(cls, d, h, kw, starting)
    ref = copy.deepcopy(m).set_backend("pytorch").float()
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(1, L, L, d, device="cuda", generator=g).bfloat16()
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.1
    mask[:, 0] = True
    with torch.no_grad():
        yr = ref(x.float(), mask)
        _on(False)
        y_pv = m(x, mask).clone()
        _on(True)
        _zero()
        m(x, mask)  # the first call calibrates the q / k scales (P / V core)
        _zero()
        y_qk = m(x, mask).clone()  # the second: the front writes e4m3 q / k
        n = dict(b200_fp8.STATS)
    assert torch.isfinite(y_qk).all()
    if not front:
        assert n.get("qk", 0) == 0 and torch.equal(y_qk, y_pv), n  # no e4m3 front there: the P / V core, the P / V-only route's bytes
        return
    assert n.get("qk", 0) == 1 and n.get("qk_front", 0) == 1, n
    assert _rel(y_qk, yr) <= TOL_QK, (_rel(y_qk, yr), _rel(y_pv, yr))
    assert _rel(y_qk, yr) >= _rel(y_pv, yr) * 0.9  # the q / k rounding is there (not suspiciously equal to the P / V-only route)


def test_ragged_batched() -> None:
    _needs_b200()
    m = _module(TriangleAttention, 128, 4, {}, False)
    ref = copy.deepcopy(m).set_backend("pytorch").float()
    g = torch.Generator(device="cuda").manual_seed(3)
    Lr = 300
    x = torch.randn(2, Lr, Lr, 128, device="cuda", generator=g).bfloat16()
    mask = torch.rand(2, Lr, device="cuda", generator=g) > 0.1
    mask[:, 0] = True
    mask[1, Lr - 9 :] = False
    with torch.no_grad():
        yr = ref(x.float(), mask)
        _on(True)
        m(x, mask)
        _zero()
        y1 = m(x, mask).clone()
    assert b200_fp8.STATS.get("qk_front", 0) == 1, b200_fp8.STATS
    assert torch.isfinite(y1).all() and _rel(y1, yr) <= TOL_QK, _rel(y1, yr)


def test_off_is_untouched() -> None:
    """FP8 off, and FP8 PV on with the q / k switch off: no q / k pass, no e4m3 q / k call; off bytes repeat and return after enabling."""
    _needs_b200()
    m = _module(TriangleAttention, 128, 4, {}, True)
    x = torch.randn(1, L, L, 128, device="cuda").bfloat16()
    with torch.no_grad():
        _zero()
        y0 = m(x).clone()
        assert torch.equal(y0, m(x))
        assert all(v == 0 for v in b200_fp8.STATS.values()), b200_fp8.STATS
        _on(False)
        _zero()
        m(x)
        assert b200_fp8.STATS.get("qk", 0) == 0 and b200_fp8.STATS.get("qk_calibrations", 0) == 0
        hs.disable_fp8_pv()
        assert torch.equal(y0, m(x))


def test_closed_gate_is_not_served() -> None:
    _needs_b200()
    m = _module(TriangleAttention, 128, 4, {}, True)
    x = torch.randn(1, 256, 256, 128, device="cuda").bfloat16()
    for k in b200_fp8.QK_MINS_SHIPPED:
        b200_fp8.QK_MINS_SHIPPED[k] = 1 << 30
    with torch.no_grad():
        _on(True)
        assert hs.FP8_QK and b200_fp8.FP8_QK is False  # the switch is on, the gate is closed: status qk False
        _zero()
        y1 = m(x).clone()
        assert b200_fp8.STATS.get("qk", 0) == 0 and b200_fp8.STATS.get("qk_calibrations", 0) == 0
        _on(False)
        assert torch.equal(y1, m(x))  # = the P / V-only route's bytes


def test_graph_equals_eager_and_uncalibrated_capture() -> None:
    _needs_b200()
    m = _module(TriangleAttention, 128, 4, {}, True)
    x = torch.randn(1, L, L, 128, device="cuda").bfloat16()
    with torch.no_grad():
        _on(True)
        _zero()
        m(x)  # calibrates the site (eager, P / V core)
        y = m(x).clone()  # the e4m3 q / k front
        assert b200_fp8.STATS.get("qk_front", 0) == 1
        gr = torch.cuda.CUDAGraph()
        for _ in range(2):
            m(x)
        torch.cuda.synchronize()
        with torch.cuda.graph(gr):
            yg = m(x)
        gr.replay()
        torch.cuda.synchronize()
        assert torch.equal(yg, y)
        # an uncalibrated site inside a capture takes the default route (counted): the q / k part is declined with it
        m2 = _module(TriangleAttention, 128, 4, {}, True)
        hs.disable_fp8_pv()
        y2 = m2(x).clone()
        _on(True)
        _zero()
        gr2 = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(gr2):
            yg2 = m2(x)
        gr2.replay()
        torch.cuda.synchronize()
        assert b200_fp8.STATS.get("capture_fallbacks", 0) >= 1
        assert torch.equal(yg2, y2)
        # a calibrated V site whose q / k scales were never made (calibrated by the P / V route first): the capture runs the P / V core with the q / k part declined
        m3 = _module(TriangleAttention, 128, 4, {}, True)
        _on(False)
        m3(x)
        _on(True)
        _zero()
        gr3 = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(gr3):
            m3(x)
        assert b200_fp8.STATS.get("qk", 0) == 0 and b200_fp8.STATS.get("qk_declined", 0) == 1, b200_fp8.STATS


def test_saturation_guard_qk() -> None:
    """q far above the calibration absmax (the Q weight x 200): the guard fires, ``fp8.check()`` widens
    the scales in place, the redone call is finite and clean."""
    _needs_b200()
    m = _module(TriangleAttention, 128, 4, {}, True)
    x = torch.randn(1, L, L, 128, device="cuda").bfloat16()
    with torch.no_grad():
        _on(True)
        m(x)  # calibration (P / V core)
        m(x)  # the e4m3 q / k front
        assert not fp8.check()
        s0 = {k: (v.qv.clone() if v.qk else None) for k, v in b200_fp8._SITES.items()}
        assert any(v is not None for v in s0.values())
        m.to_query.weight.mul_(200.0)  # q x 200 at the calibrated static scales: q8 leaves e4m3 (the Q weight is not the V weight: no recalibration of the site)
        m(x)
        assert b200_fp8.overflow_peak() > 448.0
        for _ in range(6):
            if not fp8.check():
                break
            y = m(x)
        else:
            raise AssertionError("still saturating after 6 rescales")
        assert torch.isfinite(m(x)).all()
        assert any(bool((v.qv > 1.0).any()) for v in b200_fp8._SITES.values() if v.qk)  # the q / k scales were widened in place (qv), not recalibrated


def test_route_stats_keys() -> None:
    _needs_b200()
    st = fp8.route_stats()["triattn"]
    for k in ("pv_fused", "pv_front", "calibrations"):
        assert k in st


@pytest.mark.parametrize("model,qk", [("af3", True), ("protenix1", True), ("openfold3", True), ("rosettafold3", True), ("opendde", False), ("chai1", False), ("boltz2", False), (None, False)])
def test_status_qk_follows_whitelist_and_gate(model, qk) -> None:
    """``status()["triattn"]["qk"]`` is True only for the models whose whitelist lists q / k, now that the route is built and its gate is open; the unlisted models stay False."""
    _needs_b200()
    b200_fp8.QK_MINS_SHIPPED.update(_SHIPPED)  # as shipped (the fixture opened the gates to 256)
    fp8.enable("safe", model=model)
    try:
        assert fp8.status()["triattn"]["qk"] is qk
    finally:
        fp8.disable()
    assert fp8.status()["triattn"]["qk"] is False


def test_shipped_gates_are_the_measured_ones() -> None:
    """The gates as shipped (docs/b200/fp8.md "TriAttn e4m3 q / k"): read from the module, not from an environment variable of this process."""
    import os

    if "MINIWORLD_TRIATTN_B200_FP8_QK_MINS" not in os.environ:
        assert set(_SHIPPED) == {"fused", "unfused", "d64"}
        assert _SHIPPED["fused"] == 1024 and _SHIPPED["unfused"] == 1024 and _SHIPPED["d64"] >= 1 << 30  # measured: the front-written e4m3 rows win from S = 1024; the 64-channel core has no front

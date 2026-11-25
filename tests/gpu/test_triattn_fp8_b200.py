"""B200 (sm_100a) TriangleAttention FP8 route: e4m3 P and V on tcgen05 ``kind::f8f6f4`` (``b200_fp8``, ``b200_sources/triattn_f8_sm100.cu``).

OPT-IN: the default route is untouched (bit tests live in the ``bitscan`` tools of docs/b200/fp8.md; here: off = no route call, same bytes as before).

- FP8 off: no e4m3 call, counters 0, repeated bytes; enabling then disabling returns to the same bytes.
- Accuracy against the fp32 PyTorch statements (judge form max|y - y64| / max|y64| <= 8e-3, the H100 FP8 PV tolerance) at every head-dim-32 geometry,
  starting and ending node, key mask, ragged length, batch of two.
- Large logits (Wq / Wk x3, bias x25): finite, no NaN / inf, error against the fp32 statements within a small multiple of the bf16 route's.
- CUDA-graph replay == the eager call (calibrated site); an uncalibrated capture takes the default route (counted).
- Saturation guard: a V far above the calibration absmax sets the guard, ``take_overflow`` / ``rescale_all`` clear it and the redone call is finite.
- Whitelist: only models of ``fp8.TRIATTN_MODELS`` serve it (Boltz-2 / IntelliFold2 / no model: the default route, same bytes); route stats keys.
"""

import copy

import pytest
import torch

from foldjet_engine import fp8
from foldjet_engine.kernels.triangle_attention.cuda import b200_fp8
from foldjet_engine.kernels.triangle_attention.cuda import h100_sm90 as hs
from foldjet_engine.measure.config import randomize
from foldjet_engine.modules.triangle import TriangleAttention, TriangleAttentionAF3, TriangleAttentionChai1

pytestmark = [pytest.mark.gpu]

_SHIPPED_GATES = (b200_fp8.MIN_S, b200_fp8.MIN_S64)  # as imported (the autouse fixture opens them for the small test sizes)

#: the judge's FP8 PV tolerance on the H100 (docs/fp8.md: 4.8-7.8e-3 for the 64-channel core, 0.43-0.66 % at d_pair 128)
TOL = 8e-3


def _needs_b200() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("needs a B200 (sm_100)")


@pytest.fixture(autouse=True)
def _switches():
    """Every test starts and ends with the TriAttn FP8 switches off (the H100 module's, which ``fp8.enable_triattn`` sets and ``b200_fp8`` reads); the S gates are opened to 256 for the small test sizes."""
    hs.disable_fp8_pv()
    gates = b200_fp8.MIN_S, b200_fp8.MIN_S64
    b200_fp8.MIN_S = b200_fp8.MIN_S64 = 256
    yield
    b200_fp8.MIN_S, b200_fp8.MIN_S64 = gates
    hs.disable_fp8_pv()


def _on(fused: bool = True, unfused: bool = True) -> None:
    hs.FP8_PV, hs.FP8_PV_FUSED, hs.FP8_PV_UNFUSED, hs.FP8_VSCALE_AUTO = True, fused, unfused, True


def _module(cls, d, h, kw, starting):
    torch.manual_seed(0)
    m = cls(d, h, starting=starting, master_weights=True, backend="cuda", **kw).cuda().eval()
    with torch.no_grad():
        randomize(m, torch.Generator(device="cuda").manual_seed(1))
    return m


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))


def _stats(*keys):
    return {k: b200_fp8.STATS[k] for k in keys}


def _zero() -> None:
    for k in b200_fp8.STATS:
        b200_fp8.STATS[k] = 0


L = 384

#: (class, d_pair, n_head, extra kwargs, call-site kind): the head-dim-32 geometries of the requirement rows
CASES = [
    (TriangleAttention, 128, 4, {}, "pv_fused"),
    (TriangleAttentionAF3, 128, 4, {}, "pv_fused"),
    (TriangleAttention, 64, 4, {"d_hidden": 128}, "pv_front"),
    (TriangleAttention, 64, 2, {}, "pv_front"),
    (TriangleAttention, 256, 8, {}, "pv_front"),
    (TriangleAttention, 384, 12, {}, "pv_front"),
]


@pytest.mark.parametrize("cls,d,h,kw,kind", CASES, ids=lambda v: getattr(v, "__name__", str(v)))
@pytest.mark.parametrize("starting", [True, False])
def test_accuracy_against_fp32(cls, d, h, kw, kind, starting) -> None:
    _needs_b200()
    m = _module(cls, d, h, kw, starting)
    ref = copy.deepcopy(m).set_backend("pytorch").float()
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(1, L, L, d, device="cuda", generator=g).bfloat16()
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.1
    mask[:, 0] = True
    with torch.no_grad():
        y0 = m(x, mask).clone()
        yr = ref(x.float(), mask)
        _on()
        _zero()
        y1 = m(x, mask).clone()
        n = _stats("pv_fused", "pv_front", "calibrations")
    assert n[kind] == 1 and n["calibrations"] == 1, n
    assert torch.isfinite(y1).all()
    assert _rel(y1, yr) <= TOL, (_rel(y1, yr), _rel(y0, yr))
    assert _rel(y1, yr) >= _rel(y0, yr) * 0.9  # (it is the e4m3 route: not suspiciously equal to the bf16 one)


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
        _on()
        _zero()
        y1 = m(x, mask).clone()
    assert b200_fp8.STATS["pv_fused"] == 1
    assert torch.isfinite(y1).all() and _rel(y1, yr) <= TOL, _rel(y1, yr)


def test_off_is_untouched() -> None:
    _needs_b200()
    m = _module(TriangleAttention, 128, 4, {}, True)
    x = torch.randn(1, L, L, 128, device="cuda").bfloat16()
    with torch.no_grad():
        _zero()
        y0 = m(x).clone()
        assert torch.equal(y0, m(x))
        assert all(v == 0 for v in b200_fp8.STATS.values()), b200_fp8.STATS
        _on()
        m(x)
        hs.disable_fp8_pv()
        assert torch.equal(y0, m(x))  # back on the default route: the same bytes


@pytest.mark.parametrize("which", ["qk", "bias"])
def test_large_logits_finite(which) -> None:
    """Wq / Wk x3 and bias x25 (the n7 qs_big stress): no NaN / inf; the error against the fp32 statements within a small multiple of the bf16 route's."""
    _needs_b200()
    from foldjet_engine.modules import triangle as T

    torch.manual_seed(5)
    m = T.TriangleAttention(d_pair=128, n_head=4, starting=True, backend="cuda").cuda().float().eval()
    r = T.TriangleAttention(d_pair=128, n_head=4, starting=True, backend="pytorch").cuda().float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.normal_(0, 0.5 if p.dim() == 2 else 0.2)
            if which == "qk" and n in ("to_query.weight", "to_key.weight"):
                p.mul_(3.0)
            if which == "bias" and n == "to_bias.weight":
                p.mul_(25.0)
        r.load_state_dict(m.state_dict())
    x = (torch.randn(1, 384, 384, 128, device="cuda") * 2).bfloat16()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        y0, yr = m(x).float(), r(x).float()
        _on()
        y1 = m(x).float()
    assert torch.isfinite(y1).all()
    e0, e1 = float((y0 - yr).abs().max()), float((y1 - yr).abs().max())
    assert e1 <= 4.0 * max(e0, 1e-2) + 0.05 * float(yr.abs().max()), (e0, e1)


def test_graph_equals_eager_and_uncalibrated_capture() -> None:
    _needs_b200()
    m = _module(TriangleAttention, 128, 4, {}, True)
    x = torch.randn(1, L, L, 128, device="cuda").bfloat16()
    with torch.no_grad():
        _on()
        _zero()
        y = m(x).clone()  # calibrates the site (eager)
        gr = torch.cuda.CUDAGraph()
        for _ in range(2):
            m(x)
        torch.cuda.synchronize()
        with torch.cuda.graph(gr):
            yg = m(x)
        gr.replay()
        torch.cuda.synchronize()
        assert torch.equal(yg, y)
        # an uncalibrated site inside a capture takes the default route and is counted
        m2 = _module(TriangleAttention, 128, 4, {}, True)
        hs.disable_fp8_pv()
        y2 = m2(x).clone()
        _on()
        _zero()
        for _ in range(2):
            pass
        gr2 = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(gr2):
            yg2 = m2(x)
        gr2.replay()
        torch.cuda.synchronize()
        assert b200_fp8.STATS.get("capture_fallbacks", 0) >= 1
        assert torch.equal(yg2, y2)


def test_saturation_guard() -> None:
    _needs_b200()
    m = _module(TriangleAttention, 128, 4, {}, True)
    x = torch.randn(1, L, L, 128, device="cuda").bfloat16()
    with torch.no_grad():
        _on()
        m(x)  # calibrate on this x (the LayerNorm hides the scale of x: the value weight is what saturates)
        assert not fp8.check()  # nothing saturated: ``fp8.check`` (the lazy guard of every route) sees this route's flag through the H100 module's registry
        assert b200_fp8.overflow_peak() < 448.0
        m.to_value.weight.mul_(64.0)  # the weights changed in place (``_version``): the site recalibrates on the new V, nothing saturates
        m(x)
        assert b200_fp8.overflow_peak() < 448.0
        # a stale scale: the scale tensors of a calibration against a 64x smaller V (what a changed V looks like to a static scale)
        for s in b200_fp8._SITES.values():
            s.vinv.mul_(64.0)
            s.vsc.copy_(1.0 / s.vinv)
        m(x)
        assert b200_fp8.overflow_peak() > 448.0
        for _ in range(4):  # ``fp8.check()``: the guard fired and widened every scale in place (rescale_all, x4); redo (at most 3 times in the docs: 64x = three x4 steps)
            if not fp8.check():
                break
            y = m(x)
        else:
            raise AssertionError("still saturating after 4 rescales")
        assert torch.isfinite(m(x)).all()


@pytest.mark.parametrize(
    "model,fires",
    [("af3", True), ("protenix1", True), ("openfold3", True), ("rosettafold3", True), ("opendde", False), ("boltz2", False), ("intellifold2", False), (None, False)],
)
def test_whitelist(model, fires) -> None:
    """``fp8.enable("safe", model=...)``: the route fires only for the whitelisted call-site kinds of the model (d_pair-128 = ``fused``); other models run the default route (same bytes)."""
    _needs_b200()
    m = _module(TriangleAttention, 128, 4, {}, True)
    x = torch.randn(1, L, L, 128, device="cuda").bfloat16()
    with torch.no_grad():
        y0 = m(x).clone()
        fp8.enable("safe", model=model)
        try:
            _zero()
            y1 = m(x).clone()
            n = b200_fp8.STATS["pv_fused"]
        finally:
            fp8.disable()
    assert (n == 1) == fires, (model, n)
    if not fires:
        assert torch.equal(y0, y1)
    assert b200_fp8.FP8_QK is False  # the e4m3 q / k of the H100 route is not built on the B200


def test_gate() -> None:
    _needs_b200()
    if b200_fp8.MIN_S <= 128:
        pytest.skip("no S gate")
    m = _module(TriangleAttention, 128, 4, {}, True)
    x = torch.randn(1, 128, 128, 128, device="cuda").bfloat16()
    with torch.no_grad():
        _on()
        _zero()
        m(x)
    assert b200_fp8.STATS["pv_fused"] == 0


def test_route_stats_keys() -> None:
    _needs_b200()
    st = fp8.route_stats()["triattn"]
    for k in ("pv_fused", "pv_front", "calibrations"):
        assert k in st


def test_shipped_gates() -> None:
    """The gates as shipped (measured, docs/b200/fp8.md 8.3): S >= 1024 for the head-dim-32 kinds; the 64-channel-head core is built but its gate is closed (module x1.04-1.12)."""
    import os

    if "MINIWORLD_TRIATTN_B200_FP8_MINS" not in os.environ:
        assert _SHIPPED_GATES[0] == 1024
    if "MINIWORLD_TRIATTN_B200_FP8_MINS64" not in os.environ:
        assert _SHIPPED_GATES[1] >= 1 << 30


def test_d64_core_when_opened() -> None:
    """The 64-channel-head e4m3 core (kind ``d64``, Chai-1 trunk 256 x 4 x 64) with its gate opened: fires once, accuracy inside the judge tolerance."""
    _needs_b200()
    m = _module(TriangleAttentionChai1, 256, 4, {}, True)
    ref = copy.deepcopy(m).set_backend("pytorch").float()
    x = torch.randn(1, 512, 512, 256, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3)).bfloat16()
    with torch.no_grad():
        yr = ref(x.float())
        _on()
        hs.FP8_PV_D64 = True
        _zero()
        y1 = m(x).clone()
    assert b200_fp8.STATS["pv_d64"] == 1, b200_fp8.STATS
    assert torch.isfinite(y1).all() and _rel(y1, yr) <= TOL, _rel(y1, yr)

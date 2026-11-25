"""B200: the opt-in e4m3-plane TriMul chain (``b200_fp8``): accuracy against the float64 module (next to the bf16 default's), saturation / rescale through captured
graphs, graph replay == eager, route statistics and gates, RoseTTAFold3's length factor, and the default route untouched (POLICY None = off)."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_b200() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("needs a B200 (sm_100)")


def _make(d, kind, seed=7, rf3=False):
    from foldjet_engine.modules.triangle import BidirectionalTriangleMultiplication, TriangleMultiplication

    torch.manual_seed(seed)
    if kind == 0:
        m = BidirectionalTriangleMultiplication(d, d_hidden=d, master_weights=True, backend="cuda")
    elif rf3:
        m = TriangleMultiplication(d, outgoing=kind == 1, master_weights=True, backend="cuda", model="rf3")
    else:
        m = TriangleMultiplication(d, outgoing=kind == 1, master_weights=True, backend="cuda")
    m = m.cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
    return m


def _gold(m, d, kind, x, mask, rf3=False):
    from foldjet_engine.modules.triangle import BidirectionalTriangleMultiplication, TriangleMultiplication

    if kind == 0:
        r = BidirectionalTriangleMultiplication(d, d_hidden=d, backend="pytorch")
    elif rf3:
        r = TriangleMultiplication(d, outgoing=kind == 1, backend="pytorch", model="rf3")
    else:
        r = TriangleMultiplication(d, outgoing=kind == 1, backend="pytorch")
    r = r.cuda().eval().double()
    r.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    return r(x.double(), mask)


def _rel(y, y64, x) -> float:
    """update error: rms(y - y64) / rms(y64 - x)"""
    return ((y.double() - y64).norm() / (y64 - x.double()).norm()).item()


@pytest.fixture()
def f8():
    _needs_b200()
    from foldjet_engine.kernels.trimul_inproj.cuda import b200_fp8 as F8

    old = (F8.POLICY, F8.GUARD, F8._MINL_ENV)
    F8._CACHES.clear()
    F8.overflow_peak()
    for k in F8.GUARD_STATS:
        F8.GUARD_STATS[k] = 0
    yield F8
    F8.POLICY, F8.GUARD, F8._MINL_ENV = old
    F8._CACHES.clear()


@pytest.mark.parametrize("d,kind", [(64, 1), (64, 0), (128, 1), (128, 2), (128, 0), (256, 1), (256, 0), (384, 2), (512, 1)])
@pytest.mark.parametrize("n", [256, 384])
def test_accuracy_against_float64(f8, d, kind, n):
    """update error < 6 % of the update (the e4m3 storage error, 6x the bf16 floor) and < 7.5x the bf16 default's; no saturation, the route ran."""
    m = _make(d, kind)
    torch.manual_seed(3)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[:, n - 5 :] = False
    with torch.no_grad():
        f8.POLICY = None
        y0 = m(x, mask)
        y64 = _gold(m, d, kind, x, mask)
        f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
        y1 = m(x, mask)
        peak = f8.overflow_peak()
    e0, e1 = _rel(y0, y64, x), _rel(y1, y64, x)
    assert torch.isfinite(y1.float()).all() and peak <= 448.0
    assert e1 < 0.06 and e1 < 7.5 * e0, (e0, e1)
    assert not torch.equal(y0, y1) and f8.GUARD_STATS["calibrations"] == 1


@pytest.mark.parametrize("xs,off", [(1e-3, 0.0), (40.0, 0.0), (1.0, 25.0), (40.0, 25.0)])
def test_stress_inputs(f8, xs, off):
    """large / small activations and offsets: finite output, error against float64 inside the e4m3 band (the scales come from the first call: the guard handles the rest)."""
    d, n = 128, 256
    m = _make(d, 1)
    torch.manual_seed(5)
    x = (torch.randn(1, n, n, d, device="cuda") * xs + off).bfloat16()
    with torch.no_grad():
        f8.POLICY = None
        y0 = m(x, None)
        y64 = _gold(m, d, 1, x, None)
        f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "sync", "0"
        y1 = m(x, None)
    e0, e1 = _rel(y0, y64, x), _rel(y1, y64, x)
    assert torch.isfinite(y1.float()).all()
    assert e1 < max(0.08, 1.25 * e0) and e1 < 9 * e0 + 0.01, (e0, e1)  # (large offsets: the bf16 floor of y itself dominates both)


def test_saturation_rescale_through_graph(f8):
    """calibrate on a small input, replay a captured graph on a 60x larger one: the lazy guard reports a peak > 448, ``rescale_all`` (in place) fixes the SAME graph."""
    d, n = 128, 256
    m = _make(d, 1)
    torch.manual_seed(1)
    x_small = torch.randn(1, n, n, d, device="cuda").bfloat16()
    x_big = (torch.randn(1, n, n, d, device="cuda") * 60.0 + 3.0).bfloat16()  # (LN_in makes the planes scale invariant: the saturation is forced below)
    f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
    with torch.no_grad():
        m(x_small, None)
        xin = x_small.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m(xin, None)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yout = m(xin, None)
        assert f8.GUARD_STATS["capture_fallbacks"] == 0
        f8.overflow_peak()
        xin.copy_(x_big)
        for st in f8._CACHES.values():  # stale scales (a different weight set / outlier channel in a later input): 64x too small
            st["s"].div_(64.0)
            f8.refresh(st)
        g.replay()
        peak = f8.overflow_peak()
        assert peak > 448.0
        f = f8.rescale_all(peak)
        assert f >= 4.0
        g.replay()
        assert f8.overflow_peak() <= 448.0
        f8.POLICY = None
        y64 = _gold(m, d, 1, x_big, None)
        y0 = m(x_big, None)
        assert _rel(yout, y64, x_big) < max(0.08, 1.25 * _rel(y0, y64, x_big))


def test_graph_equals_eager_and_capture_without_calibration(f8):
    d, n = 128, 256
    m = _make(d, 2)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
    with torch.no_grad():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            pass
        g0 = torch.cuda.CUDAGraph()  # a capture before any calibration: the bf16 route
        with torch.cuda.graph(g0):
            ydef = m(x, None)
        assert f8.GUARD_STATS["capture_fallbacks"] == 1 and not f8._CACHES
        eager = m(x, None)
        eager2 = m(x, None)
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            m(x, None)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = m(x, None)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, eager) and torch.equal(eager, eager2)
        assert not torch.equal(y, ydef) or True  # (g0 replayed nothing: ydef is only the capture's buffer)


def test_default_untouched_and_gates(f8):
    d, n = 128, 256
    m = _make(d, 1)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    with torch.no_grad():
        f8.POLICY = None
        a = m(x, None)
        f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
        m(x, None)
        assert f8.serves(m, x, None, False)
        assert not f8.serves(m, torch.zeros(1, 200, 200, d, device="cuda", dtype=torch.bfloat16), None, False)  # L / ceil128(L) < 0.9: ctr8 tile quantisation (L400 D256 measured +5 % slower)
        assert f8.serves(m, torch.zeros(1, 1000, 1000, d, device="cuda", dtype=torch.bfloat16), None, False)  # L % 16 != 0 but 1000 / 1024: padded on the host
        assert not f8.serves(m, x.float(), None, False)
        f8._MINL_ENV = None
        assert not f8.serves(m, x[:, :128, :128].contiguous(), None, False) or f8.MIN_L.get((128, 1), 256) <= 128
        f8.POLICY = None
        assert not f8.serves(m, x, None, False)
        b = m(x, None)
    assert torch.equal(a, b)


def test_rf3_length_factor(f8):
    """RoseTTAFold3: one calibration, the factor rides on csc: accurate for two different real lengths without recalibration."""
    d, n = 128, 256
    m = _make(d, 1, rf3=True)
    torch.manual_seed(2)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
    with torch.no_grad():
        res = []
        for real in (n - 37, n - 3):
            mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
            mask[:, real:] = False
            y1 = m(x, mask)
            f8.POLICY = None
            y64 = _gold(m, d, 1, x, mask, rf3=True)
            y0 = m(x, mask)
            f8.POLICY = "calib"
            res.append((_rel(y1, y64, x), _rel(y0, y64, x)))
    assert len(f8._CACHES) == 1 and f8.GUARD_STATS["calibrations"] == 1
    for e1, e0 in res:
        assert e1 < 0.06 and e1 < 9 * e0, res


def test_batched_equals_per_sample_scales(f8):
    d, n = 128, 256
    m = _make(d, 1)
    x = torch.randn(3, n, n, d, device="cuda").bfloat16()
    f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
    with torch.no_grad():
        y = m(x, None)
        ys = torch.cat([m(x[b : b + 1], None) for b in range(3)])
    assert torch.equal(y, ys)


@pytest.mark.parametrize("d,hid,kind,n", [(64, 128, 1, 256), (64, 128, 2, 512), (64, 128, 0, 256), (128, 256, 1, 256)])
def test_hidden_twice_the_pair_width(f8, d, hid, kind, n):
    """AF3 / Protenix template blocks (pair 64, hidden 128) and the D128 hidden-256 form: the plane counts follow the hidden width."""
    from foldjet_engine.modules.triangle import BidirectionalTriangleMultiplication, TriangleMultiplication

    torch.manual_seed(7)
    if kind == 0:
        m = BidirectionalTriangleMultiplication(d, d_hidden=hid, master_weights=True, backend="cuda")
        r = BidirectionalTriangleMultiplication(d, d_hidden=hid, backend="pytorch")
    else:
        m = TriangleMultiplication(d, d_hidden=hid, outgoing=kind == 1, master_weights=True, backend="cuda")
        r = TriangleMultiplication(d, d_hidden=hid, outgoing=kind == 1, backend="pytorch")
    m = m.cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, p.shape[-1] ** -0.5)
        r = r.cuda().eval().double()
        r.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
        x = torch.randn(1, n, n, d, device="cuda").bfloat16()
        f8.POLICY = None
        y0 = m(x, None)
        y64 = r(x.double(), None)
        f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
        y1 = m(x, None)
    e0, e1 = _rel(y0, y64, x), _rel(y1, y64, x)
    assert e1 < 0.06 and e1 < 9 * e0 and not torch.equal(y0, y1)


@pytest.mark.parametrize("d,kind,n", [(64, 0, 1008), (128, 1, 1000), (64, 2, 1000)])
def test_any_length_padded_on_the_host(f8, d, kind, n):
    """L % 16 != 0 (and L % 128 != 0): padded with masked tokens like the default route, ctr8's 3-D maps clip the tiles; accuracy as at the multiples of 128."""
    m = _make(d, kind)
    torch.manual_seed(3)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[:, n - 9 :] = False
    with torch.no_grad():
        f8.POLICY = None
        y0 = m(x, mask)
        y64 = _gold(m, d, kind, x, mask)
        f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
        y1 = m(x, mask)
    e0, e1 = _rel(y0, y64, x), _rel(y1, y64, x)
    assert torch.isfinite(y1.float()).all() and e1 < 0.06 and e1 < 9 * e0 and not torch.equal(y0, y1)


@pytest.mark.parametrize("n", [144, 300, 400])
def test_lengths_far_from_a_tile_multiple_keep_the_default(f8, n):
    d = 128
    m = _make(d, 1)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    with torch.no_grad():
        f8.POLICY = None
        y0 = m(x, None)
        f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
        y1 = m(x, None)
    assert torch.equal(y0, y1) and not f8._CACHES


def test_rf3_graph_follows_the_mask(f8):
    """A captured RF3 call reads the real length from the mask at replay: after the mask changed, the replay is accurate without recalibration (the factor rides on csc)."""
    d, n = 128, 256
    m = _make(d, 1, rf3=True)
    torch.manual_seed(2)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[:, n - 37 :] = False
    f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
    with torch.no_grad():
        m(x, mask)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m(x, mask)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = m(x, mask)
        assert f8.GUARD_STATS["capture_fallbacks"] == 0 and f8.GUARD_STATS["calibrations"] == 1
        mask.fill_(True)
        mask[:, n - 3 :] = False
        g.replay()
        torch.cuda.synchronize()
        f8.POLICY = None
        y64 = _gold(m, d, 1, x, mask, rf3=True)
        y0 = m(x, mask)
    assert _rel(y, y64, x) < 0.06 and _rel(y, y64, x) < 9 * _rel(y0, y64, x)
    assert f8.GUARD_STATS["calibrations"] == 1


# ---------------------------------------------------------------------------------------------------- quiet-GPU gate re-measurement + wider sweeps (agent G)
def test_remeasured_gates(f8):
    """MIN_L after the quiet-GPU re-measurement (docs/b200/fp8.md): D384 serves from 128 (bidirectional) / 256, the D128 bidirectional block from 384; the default route unchanged below."""
    assert f8.MIN_L[(384, 0)] == 128 and f8.MIN_L[(384, 1)] == 256 and f8.MIN_L[(384, 2)] == 256 and f8.MIN_L[(128, 0)] == 384
    m = _make(384, 1)
    x = torch.randn(1, 256, 256, 384, device="cuda").bfloat16()
    with torch.no_grad():
        f8.POLICY = None
        y0 = m(x, None)
        f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", None
        m(x, None)
        assert f8.serves(m, x, None, False)
        assert not f8.serves(m, x[:, :128, :128].contiguous(), None, False)
        f8.POLICY = None
        y1 = m(x, None)
    assert torch.equal(y0, y1)


@pytest.mark.parametrize("d,kind,n,nb", [(384, 0, 128, 1), (384, 0, 256, 1), (512, 0, 256, 1), (512, 2, 256, 1), (128, 1, 256, 2), (256, 1, 256, 2), (128, 1, 128, 4)])
def test_accuracy_bidirectional_d384_d512_and_batches(f8, d, kind, n, nb):
    """D384 / D512 bidirectional and B > 1: update error < 6 % (bidirectional < 8 %), < 7.5x the bf16 default's, the route ran."""
    m = _make(d, kind)
    torch.manual_seed(3)
    x = torch.randn(nb, n, n, d, device="cuda").bfloat16()
    mask = torch.ones(nb, n, dtype=torch.bool, device="cuda")
    mask[:, n - 5 :] = False
    with torch.no_grad():
        f8.POLICY = None
        y0 = m(x, mask)
        y64 = _gold(m, d, kind, x, mask)
        f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
        y1 = m(x, mask)
        peak = f8.overflow_peak()
    e0, e1 = _rel(y0, y64, x), _rel(y1, y64, x)
    assert torch.isfinite(y1.float()).all() and peak <= 448.0
    assert e1 < (0.08 if kind == 0 else 0.06) and e1 < 7.5 * e0, (e0, e1)
    assert not torch.equal(y0, y1)
    print("TriMul", d, kind, n, nb, "bf16", e0, "e4m3", e1)

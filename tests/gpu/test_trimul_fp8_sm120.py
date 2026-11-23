"""RTX 5090: the opt-in e4m3-plane TriMul chain (``sm120_fp8``: k1w8 front, cuBLASLt batched e4m3 contraction): accuracy against the float64 module (next to the bf16
default's), saturation / rescale through a captured graph, graph replay == eager, route statistics and gates, and the default route untouched (POLICY None = off)."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_5090() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("needs an RTX 5090 (sm_120)")


def _make(d, kind, seed=7):
    from miniworld_engine.modules.triangle import BidirectionalTriangleMultiplication, TriangleMultiplication

    torch.manual_seed(seed)
    if kind == 0:
        m = BidirectionalTriangleMultiplication(d, d_hidden=d, master_weights=True, backend="cuda")
    else:
        m = TriangleMultiplication(d, outgoing=kind == 1, master_weights=True, backend="cuda")
    m = m.cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
    return m


def _gold(m, d, kind, x, mask):
    from miniworld_engine.modules.triangle import BidirectionalTriangleMultiplication, TriangleMultiplication

    if kind == 0:
        r = BidirectionalTriangleMultiplication(d, d_hidden=d, backend="pytorch")
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
    _needs_5090()
    from miniworld_engine.kernels.trimul_inproj.cuda import sm120_fp8 as F8

    old = (F8.POLICY, F8.GUARD, F8._MINL_ENV)
    F8._CACHES.clear()
    F8.overflow_peak()
    for k in F8.GUARD_STATS:
        F8.GUARD_STATS[k] = 0
    yield F8
    F8.POLICY, F8.GUARD, F8._MINL_ENV = old
    F8._CACHES.clear()


@pytest.mark.parametrize("d,kind", [(128, 1), (128, 2), (128, 0), (256, 1), (256, 2), (256, 0), (384, 1)])
@pytest.mark.parametrize("n", [256, 384])
def test_accuracy_against_float64(f8, d, kind, n):
    """update error < 6 % of the update (the e4m3 storage error, ~6x the bf16 floor) and < 9x the bf16 default's; no saturation, the route ran."""
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
    print("TriMul", d, kind, n, "bf16", e0, "e4m3", e1)
    assert torch.isfinite(y1.float()).all() and peak <= 448.0
    assert e1 < 0.06 and e1 < 9 * e0, (e0, e1)
    assert not torch.equal(y0, y1) and f8.GUARD_STATS["calibrations"] == 1


@pytest.mark.parametrize("xs,off", [(1e-3, 0.0), (40.0, 0.0), (1.0, 25.0), (40.0, 25.0)])
def test_stress_inputs(f8, xs, off):
    """large / small activations and offsets: finite output, error against float64 inside the e4m3 band."""
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
    assert e1 < max(0.08, 1.25 * e0) and e1 < 9 * e0 + 0.01, (e0, e1)


def test_saturation_rescale_through_graph(f8):
    """calibrate, then replay a captured graph with scales 64x too small: the lazy guard reports a peak > 448, ``rescale_all`` (in place) fixes the SAME graph."""
    d, n = 128, 256
    m = _make(d, 1)
    torch.manual_seed(1)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
    with torch.no_grad():
        m(x, None)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m(x, None)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yout = m(x, None)
        assert f8.GUARD_STATS["capture_fallbacks"] == 0
        f8.overflow_peak()
        for st in f8._CACHES.values():  # stale scales: 64x too small
            st["s"].div_(64.0)
            f8._derive(st)
        g.replay()
        peak = f8.overflow_peak()
        assert peak > 448.0
        f = f8.rescale_all(peak)
        assert f >= 4.0
        g.replay()
        assert f8.overflow_peak() <= 448.0
        f8.POLICY = None
        y64 = _gold(m, d, 1, x, None)
        y0 = m(x, None)
        assert _rel(yout, y64, x) < max(0.08, 9 * _rel(y0, y64, x))


def test_sync_guard_redoes_a_saturating_call(f8):
    """eager ``sync`` guard: scales 64x too small after calibration -> the call detects the overflow, raises the scales and redoes itself (retries >= 1), output accurate."""
    d, n = 128, 256
    m = _make(d, 2)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "sync", "0"
    with torch.no_grad():
        m(x, None)
        for st in f8._CACHES.values():
            st["s"].div_(64.0)
            f8._derive(st)
        y1 = m(x, None)
        f8.POLICY = None
        y64 = _gold(m, d, 2, x, None)
    assert f8.GUARD_STATS["retries"] >= 1 and f8.GUARD_STATS["fallbacks"] == 0 and _rel(y1, y64, x) < 0.06


def test_graph_equals_eager_and_capture_without_calibration(f8):
    d, n = 128, 256
    m = _make(d, 2)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
    with torch.no_grad():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g0 = torch.cuda.CUDAGraph()  # a capture before any calibration: the default route
        with torch.cuda.graph(g0):
            m(x, None)
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


def test_default_untouched_and_gates(f8):
    d, n = 128, 256
    m = _make(d, 1)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    with torch.no_grad():
        f8.POLICY = None
        a = m(x, None)
        f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
        m(x, None)
        assert f8.serves(x, d, n, False)
        assert not f8.serves(x.float(), d, n, False)
        assert not f8.serves(x, d, n, True)  # grouped plane formation: default
        f8._MINL_ENV = None
        assert not f8.serves(x[:, :128, :128].contiguous(), d, 128, False)
        f8.POLICY = None
        assert not f8.serves(x, d, n, False)
        b = m(x, None)
    assert torch.equal(a, b)


@pytest.mark.parametrize("n", [144, 400])
def test_lengths_not_a_multiple_of_32_keep_the_default(f8, n):
    d = 128
    m = _make(d, 2)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    with torch.no_grad():
        f8.POLICY = None
        y0 = m(x, None)
        f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
        y1 = m(x, None)
    assert torch.equal(y0, y1) and not f8._CACHES


def test_batched_equals_per_sample(f8):
    d, n = 128, 256
    m = _make(d, 1)
    x = torch.randn(3, n, n, d, device="cuda").bfloat16()
    f8.POLICY, f8.GUARD, f8._MINL_ENV = "calib", "lazy", "0"
    with torch.no_grad():
        y = m(x, None)
        ys = torch.cat([m(x[b : b + 1], None) for b in range(3)])
    assert torch.equal(y, ys)

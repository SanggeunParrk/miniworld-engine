"""B200 (sm_100a) FP8 Transition route (``kernels/transition/cuda/fp8_sm100a.py``, docs/b200/fp8.md): route on / off, error against fp64 next to the default
route, saturation / rescale (sync and lazy guard, ``fp8.check()``), CUDA-graph replay = eager bytes, route_stats counters, default route untouched."""

from __future__ import annotations

import hashlib
import math

import pytest
import torch

from foldjet_engine import fp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="needs a B200 (sm_100a)"
)

EPS = 1e-5


def _mods():
    from foldjet_engine.kernels.transition.cuda import fp8_sm100a as f8
    from foldjet_engine.kernels.transition.cuda import route_sm100a as r

    return f8, r


@pytest.fixture()
def route():
    f8, r = _mods()
    old = (f8.POLICY, f8.MIN_ROWS, f8.GUARD, f8.WIDTHS, f8.FUSED_WIDTHS, f8.FUSED_MIN_ROWS)
    f8.POLICY, f8.MIN_ROWS, f8.GUARD, f8.WIDTHS, f8.FUSED_WIDTHS, f8.FUSED_MIN_ROWS = "calib", 0, "sync", (128, 256, 384, 512, 768), (128, 256), {}
    f8._CACHES.clear()
    f8.overflow_peak()
    for k in f8.GUARD_STATS:
        f8.GUARD_STATS[k] = 0
    yield f8, r
    f8.POLICY, f8.MIN_ROWS, f8.GUARD, f8.WIDTHS, f8.FUSED_WIDTHS, f8.FUSED_MIN_ROWS = old
    f8._CACHES.clear()


def _weights(d, n, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    h = n * d
    wa = torch.randn(h, d, device=device, generator=g) / math.sqrt(d)
    wb = torch.randn(h, d, device=device, generator=g) / math.sqrt(d)
    ws = torch.randn(d, h, device=device, generator=g) / math.sqrt(h)
    gm = 1 + 0.1 * torch.randn(d, device=device, generator=g)
    bt = 0.1 * torch.randn(d, device=device, generator=g)
    return gm, bt, wa, wb, ws


def _gold(x, gm, bt, wa, wb, ws):
    xd = x.double()
    mu = xd.mean(-1, keepdim=True)
    var = ((xd - mu) ** 2).mean(-1, keepdim=True)
    xn = (xd - mu) / torch.sqrt(var + EPS) * gm.double() + bt.double()
    a, b = xn @ wa.double().T, xn @ wb.double().T
    return xd + (a * torch.sigmoid(a) * b) @ ws.double().T


def _judge(y, y64):
    return ((y.double() - y64).abs().max() / y64.abs().max()).item()


def _x(rows, d, scale=1.0, off=0.0, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, d, device="cuda", generator=g) * scale + off).bfloat16()


def _sha(t):
    return hashlib.sha1(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def test_off_by_default_is_the_default_route():
    f8, r = _mods()
    d, n = 384, 4
    gm, bt, wa, wb, ws = _weights(d, n)
    x = _x(65536, d)
    old = f8.POLICY
    f8.POLICY = None
    try:
        assert not f8.serves(x, wa, ws)
        y_off = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
        y_ref = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS, _fp8=False)
        assert _sha(y_off) == _sha(y_ref)
    finally:
        f8.POLICY = old


@pytest.mark.parametrize("d,n", [(384, 4), (384, 2), (512, 4), (512, 2), (768, 2), (256, 4), (128, 4)])
def test_error_vs_fp64(route, d, n):
    f8, r = route
    gm, bt, wa, wb, ws = _weights(d, n)
    x = _x(16384, d)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    assert f8.serves(x, wa, ws)
    r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)  # calibration call
    y8 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    yd = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS, _fp8=False)
    assert torch.isfinite(y8.float()).all()
    j8, jd = _judge(y8, y64), _judge(yd, y64)
    assert j8 < 0.05, (j8, jd)  # H100 acceptance band: 0.035-0.045 (full e4m3)
    assert j8 / jd < 16, (j8, jd)
    st = f8.GUARD_STATS
    assert st["calibrations"] == 1 and st["calls"] >= 2 and st["fallbacks"] == 0


@pytest.mark.parametrize("scale,off", [(1e-3, 0.0), (1e-3, 25.0), (40.0, 0.0), (40.0, 25.0), (1.0, 25.0)])
def test_large_activation_stress(route, scale, off):
    f8, r = route
    d, n = 384, 4
    gm, bt, wa, wb, ws = _weights(d, n)
    x = _x(16384, d, scale, off)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    y8 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    y8 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    yd = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS, _fp8=False)
    assert torch.isfinite(y8.float()).all()
    assert _judge(y8, y64) < 0.07 and _judge(y8, y64) / _judge(yd, y64) < 16  # (x scale 1e-3: eps dominates the LayerNorm, the update is ~6.5 % of y - x as on the H100)


def test_sync_guard_redoes_a_saturated_call(route):
    f8, r = route
    d, n = 384, 4
    gm, bt, wa, wb, ws = _weights(d, n)
    small = _x(16384, d, 1e-4)  # eps dominates the LayerNorm here: h is ~1000x smaller than at scale 1
    r.transition_sm100a(small, gm, bt, wa, wb, ws, EPS)  # calibrates on the small input
    x = _x(16384, d)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    y8 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    assert f8.GUARD_STATS["retries"] >= 1 and f8.GUARD_STATS["fallbacks"] == 0
    assert _judge(y8, y64) < 0.06


def test_lazy_guard_check_and_rerun_through_a_graph(route):
    f8, r = route
    f8.GUARD = "lazy"
    d, n = 384, 4
    gm, bt, wa, wb, ws = _weights(d, n)
    small = _x(16384, d, 1e-4)
    r.transition_sm100a(small, gm, bt, wa, wb, ws, EPS)  # eager calibration on the small input
    x = _x(16384, d)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    out = torch.empty_like(x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    torch.cuda.synchronize()
    f8.overflow_peak()  # (the warm-up call saturated too: forget it)
    f8.refresh(next(iter(f8._CACHES.values())))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS))
    graph.replay()
    torch.cuda.synchronize()
    assert fp8.check() is True  # saturation seen through the captured graph; the scale was lowered in place
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        if not fp8.check():
            break
    else:
        pytest.fail("still saturating after 3 reruns")
    assert _judge(out, y64) < 0.06


def test_graph_replay_equals_eager_bytes(route):
    f8, r = route
    f8.GUARD = "lazy"
    d, n = 512, 4
    gm, bt, wa, wb, ws = _weights(d, n)
    x = _x(16384, d)
    r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)  # calibrate
    eager = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS).clone()
    out = torch.empty_like(x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS))
    g.replay()
    torch.cuda.synchronize()
    assert _sha(out) == _sha(eager)
    assert f8.GUARD_STATS["capture_fallbacks"] == 0


def test_capture_without_calibration_takes_the_default_route(route):
    f8, r = route
    d, n = 384, 4
    gm, bt, wa, wb, ws = _weights(d, n)
    x = _x(16384, d)
    out = torch.empty_like(x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS, _fp8=False)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS))
    g.replay()
    torch.cuda.synchronize()
    assert f8.GUARD_STATS["capture_fallbacks"] == 1
    assert _sha(out) == _sha(r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS, _fp8=False))


def test_gates_and_scope(route):
    f8, r = route
    d, n = 512, 4
    gm, bt, wa, wb, ws = _weights(d, n)
    f8.MIN_ROWS = None
    f8.WIDTHS = (512, 768)
    assert not f8.serves(_x(512, d), wa, ws) and f8.serves(_x(f8.ROW_GATE[(d, n)], d), wa, ws)
    gm3, bt3, wa3, wb3, ws3 = _weights(384, 4)
    assert not f8.serves(_x(65536, 384), wa3, ws3)  # D 384: no gain, stays on the default route
    f8.MIN_ROWS = 0
    assert not f8.serves(_x(1024, d).float(), wa, ws)  # fp32 activations stay on the default route
    gm2, bt2, wa2, wb2, ws2 = _weights(64, 4)
    assert not f8.serves(_x(4096, 64), wa2, ws2)  # D not a multiple of 128


def test_route_stats_and_framework(route):
    f8, r = route
    assert fp8.serves_sm100("transition") and fp8.arch() == "sm100"
    assert fp8._modules()[4] is f8
    fp8.reset_route_stats()
    d, n = 384, 4
    gm, bt, wa, wb, ws = _weights(d, n)
    x = _x(16384, d)
    r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    st = fp8.route_stats()["transition"]
    assert st["calibrations"] == 1 and st["calls"] >= 2 and st["fallbacks"] == 0
    assert fp8.status()["transition"] == "calib"


@pytest.mark.parametrize("rows", [16384 + 37, 4096 + 1, 1000])
def test_ragged_rows(route, rows):
    f8, r = route
    d, n = 512, 4
    gm, bt, wa, wb, ws = _weights(d, n)
    x = _x(rows, d)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    y8 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    assert y8.shape == x.shape and torch.isfinite(y8.float()).all()
    assert _judge(y8, y64) < 0.05


def test_module_engine_fp8_backend_serves_the_route(route):
    """``Transition(backend="engine-fp8")`` (the framework scope) reaches the B200 route; the cuda backend after it is the default route's bytes; training stays default."""
    from foldjet_engine.modules import Transition

    f8, _r = route
    f8.MIN_ROWS, f8.POLICY = None, None  # the process default: every route off; only the engine-fp8 scope turns it on
    d, n, rows = 512, 4, 8192
    torch.manual_seed(0)
    m = Transition(d, n=n, backend="cuda").cuda().float().eval()
    for nm, p in m.named_parameters():  # (the default init zeroes the squeeze: random weights so the update is non-trivial)
        with torch.no_grad():
            if p.dim() > 1:
                p.normal_(0, d ** -0.5 if p.shape[1] == d else (n * d) ** -0.5)
            elif nm.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.1 * torch.randn_like(p))
    m8 = Transition(d, n=n, backend="engine-fp8").cuda().float().eval()
    m8.load_state_dict(m.state_dict())
    x = _x(rows, d).view(1, rows, d)

    def run(mod):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return mod(x).clone()

    before = run(m)
    fp8.reset_route_stats()
    y8 = run(m8)
    assert fp8.route_stats()["transition"]["calls"] == 1 and fp8.route_stats()["transition"]["calibrations"] == 1
    assert not torch.equal(y8, before) and f8.POLICY is None
    assert (y8.float() - before.float()).abs().max() / before.float().abs().max() < 0.2
    assert torch.equal(run(m), before)  # the cuda backend outside the scope: default bytes (scope restored)
    fp8.reset_route_stats()
    m8.train()
    xt = x.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m8(xt).float().sum().backward()
    assert fp8.route_stats()["transition"]["calls"] == 0


def test_nonfinite_input_does_not_break_the_guard(route):
    f8, r = route
    d, n = 512, 4
    gm, bt, wa, wb, ws = _weights(d, n)
    x = _x(16384, d)
    r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)  # calibrate
    sh0 = f8._CACHES[wa.data_ptr()]["sh"].clone()
    xb = x.clone()
    xb[7, 3] = float("inf")
    y = r.transition_sm100a(xb, gm, bt, wa, wb, ws, EPS)  # must return (default route after the guard gives up), not raise
    assert y.shape == x.shape
    ok = torch.ones(x.shape[0], dtype=torch.bool, device="cuda")
    ok[7] = False
    assert torch.isfinite(y[ok].float()).all()
    assert torch.equal(f8._CACHES[wa.data_ptr()]["sh"], sh0)  # the scale was not wrecked by the inf
    f8.GUARD = "lazy"
    r.transition_sm100a(xb, gm, bt, wa, wb, ws, EPS)
    fp8.check()  # (peak inf: no rescale, no exception)
    assert torch.equal(f8._CACHES[wa.data_ptr()]["sh"], sh0)


# ---------------------------------------------------------------------------------------------------------------------------------
# the fused one-kernel e4m3 forward (sm100/fp8/tr8_fwd_sm100.cu): D 128 / 256, n 2 / 4


def _both(route, x, gm, bt, wa, wb, ws, fused):
    """(calibrated, then measured) output of the route with the fused kernel on / off (the 3-launch chain)."""
    f8, r = route
    f8.FUSED_WIDTHS = (128, 256) if fused else ()
    f8._CACHES.clear()
    r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    return r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)


@pytest.mark.parametrize("d,n", [(128, 4), (128, 2), (256, 4), (256, 2)])
def test_fused_error_and_chain_agreement(route, d, n):
    f8, r = route
    gm, bt, wa, wb, ws = _weights(d, n)
    x = _x(16384, d)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    yf = _both(route, x, gm, bt, wa, wb, ws, True)
    yc = _both(route, x, gm, bt, wa, wb, ws, False)
    yd = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS, _fp8=False)
    assert torch.isfinite(yf.float()).all()
    jf, jd = _judge(yf, y64), _judge(yd, y64)
    assert jf < 0.05 and jf / jd < 16, (jf, jd)  # H100 acceptance band 0.035-0.045 (full e4m3)
    # same quantisation points as the chain (the packed LayerNorm rounds a few xn values differently: an e4m3 step on a handful of elements)
    assert (yf.float() - yc.float()).abs().max() / yc.float().abs().max() < 0.02 and abs(jf - _judge(yc, y64)) < 0.01
    assert (yf != yc).float().mean() < 1e-3


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("scale,off", [(1e-3, 0.0), (1e-3, 25.0), (40.0, 0.0), (40.0, 25.0), (1.0, 25.0)])
def test_fused_stress(route, d, scale, off):
    f8, r = route
    gm, bt, wa, wb, ws = _weights(d, 4)
    x = _x(8192, d, scale, off)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    y8 = _both(route, x, gm, bt, wa, wb, ws, True)
    yc = _both(route, x, gm, bt, wa, wb, ws, False)
    yd = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS, _fp8=False)
    assert torch.isfinite(y8.float()).all()
    # (x scale 1e-3: eps dominates the LayerNorm, the update is ~6.5 % of y - x: the fused kernel is within 5 % of the chain's error, whatever the band)
    assert _judge(y8, y64) < 0.08 and _judge(y8, y64) <= 1.05 * _judge(yc, y64) + 1e-3 and _judge(y8, y64) / _judge(yd, y64) < 20


@pytest.mark.parametrize("d", [128, 256])
def test_fused_sync_guard_redoes_a_saturated_call(route, d):
    f8, r = route
    gm, bt, wa, wb, ws = _weights(d, 4)
    f8.FUSED_WIDTHS = (128, 256)
    r.transition_sm100a(_x(8192, d, 1e-4), gm, bt, wa, wb, ws, EPS)  # calibrates on the small input
    x = _x(8192, d)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    y8 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    assert f8.GUARD_STATS["retries"] >= 1 and f8.GUARD_STATS["fallbacks"] == 0
    assert _judge(y8, y64) < 0.06


@pytest.mark.parametrize("d", [128, 256])
def test_fused_lazy_guard_and_graph_replay(route, d):
    """Saturation seen through a captured graph (fp8.check() lowers the scale in place, rerun), then graph replay = eager bytes."""
    f8, r = route
    f8.GUARD = "lazy"
    gm, bt, wa, wb, ws = _weights(d, 4)
    r.transition_sm100a(_x(8192, d, 1e-4), gm, bt, wa, wb, ws, EPS)  # eager calibration on the small input
    x = _x(8192, d)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    out = torch.empty_like(x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    torch.cuda.synchronize()
    f8.overflow_peak()
    f8.refresh(next(iter(f8._CACHES.values())))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS))
    graph.replay()
    torch.cuda.synchronize()
    assert fp8.check() is True
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        if not fp8.check():
            break
    else:
        pytest.fail("still saturating after 3 reruns")
    assert _judge(out, y64) < 0.06
    eager = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS).clone()
    graph.replay()
    torch.cuda.synchronize()
    assert _sha(out) == _sha(eager)
    assert f8.GUARD_STATS["capture_fallbacks"] == 0


@pytest.mark.parametrize("d,rows", [(128, 8192 + 37), (128, 1000), (256, 4096 + 1), (256, 300)])
def test_fused_ragged_rows(route, d, rows):
    f8, r = route
    gm, bt, wa, wb, ws = _weights(d, 4)
    x = _x(rows, d)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    y8 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    assert y8.shape == x.shape and torch.isfinite(y8.float()).all()
    assert _judge(y8, y64) < 0.05


def test_fused_off_is_the_default_route(route):
    """FP8 off: the fused build is never reached and the default route's bytes do not change."""
    f8, r = route
    gm, bt, wa, wb, ws = _weights(128, 4)
    x = _x(16384, 128)
    f8.POLICY = None
    assert not f8.serves(x, wa, ws)
    assert _sha(r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)) == _sha(r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS, _fp8=False))


def test_gates_of_the_fused_widths(route):
    """D 128 / 256 gates, the D 256 n 2 upper row limit and the chain below the fused kernel's row threshold (served rows are the measured ones, docs/b200/fp8.md section 8)."""
    f8, r = route
    f8.MIN_ROWS = None
    f8.FUSED_MIN_ROWS = {(256, 4): 12288, (256, 2): 1 << 60}
    f8.WIDTHS = (128, 256, 512, 768)
    g128 = _weights(128, 4)
    assert not f8.serves(_x(2048, 128), g128[2], g128[4]) and f8.serves(_x(4096, 128), g128[2], g128[4])
    g128n2 = _weights(128, 2)
    assert not f8.serves(_x(8192, 128), g128n2[2], g128n2[4]) and f8.serves(_x(16384, 128), g128n2[2], g128n2[4])
    g256, g256n2 = _weights(256, 4), _weights(256, 2)
    assert not f8.serves(_x(512, 256), g256[2], g256[4]) and f8.serves(_x(1024, 256), g256[2], g256[4]) and f8.serves(_x(262144, 256), g256[2], g256[4])
    assert f8.serves(_x(4096, 256), g256n2[2], g256n2[4]) and not f8.serves(_x(8192, 256), g256n2[2], g256n2[4])
    # D 256 n 4 at 4096 rows runs the chain (the fused kernel has a floor there); at 16384 rows the fused kernel: both within the e4m3 band, the chain bytes differ from the fused bytes
    gm, bt, wa, wb, ws = g256
    for rows in (4096, 16384):
        x = _x(rows, 256)
        y64 = _gold(x, gm, bt, wa, wb, ws)
        f8._CACHES.clear()
        r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
        y8 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
        assert _judge(y8, y64) < 0.05
    f8.MIN_ROWS = 0


@pytest.mark.parametrize("d", [128, 256])
def test_fused_nonfinite_input_does_not_break_the_guard(route, d):
    f8, r = route
    gm, bt, wa, wb, ws = _weights(d, 4)
    x = _x(8192, d)
    r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)  # calibrate
    sh0 = f8._CACHES[wa.data_ptr()]["sh"].clone()
    xb = x.clone()
    xb[7, 3] = float("inf")
    y = r.transition_sm100a(xb, gm, bt, wa, wb, ws, EPS)  # must return, not raise
    assert y.shape == x.shape
    ok = torch.ones(x.shape[0], dtype=torch.bool, device="cuda")
    ok[7] = False
    assert torch.isfinite(y[ok].float()).all()
    assert torch.equal(f8._CACHES[wa.data_ptr()]["sh"], sh0)
    f8.GUARD = "lazy"
    r.transition_sm100a(xb, gm, bt, wa, wb, ws, EPS)
    fp8.check()
    assert torch.equal(f8._CACHES[wa.data_ptr()]["sh"], sh0)


@pytest.mark.parametrize("d,n", [(128, 4), (256, 4)])
def test_fused_second_module_and_weight_update(route, d, n):
    """Two modules (different weights) keep separate calibrations; an in-place weight update is repacked (the cache key is data_ptr + _version)."""
    f8, r = route
    gm, bt, wa, wb, ws = _weights(d, n, seed=0)
    gm2, bt2, wa2, wb2, ws2 = _weights(d, n, seed=5)
    x = _x(8192, d)
    y1 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    y1 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    r.transition_sm100a(x, gm2, bt2, wa2, wb2, ws2, EPS)
    assert f8.GUARD_STATS["calibrations"] == 2
    ws.mul_(0.5)  # in-place update: new bytes
    y2 = r.transition_sm100a(x, gm, bt, wa, wb, ws, EPS)
    y64 = _gold(x, gm, bt, wa, wb, ws)
    assert not torch.equal(y1, y2) and _judge(y2, y64) < 0.06

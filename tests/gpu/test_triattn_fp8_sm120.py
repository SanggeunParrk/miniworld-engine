"""RTX 5090: the opt-in e4m3 TriAttn core (``sm120_fp8``: e4m3 V (and, on the fused d_pair-128 kind, e4m3 q / k) in the register-resident core, V / K quantisation pass,
static per-head scales): accuracy against the float64 module next to the bf16 default's, route statistics and gates, saturation / rescale (eager redo, captured graph), graph
replay == eager, masks, and the default route untouched with the switch off (bit for bit)."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_5090() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("needs an RTX 5090 (sm_120)")


def _make(d, h, seed=7):
    from miniworld_engine.modules import TriangleAttention

    torch.manual_seed(seed)
    m = TriangleAttention(d, n_head=h, master_weights=True, backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.1)
    return m


def _run(m, x, mask):
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        return m(x, mask)


def _rel(y, y64, x) -> float:
    return ((y.double() - y64).norm() / (y64 - x.double()).norm()).item()


@pytest.fixture()
def f8():
    _needs_5090()
    from miniworld_engine import fp8
    from miniworld_engine.kernels.triangle_attention.cuda import sm120_fp8 as F8

    old = (F8.MIN_L, F8.GUARD, F8.QK, F8.PV_ONLY, F8.D64_MIN_L, F8.D64_QK)
    F8.PV_ONLY = True  # the tests below exercise the PV-only core too; test_pv_only_gate checks the shipped setting
    F8._SITES.clear()
    fp8.disable()
    fp8.reset_route_stats()
    yield F8
    fp8.disable()
    F8.MIN_L, F8.GUARD, F8.QK, F8.PV_ONLY, F8.D64_MIN_L, F8.D64_QK = old
    F8._SITES.clear()


def _on(model="af3"):
    from miniworld_engine import fp8

    fp8.enable("safe", model=model)


@pytest.mark.parametrize("d,h", [(128, 4), (256, 8)])
@pytest.mark.parametrize("n", [256, 512])
@pytest.mark.parametrize("masked", [False, True])
def test_accuracy_against_float64(f8, d, h, n, masked):
    """update error < 8 % of the update and < 8x the bf16 default's (e4m3 P / V, plus e4m3 q / k on the d_pair-128 kind); no saturation, the route ran."""
    from miniworld_engine import fp8

    m = _make(d, h)
    torch.manual_seed(3)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    if masked:
        mask[:, n - 5 :] = False
    y0 = _run(m, x, mask)
    m64 = copy.deepcopy(m).double().set_backend("pytorch")
    with torch.no_grad():
        y64 = m64(x.double(), mask).double()
    f8.MIN_L = 0
    _on()
    y1 = _run(m, x, mask)
    y1 = _run(m, x, mask)
    st = fp8.route_stats()["triattn"]
    assert st["pv_front"] >= 2, st
    assert not fp8.check()
    e0, e1 = _rel(y0, y64, x), _rel(y1, y64, x)
    assert e1 < 0.08, (e0, e1)
    assert e1 < 8 * e0, (e0, e1)
    assert (st.get("qk", 0) > 0) == (d == 128), st


def test_pv_only_is_closer(f8):
    """with e4m3 q / k off the error is the P / V one only (H100 route: ~3 %), below the P / V + q / k route's."""
    m = _make(128, 4)
    torch.manual_seed(3)
    n = 256
    x = torch.randn(1, n, n, 128, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    m64 = copy.deepcopy(m).double().set_backend("pytorch")
    with torch.no_grad():
        y64 = m64(x.double(), mask).double()
    f8.MIN_L = 0
    _on()
    f8.QK = False
    f8._SITES.clear()
    e_pv = _rel(_run(m, x, mask), y64, x)
    f8.QK = True
    f8._SITES.clear()
    e_pvqk = _rel(_run(m, x, mask), y64, x)
    assert e_pv < 0.045 and e_pv < e_pvqk, (e_pv, e_pvqk)


def test_default_untouched_when_off(f8):
    """with the switch off (and after the route ran) the default core runs: outputs equal byte for byte."""
    from miniworld_engine import fp8

    m = _make(128, 4)
    x = torch.randn(1, 768, 768, 128, device="cuda").bfloat16()
    mask = torch.ones(1, 768, dtype=torch.bool, device="cuda")
    y0 = _run(m, x, mask)
    f8.MIN_L = 0
    _on()
    _run(m, x, mask)
    _run(m, x, mask)
    fp8.disable()
    fp8.reset_route_stats()
    y2 = _run(m, x, mask)
    assert torch.equal(y0, y2)
    assert fp8.route_stats()["triattn"]["pv_front"] == 0


def test_gates_and_models(f8):
    """below MIN_L, for a model without the call-site kind, for L % 128 != 0, in training: the default core."""
    from miniworld_engine import fp8

    m = _make(128, 4)
    mask = lambda n: torch.ones(1, n, dtype=torch.bool, device="cuda")  # noqa: E731
    _on()
    x = torch.randn(1, 384, 384, 128, device="cuda").bfloat16()
    _run(m, x, mask(384))
    assert fp8.route_stats()["triattn"]["pv_front"] == 0  # 384 < MIN_L (640)
    f8.MIN_L = 0
    _run(m, x, mask(384))
    assert fp8.route_stats()["triattn"]["pv_front"] == 1
    fp8.reset_route_stats()
    _on(model=None)  # no model: TriAttn stays default
    _run(m, x, mask(384))
    assert fp8.route_stats()["triattn"]["pv_front"] == 0
    _on("opendde")  # PV only on the unfused kind: the d_pair-128 / 4-head site is the fused kind
    _run(m, x, mask(384))
    assert fp8.route_stats()["triattn"]["pv_front"] == 0
    _on()
    x2 = torch.randn(1, 200, 200, 128, device="cuda").bfloat16()
    _run(m, x2, mask(200))
    assert fp8.route_stats()["triattn"]["pv_front"] == 0


def test_graph_equals_eager_and_saturation(f8):
    """a captured graph (lazy guard) replays the eager bytes; a V scale made too small saturates: the flag is read by fp8.check(), the scale widened in place, the replay follows;
    the eager guard (sync) redoes a saturating call."""
    from miniworld_engine import fp8

    m = _make(128, 4)
    n = 256
    torch.manual_seed(5)
    x = torch.randn(1, n, n, 128, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    f8.MIN_L = 0
    _on()
    y_e = _run(m, x, mask).clone()  # calibrates
    assert not fp8.check()
    g = torch.cuda.CUDAGraph()
    out = {}
    _run(m, x, mask)
    with torch.cuda.graph(g):
        out["y"] = _run(m, x, mask)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out["y"], y_e)
    assert not fp8.check()
    (site,) = list(f8._SITES.values())
    site.vinv.mul_(2.0**12)  # now |v vinv| >> 448: saturates (lazy, in the graph)
    site.vsc.copy_(1.0 / site.vinv)
    g.replay()
    torch.cuda.synchronize()
    assert fp8.check()  # flagged, V scales widened by 4 in place
    for _ in range(8):
        g.replay()
        if not fp8.check():
            break
    assert not fp8.check()
    assert torch.isfinite(out["y"]).all()
    # eager sync guard: the saturating call is redone with widened scales
    site.vinv.mul_(2.0**12)
    site.vsc.copy_(1.0 / site.vinv)
    st0 = dict(fp8.route_stats()["triattn"])
    y = _run(m, x, mask)
    st1 = fp8.route_stats()["triattn"]
    assert st1["retries"] > st0["retries"] or st1["fallbacks"] > st0["fallbacks"]
    assert torch.isfinite(y).all()


def test_qk_saturation_drops_qk(f8):
    """a q / k scale made too small saturates: the sync guard drops e4m3 q / k (PV-only core) and redoes the call."""
    from miniworld_engine import fp8

    m = _make(128, 4)
    n = 256
    x = torch.randn(1, n, n, 128, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    f8.MIN_L = 0
    _on()
    _run(m, x, mask)
    (site,) = list(f8._SITES.values())
    assert site.qk
    site.ks.mul_(2.0**10)
    site.qs.mul_(2.0**10)
    y = _run(m, x, mask)
    assert not site.qk
    assert torch.isfinite(y).all()
    assert fp8.route_stats()["triattn"]["retries"] >= 1


def test_pv_only_gate(f8):
    """shipped setting (PV_ONLY off): only calls with e4m3 q / k take the e4m3 core; the unfused kind, QK off, and a site whose q / k saturated run the default core."""
    from miniworld_engine import fp8

    f8.PV_ONLY = False
    f8.MIN_L = 0
    n = 256
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    _on()
    m256 = _make(256, 8)
    _run(m256, torch.randn(1, n, n, 256, device="cuda").bfloat16(), mask)
    assert fp8.route_stats()["triattn"]["pv_front"] == 0  # unfused kind: PV only
    m = _make(128, 4)
    x = torch.randn(1, n, n, 128, device="cuda").bfloat16()
    f8.QK = False
    _run(m, x, mask)
    assert fp8.route_stats()["triattn"]["pv_front"] == 0
    f8.QK = True
    y0 = _run(m, x, mask)
    assert fp8.route_stats()["triattn"]["qk"] == 1
    (site,) = [s for s in f8._SITES.values() if s.qk]
    site.ks.mul_(2.0**10)
    site.qs.mul_(2.0**10)
    y = _run(m, x, mask)  # saturates, q / k dropped, the default core redoes the call
    assert not site.qk and torch.isfinite(y).all()
    fp8.disable()
    y_def = _run(m, x, mask)
    assert torch.equal(y, y_def), "after the q / k drop the call is the default core's"
    _on()
    st = dict(fp8.route_stats()["triattn"])
    _run(m, x, mask)
    assert fp8.route_stats()["triattn"]["pv_front"] == st["pv_front"]
    assert torch.isfinite(y0).all()


#: every model of the whitelist plus two deliberately absent ones and no model (2026-11-23: the 5090 route honours fp8.TRIATTN_MODELS exactly)
_MODELS = [None, "boltz2", "intellifold2", "af3", "protenix1", "opendde", "protenix2", "openfold3", "rosettafold3", "chai1"]


def _expect(model, kind: str, pv_only: bool) -> tuple[bool, bool]:
    """(the e4m3 core fires, with e4m3 q / k) for a 32-channel-head site of ``kind`` under ``model``, from the table alone: the kind listed for the model; q / k only on the fused
    kind and only where the table has them; without q / k only with the PV-only switch (the 5090 gate). The d64 kind (64-channel heads) has no 5090 core."""
    from miniworld_engine import fp8

    kinds, qk = fp8.triattn_for(model)
    if kind not in kinds:
        return False, False
    with_qk = qk and kind == "fused"
    return (with_qk or pv_only), with_qk


@pytest.mark.parametrize("pv_only", [False, True], ids=["shipped", "pvonly"])
@pytest.mark.parametrize("model", _MODELS, ids=[str(m) for m in _MODELS])
def test_model_whitelist_table(f8, model, pv_only):
    """fp8.enable("safe", model=m): the 5090 core fires exactly on the call-site kinds TRIATTN_MODELS lists for m (fused = d_pair 128 / 4 x 32, unfused = d_pair 256 / 8 x 32),
    with e4m3 q / k exactly where the table has them; nothing for no model / an unlisted one; a 64-channel-head site (Chai-1 d64) never; disable() = nothing."""
    from miniworld_engine import fp8

    f8.PV_ONLY = pv_only
    f8.MIN_L = 0
    n = 256
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    fp8.enable("safe", model=model)
    for (d, h), kind in (((128, 4), "fused"), ((256, 8), "unfused")):
        f8._SITES.clear()
        fp8.reset_route_stats()
        m = _make(d, h)
        _run(m, torch.randn(1, n, n, d, device="cuda").bfloat16(), mask)
        st = fp8.route_stats()["triattn"]
        fires, with_qk = _expect(model, kind, pv_only)
        assert (st["pv_front"] > 0, st.get("qk", 0) > 0) == (fires, with_qk), (model, kind, st)
    # the d64 kind (64-channel heads, Chai-1): the 32-channel gate never takes them; the 64-channel core exactly where the table lists d64 (chai1), at L 512
    x = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert not f8.serves(x, 128, 2, 64, 1024)
        f8._SITES.clear()
        fp8.reset_route_stats()
        mc, _ = _make_chai()
        xc, mkc = _chai_x(512)
        mc(xc, mkc)
    assert (fp8.route_stats()["triattn"].get("pv_d64", 0) > 0) == ("d64" in fp8.triattn_for(model)[0]), model
    fp8.disable()
    fp8.reset_route_stats()
    _run(_make(128, 4), torch.randn(1, n, n, 128, device="cuda").bfloat16(), mask)
    assert fp8.route_stats()["triattn"]["pv_front"] == 0


def test_env_model_at_import():
    """MINIWORLD_FP8=safe MINIWORLD_FP8_MODEL=<m> (read at import): af3 fires the e4m3 q / k core at L640, boltz2 / no model do not (one fresh process each)."""
    import json
    import os
    import subprocess
    import sys

    _needs_5090()
    code = (
        "import json, torch; from miniworld_engine import fp8; from miniworld_engine.modules import TriangleAttention\n"
        "torch.manual_seed(0); m = TriangleAttention(128, n_head=4, master_weights=True, backend='cuda').cuda().eval()\n"
        "x = torch.randn(1, 640, 640, 128, device='cuda').bfloat16(); mk = torch.ones(1, 640, dtype=torch.bool, device='cuda')\n"
        "with torch.no_grad(), torch.autocast('cuda', torch.bfloat16): m(x, mk)\n"
        "print(json.dumps(fp8.route_stats()['triattn']))"
    )
    base = {k: v for k, v in os.environ.items() if not k.startswith(("MINIWORLD_FP8", "MINIWORLD_TRIATTN"))}

    def stats(**env):
        out = subprocess.run([sys.executable, "-c", code], env={**base, **env}, capture_output=True, text=True, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])

    a = stats(MINIWORLD_FP8="safe", MINIWORLD_FP8_MODEL="af3")
    assert a["pv_front"] == 1 and a.get("qk", 0) == 1, a
    for env in ({"MINIWORLD_FP8": "safe", "MINIWORLD_FP8_MODEL": "boltz2"}, {"MINIWORLD_FP8": "safe"}, {"MINIWORLD_FP8_MODEL": "af3"}):
        st = stats(**env)
        assert st["pv_front"] == 0 and st.get("qk", 0) == 0, (env, st)


def test_late_qk_calibration(f8):
    """a fused site calibrated while q / k was not wanted (QK off) takes its q / k scales on the first call with q / k wanted (no stale PV-only site)."""
    from miniworld_engine import fp8

    f8.MIN_L = 0
    f8.PV_ONLY = True
    m = _make(128, 4)
    n = 256
    x = torch.randn(1, n, n, 128, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    _on()
    f8.QK = False
    _run(m, x, mask)
    (site,) = list(f8._SITES.values())
    assert not site.qk and not site.qk_cal
    f8.QK = True
    f8.PV_ONLY = False
    fp8.reset_route_stats()
    _run(m, x, mask)
    assert site.qk and fp8.route_stats()["triattn"]["qk"] == 1


# ---------------------------------------------------------------------------------------------------------------- 64-channel heads (kind d64)
# Chai-1's trunk / confidence-head triangle attention (d_pair 256, 4 heads x 64 channels) on the RTX 5090 runs sm80_wide.variant_forward_ops; the e4m3 core of 64-channel heads
# (sm120_fp8.serves_d64 / route(hd=64)) replaces its core + gate_rows when fp8.TRIATTN_MODELS turns the d64 kind on (MINIWORLD_FP8_MODEL=chai1).


def _make_chai(model="chai1", starting=True, seed=1):
    from miniworld_engine.modules.triangle import TriangleAttention

    m = TriangleAttention(256, 4, model=model, starting=starting, p_drop=0.0).cuda().eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for nm, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif nm.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    m64 = copy.deepcopy(m).double().set_backend("pytorch")
    m = m.to(torch.bfloat16)
    m.set_backend("cuda")
    return m, m64


def _chai_x(n, seed=5, masked=True):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, n, n, 256, generator=g).cuda().to(torch.bfloat16)
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    if masked:
        mask[:, n - n // 8 :] = False
    return x, mask


def _upd_err(y, y64, x):
    u = y64 - x.double()
    e = y.double() - y64
    return (e.pow(2).mean().sqrt() / u.pow(2).mean().sqrt()).item(), (e.abs().max() / u.abs().max()).item()


@pytest.mark.parametrize("model", ["chai1", "chai1_confidence"])
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("n", [256, 512])
def test_d64_accuracy_against_float64(f8, model, starting, n):
    """update error of the e4m3 P / V core of 64-channel heads (q / k bf16, as the H100 d64 route: rms 3.3-3.7 %, max 3.0-4.3 %): rms < 6 %, max < 8 %, < 8x the bf16 default's;
    the route ran, nothing saturated."""
    from miniworld_engine import fp8

    m, m64 = _make_chai(model, starting)
    x, mask = _chai_x(n)
    with torch.no_grad():
        y0 = m(x, mask)
        y64 = m64(x.double(), mask).double()
    f8.D64_MIN_L = 0
    _on("chai1")
    with torch.no_grad():
        m(x, mask)  # calibrates
        y1 = m(x, mask)
    st = fp8.route_stats()["triattn"]
    assert st.get("pv_d64", 0) >= 2 and st.get("qk", 0) == 0, st
    assert not fp8.check()
    e0, e1 = _upd_err(y0, y64, x), _upd_err(y1, y64, x)
    assert e1[0] < 0.06 and e1[1] < 0.08, (e0, e1)
    assert e1[0] < 8 * e0[0], (e0, e1)


def test_d64_default_untouched_when_off(f8):
    """switch off (and after the route ran): the default core, byte for byte; no pv_d64 call."""
    from miniworld_engine import fp8

    m, _ = _make_chai()
    x, mask = _chai_x(512)
    with torch.no_grad():
        y0 = m(x, mask)
    f8.D64_MIN_L = 0
    _on("chai1")
    with torch.no_grad():
        m(x, mask)
        m(x, mask)
    assert fp8.route_stats()["triattn"].get("pv_d64", 0) == 2
    fp8.disable()
    fp8.reset_route_stats()
    with torch.no_grad():
        y2 = m(x, mask)
    assert torch.equal(y0, y2)
    assert fp8.route_stats()["triattn"].get("pv_d64", 0) == 0


def test_d64_gates_and_models(f8):
    """below D64_MIN_L, a model without the d64 kind (af3, opendde, no model), training (grad): the default core; L % 128 != 0 refused by serves_d64."""
    from miniworld_engine import fp8

    m, _ = _make_chai()
    x, mask = _chai_x(256)
    _on("chai1")
    f8.D64_MIN_L = 512
    with torch.no_grad():
        m(x, mask)
    assert fp8.route_stats()["triattn"].get("pv_d64", 0) == 0  # 256 < D64_MIN_L
    f8.D64_MIN_L = 0
    with torch.no_grad():
        m(x, mask)
    assert fp8.route_stats()["triattn"].get("pv_d64", 0) == 1
    for model in ("af3", "opendde", None):
        fp8.reset_route_stats()
        _on(model)
        with torch.no_grad():
            m(x, mask)
        assert fp8.route_stats()["triattn"].get("pv_d64", 0) == 0, model
    _on("chai1")
    fp8.reset_route_stats()
    xg = x.clone().requires_grad_(True)
    m(xg, mask).float().sum().backward()
    assert fp8.route_stats()["triattn"].get("pv_d64", 0) == 0
    x1 = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert f8.serves_d64(x1, 4, 64, 1024) and not f8.serves_d64(x1, 4, 32, 1024) and not f8.serves_d64(x1, 4, 64, 1000)


def test_d64_graph_equals_eager_and_saturation(f8):
    """a captured graph (lazy guard) replays the eager bytes; a V scale made too small saturates inside the graph: fp8.check() flags it and widens the scale in place, the replay
    follows; the eager sync guard redoes a saturating call."""
    from miniworld_engine import fp8

    m, _ = _make_chai(starting=False)
    x, mask = _chai_x(256)
    f8.D64_MIN_L = 0
    _on("chai1")
    with torch.no_grad():
        y_e = m(x, mask).clone()  # calibrates
        assert not fp8.check()
        m(x, mask)
        g = torch.cuda.CUDAGraph()
        out = {}
        with torch.cuda.graph(g):
            out["y"] = m(x, mask)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out["y"], y_e)
    assert not fp8.check()
    (site,) = list(f8._SITES.values())
    site.vinv.mul_(2.0**12)
    site.vsc.copy_(1.0 / site.vinv)
    g.replay()
    torch.cuda.synchronize()
    assert fp8.check()
    for _ in range(8):
        g.replay()
        if not fp8.check():
            break
    assert not fp8.check()
    assert torch.isfinite(out["y"]).all()
    site.vinv.mul_(2.0**12)
    site.vsc.copy_(1.0 / site.vinv)
    st0 = dict(fp8.route_stats()["triattn"])
    with torch.no_grad():
        y = m(x, mask)
    st1 = fp8.route_stats()["triattn"]
    assert st1["retries"] > st0["retries"] or st1["fallbacks"] > st0["fallbacks"]
    assert torch.isfinite(y).all()


def test_d64_capture_before_calibration_runs_default(f8):
    """a capture of an uncalibrated d64 site runs the default core (capture_fallbacks); the replay equals the default bytes."""
    from miniworld_engine import fp8

    m, _ = _make_chai()
    x, mask = _chai_x(256)
    with torch.no_grad():
        y_def = m(x, mask).clone()
    f8.D64_MIN_L = 0
    _on("chai1")
    g = torch.cuda.CUDAGraph()
    out = {}
    with torch.no_grad(), torch.cuda.graph(g):
        out["y"] = m(x, mask)
    g.replay()
    torch.cuda.synchronize()
    assert fp8.route_stats()["triattn"]["capture_fallbacks"] >= 1
    assert torch.equal(out["y"], y_def)


def test_d64_env_model_at_import():
    """MINIWORLD_FP8=safe MINIWORLD_FP8_MODEL=chai1 (read at import) turns the d64 core on exactly as on the H100; af3 / no model leave the Chai-1 module on the default core."""
    import json
    import os
    import subprocess
    import sys

    _needs_5090()
    code = (
        "import json, torch; from miniworld_engine import fp8; from miniworld_engine.modules.triangle import TriangleAttention\n"
        "torch.manual_seed(0); m = TriangleAttention(256, 4, model='chai1', p_drop=0.0).cuda().eval().to(torch.bfloat16); m.set_backend('cuda')\n"
        "x = torch.randn(1, 512, 512, 256, device='cuda').bfloat16(); mk = torch.ones(1, 512, dtype=torch.bool, device='cuda')\n"
        "with torch.no_grad(): m(x, mk)\n"
        "print(json.dumps(fp8.route_stats()['triattn']))"
    )
    base = {k: v for k, v in os.environ.items() if not k.startswith(("MINIWORLD_FP8", "MINIWORLD_TRIATTN"))}

    def stats(**env):
        out = subprocess.run([sys.executable, "-c", code], env={**base, **env}, capture_output=True, text=True, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])

    a = stats(MINIWORLD_FP8="safe", MINIWORLD_FP8_MODEL="chai1")
    assert a.get("pv_d64", 0) == 1 and a.get("qk", 0) == 0, a
    for env in ({"MINIWORLD_FP8": "safe", "MINIWORLD_FP8_MODEL": "af3"}, {"MINIWORLD_FP8": "safe"}, {"MINIWORLD_FP8_MODEL": "chai1"}):
        st = stats(**env)
        assert st.get("pv_d64", 0) == 0, (env, st)

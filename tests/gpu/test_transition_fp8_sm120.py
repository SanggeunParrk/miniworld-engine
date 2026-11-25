"""RTX 5090: the e4m3 Transition inference forward (``kernels/transition/cuda/fp8_sm120``, ``fp8.SM120_ROUTES["transition"]``, docs/fp8.md "RTX 5090"): error against the
float64 module at the level of the H100 e4m3 Transition route (update rms 6.6 %, judge 4.5 % there), switched on by ``MINIWORLD_FP8=safe`` / ``engine-fp8`` exactly as
the H100 route, the guard (sync redo, lazy ``overflow_peak`` / ``rescale_all`` through ``fp8.check``), graph == eager, a capture before calibration runs the default,
ragged rows, and the default route untouched (bytes) while the switch is off."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_5090() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("needs an RTX 5090 (sm_120)")


def _module(n: int, d: int = 128, seed: int = 0):
    from foldjet_engine.modules.transition import Transition

    m = Transition(d, n, master_weights=True, backend="cuda").cuda().eval()
    g = torch.Generator(device="cpu").manual_seed(seed + n + d)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    return m


@pytest.fixture
def F():
    _needs_5090()
    from foldjet_engine.kernels.transition.cuda import fp8_sm120 as F

    assert F.POLICY is None  # off unless the preset / its variable is set
    saved = (F.GUARD, F.MIN_ROWS)
    F.POLICY, F.MIN_ROWS = "calib", 0
    yield F
    F.POLICY = None
    F.GUARD, F.MIN_ROWS = saved


def _err(y, y64, x):
    y, x = y.double(), x.double()
    judge = ((y - y64).abs().max() / y64.abs().max()).item()
    rms = ((y - y64).pow(2).mean().sqrt() / y64.pow(2).mean().sqrt()).item()
    upd = ((y - y64).pow(2).mean().sqrt() / (y64 - x).pow(2).mean().sqrt()).item()
    return judge, rms, upd


def test_perms_are_permutations() -> None:
    from foldjet_engine.kernels.transition.cuda.fp8_sm120 import _gb_cols, _o_perm, _perm_k1, _perm_k2

    for d in (128, 256):
        for p in (_perm_k1(d, "cpu"), _o_perm(d, "cpu"), _gb_cols(d, "cpu")):
            assert sorted(p.tolist()) == list(range(d))
        p1 = _perm_k1(d, "cpu").tolist()
        for i in range(d // 32):  # k32 step i: thread q's A bytes = its own columns 32 i + 8 q .. + 7
            for q in range(4):
                assert p1[32 * i + 4 * q : 32 * i + 4 * q + 4] == [32 * i + 8 * q + j for j in range(4)]
                assert p1[32 * i + 16 + 4 * q : 32 * i + 16 + 4 * q + 4] == [32 * i + 8 * q + 4 + j for j in range(4)]
    assert sorted(_perm_k2(512, "cpu").tolist()) == list(range(512))


@pytest.mark.parametrize("d,n", [(128, 4), (128, 2), (256, 4), (256, 2), (384, 4), (384, 2), (512, 4), (512, 2)])
@pytest.mark.parametrize("rows", [16384, 300 * 128 + 77])
def test_error_vs_fp64(F, d: int, n: int, rows: int) -> None:
    torch.manual_seed(0)
    m = _module(n, d)
    x = torch.randn(1, rows, d, device="cuda").bfloat16()
    ref = copy.deepcopy(m).double()
    ref.set_backend("pytorch")
    calls = F.GUARD_STATS["calls"]
    with torch.no_grad():
        y = m(x)
        y64 = ref(x.double())
        F.POLICY = None
        yd = m(x)  # the default route, same inputs
    assert F.GUARD_STATS["calls"] > calls  # the e4m3 route ran
    assert torch.isfinite(y.float()).all()
    judge, rms, upd = _err(y, y64, x)
    _j0, _r0, upd0 = _err(yd, y64, x)
    # H100 e4m3 route (docs/fp8.md, Transition "Error against fp64"): judge 0.041-0.045, rms 0.034-0.035, update 0.066
    assert judge < 0.08 and rms < 0.05 and upd < 0.09, (judge, rms, upd)
    assert upd > upd0  # (a sanity check that the e4m3 route, not the default, produced y)
    assert F.GUARD_STATS["fallbacks"] == 0


def test_safe_preset_leaves_it_off_explicit_variable_turns_it_on(monkeypatch) -> None:
    """RTX 5090: the e4m3 Transition is NOT in the ``safe`` preset / ``engine-fp8`` scope (``fp8.SM120_PRESET_EXCLUDED``; owner decision 2026-11-23,
    docs/fp8.md "RTX 5090": FoldForge end level 24 / 36 inside with it, 33 / 36 without). ``fp8.enable("safe")`` and the ``engine-fp8`` backend leave it
    off; with ``MINIWORLD_TRANSITION_FP8=calib`` set explicitly they turn it on; ``status()`` / ``route_stats()`` / ``enable(guard=)`` / ``disable()``
    act on this module (the slot of ``fp8.SM120_ROUTES``)."""
    _needs_5090()
    from foldjet_engine import fp8
    from foldjet_engine.kernels.transition.cuda import fp8_sm120 as F

    assert fp8.SM120_ROUTES["transition"] == F.__name__ and "transition" in fp8.SM120_PRESET_EXCLUDED
    assert F.POLICY is None  # off by default
    m = _module(4)
    x = torch.randn(1, 16384, 128, device="cuda").bfloat16()  # above the 12288-row gate
    guard = F.GUARD
    try:
        monkeypatch.delenv("MINIWORLD_TRANSITION_FP8", raising=False)
        monkeypatch.delenv("MINIWORLD_TRANSITION_FP8_SM120", raising=False)
        for model in (None, "af3", "boltz2"):
            fp8.enable("safe", model=model)
            assert F.POLICY is None and fp8.status()["transition"] is None, model
        fp8.reset_route_stats()
        with torch.no_grad():
            m(x)
        assert fp8.route_stats()["transition"]["calls"] == 0  # safe: the default Transition
        m.set_backend("engine-fp8")
        with torch.no_grad():
            m(x)
        assert fp8.route_stats()["transition"]["calls"] == 0  # engine-fp8 scope: the default Transition
        m.set_backend("cuda")
        fp8.disable()
        # the explicit variable: on under safe and inside the engine-fp8 scope
        monkeypatch.setenv("MINIWORLD_TRANSITION_FP8", "calib")
        fp8.enable("safe", guard="lazy")
        assert F.POLICY == "calib" and fp8.status()["transition"] == "calib"
        assert F.GUARD == "lazy" and fp8.status()["guard"]["transition"] == "lazy"
        fp8.enable("safe", guard="sync")
        fp8.reset_route_stats()
        with torch.no_grad():
            m(x)
        st = fp8.route_stats()["transition"]
        assert st["calls"] >= 1 and st == F.GUARD_STATS, st
        assert not fp8.check()
        fp8.disable()
        assert F.POLICY is None and fp8.status()["transition"] is None
        m.set_backend("engine-fp8")
        fp8.reset_route_stats()
        with torch.no_grad():
            m(x)
        assert fp8.route_stats()["transition"]["calls"] >= 1
        assert F.POLICY is None  # the process's switch after the scope
        monkeypatch.setenv("MINIWORLD_TRANSITION_FP8_SM120", "")  # the alias wins over the generic name
        fp8.enable("safe")
        assert F.POLICY is None
    finally:
        fp8.disable()
        F.GUARD = guard
        m.set_backend("cuda")


def test_disable_turns_it_off(F, monkeypatch) -> None:
    from foldjet_engine import fp8

    fp8.disable()
    assert F.POLICY is None
    monkeypatch.setenv("MINIWORLD_TRANSITION_FP8", "calib")  # the explicit opt-in (the route is not in safe on the 5090)
    fp8.enable("safe")
    assert F.POLICY == "calib"
    fp8.disable()
    assert F.POLICY is None


def test_guard_sync_and_lazy(F) -> None:
    from foldjet_engine import fp8

    m = _module(4)
    x = torch.randn(1, 16384, 128, device="cuda").bfloat16()
    with torch.no_grad():
        y_ok = m(x)
        cache = F._CACHES[m.expand_a.weight.data_ptr()]
        cache["sh"].mul_(256.0)  # force a saturating h scale
        F.refresh(cache)
        retries = F.GUARD_STATS["retries"]
        y = m(x)  # sync guard: reads the peak, lowers sh, redoes the call
        assert F.GUARD_STATS["retries"] > retries
        F.GUARD = "lazy"
        cache["sh"].mul_(256.0)
        F.refresh(cache)
        F.overflow_peak()
        m(x)
        assert fp8.check()  # the lazy flag is read by fp8.check() and the scale lowered in place
        y2 = m(x)
        assert F.overflow_peak() <= 448.0
    for t in (y, y2):
        assert ((t.float() - y_ok.float()).abs().max() / y_ok.float().abs().max()).item() < 0.05


def test_graph_equals_eager_and_uncalibrated_capture(F) -> None:
    F.GUARD = "lazy"
    m = _module(4, seed=7)
    x = torch.randn(1, 20000, 128, device="cuda").bfloat16()
    fb = F.GUARD_STATS["capture_fallbacks"]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        F.POLICY = None
        yd = m(x)  # the default route, eager (its weight forms exist before a capture reads them)
        F.POLICY = "calib"
        g0 = torch.cuda.CUDAGraph()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.cuda.graph(g0, stream=s):
            y0 = m(x)  # not calibrated: the default route is captured
        torch.cuda.current_stream().wait_stream(s)
        g0.replay()
        torch.cuda.synchronize()
        assert F.GUARD_STATS["capture_fallbacks"] > fb
        assert torch.equal(y0, yd)
        ye = m(x)  # eager: calibrates
        g = torch.cuda.CUDAGraph()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
            yg = m(x)
        torch.cuda.current_stream().wait_stream(s)
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(yg, ye)


def test_rows_are_independent(F) -> None:
    """One call == the same rows in two calls (batch / ragged rows; the kernel predicates the tail tile)."""
    m = _module(2)
    x = torch.randn(40000, 128, device="cuda").bfloat16()
    with torch.no_grad():
        y = m(x)
        ya, yb = m(x[:23011]), m(x[23011:])
    assert torch.equal(y, torch.cat([ya, yb]))


def test_default_route_untouched_when_off() -> None:
    _needs_5090()
    from foldjet_engine.kernels.transition.cuda import fp8_sm120 as F
    from foldjet_engine.kernels.transition.cuda import transition_sm80

    assert F.POLICY is None
    m = _module(4)
    x = torch.randn(1, 65536, 128, device="cuda").bfloat16()
    calls = F.GUARD_STATS["calls"]
    with torch.no_grad():
        y = m(x)
        y2 = transition_sm80(x, m.ln_in.weight, m.ln_in.bias, m.expand_a.weight, m.expand_b.weight, m.squeeze.weight, m.ln_in.eps, fp8=False)
    assert F.GUARD_STATS["calls"] == calls
    assert torch.equal(y, y2)

"""H100: the experimental e4m3 D256 Transition forward (``kernels/transition/cuda/transition_fp8_d256_sm90a.cu`` through ``fp8_sm90a``, opt-in,
docs/fp8.md): the route takes D 256 (n 4 / 2) with the expand in e4m3 (G1 = 1) or bf16 (G1 = 0, "h + squeeze"), stays inside the e4m3 storage
error of the fp64 reference, the guard catches a saturated h scale (sync retry and lazy ``overflow_peak`` / ``rescale_all``), a CUDA-graph
replay of a calibrated module gives the eager bytes, the row gate keeps small calls on the 2.5.12 route, and the default is untouched while the
switch is off."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.gpu]

D = 256


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _module(n: int):
    from miniworld_engine.modules.transition import Transition

    m = Transition(D, n, master_weights=True, backend="cuda").cuda().eval()
    g = torch.Generator(device="cpu").manual_seed(n)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    return m


def _variant(F, g1: int = 1, slots: int = 4, guard: str = "sync", min_rows: int | None = 0):
    F.POLICY, F.GUARD, F.G1, F.G2, F.SIGT, F.SLOTS, F.ABL, F.MIN_ROWS = "calib", guard, g1, 1, 1, slots, 0, min_rows


def _errs(y: torch.Tensor, y64: torch.Tensor) -> tuple[float, float]:
    d = y.double() - y64
    return (d.abs().max() / y64.abs().max()).item(), (d.pow(2).mean().sqrt() / y64.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("n", [4, 2])
@pytest.mark.parametrize("rows", [128, 16384, 300 * 128])
@pytest.mark.parametrize("g1", [1, 0])
def test_d256_fp8_error_vs_fp64(n: int, rows: int, g1: int) -> None:
    _needs_h100()
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as F

    assert F.POLICY is None
    torch.manual_seed(0)
    m = _module(n)
    x = torch.randn(1, rows, D, device="cuda").bfloat16()
    ref = copy.deepcopy(m).double()
    ref.set_backend("pytorch")
    fallbacks = F.GUARD_STATS["fallbacks"]
    try:
        _variant(F, g1)
        with torch.no_grad():
            y = m(x)
            y64 = ref(x.double())
    finally:
        F.POLICY = None
    assert torch.isfinite(y.float()).all()
    err, rms = _errs(y, y64)
    # e4m3 storage error: about 3.5 % rms with the e4m3 expand, 2 % with the bf16 expand (h + squeeze only); the max at most ~2x the rms
    assert err < 0.08 and rms < (0.05 if g1 else 0.03), (err, rms)
    assert F.GUARD_STATS["fallbacks"] == fallbacks
    assert F._CACHES[m.expand_a.weight.data_ptr()]["d256"] == n


@pytest.mark.parametrize("n", [4, 2])
def test_d256_graph_replay_matches_eager(n: int) -> None:
    """Calibrated eagerly, then captured (lazy guard): the replay writes the eager call's bytes (the kernel is deterministic)."""
    _needs_h100()
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as F

    m = _module(n)
    x = torch.randn(1, 64 * 128, D, device="cuda").bfloat16()
    try:
        _variant(F, guard="lazy")
        with torch.no_grad():
            m(x)
            y_eager = m(x).clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                m(x)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y_graph = m(x)
            g.replay()
            torch.cuda.synchronize()
        assert F.overflow_peak() <= 448.0
    finally:
        F.POLICY = None
    assert torch.equal(y_eager, y_graph)


def test_d256_guard_rescales_a_saturated_h_scale() -> None:
    _needs_h100()
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as F

    m = _module(4)
    x = torch.randn(1, 4096, D, device="cuda").bfloat16()
    try:
        _variant(F)
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
            peak = F.overflow_peak()
            assert peak > 448.0
            F.rescale_all(peak)
            y2 = m(x)
            assert F.overflow_peak() <= 448.0
    finally:
        F.POLICY = None
    for t in (y, y2):
        assert ((t.float() - y_ok.float()).abs().max() / y_ok.float().abs().max()).item() < 0.05


def test_d256_row_gate() -> None:
    """With MIN_ROWS unset, a D 256 call below ROW_GATE[256] rows takes the 2.5.12 route (no FP8 call counted); at the gate it is served."""
    _needs_h100()
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as F

    m = _module(4)
    gate = F.ROW_GATE[D]
    try:
        _variant(F, min_rows=None)
        assert not F.serves(torch.empty(1, gate - 128, D, device="cuda", dtype=torch.bfloat16), m.expand_a.weight, m.squeeze.weight)
        assert F.serves(torch.empty(1, gate, D, device="cuda", dtype=torch.bfloat16), m.expand_a.weight, m.squeeze.weight)
        calls = F.GUARD_STATS["calls"]
        with torch.no_grad():
            m(torch.randn(1, gate - 128, D, device="cuda").bfloat16())
        assert F.GUARD_STATS["calls"] == calls
    finally:
        F.POLICY = None


def test_d256_default_route_untouched_when_off() -> None:
    _needs_h100()
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as F

    assert F.POLICY is None
    m = _module(4)
    x = torch.randn(1, 16384, D, device="cuda").bfloat16()
    calls = F.GUARD_STATS["calls"]
    with torch.no_grad():
        m(x)
    assert F.GUARD_STATS["calls"] == calls

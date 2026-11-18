"""H100 (opt-in FP8 route, off by default): the e4m3 TriMul planes at D512 (IntelliFold2's trunk pairformer): the direct n64-block F8 K1
(``h100_b200style_direct.F8_CFG[512]``), the cuBLASLt e4m3 contraction and ``k3reorder.cu`` with ``K3_CSCALE=2`` (token-flat K3, L % 128 == 0, one
direction). Accuracy bounds only (the e4m3 storage error is ~3.7 % of the update, bf16 floor 0.65 %); the default route is not touched (POLICY None)."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _module(outgoing):
    from miniworld_engine.modules.triangle import TriangleMultiplication

    d = 512
    m = TriangleMultiplication(d, outgoing=outgoing, master_weights=True, backend="cuda").cuda().eval()
    ref = TriangleMultiplication(d, outgoing=outgoing, backend="pytorch").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
        ref.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
    return m, ref


def _route(m, x, mask):
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    old = (F8.POLICY, F8.GUARD, F8._MINL_ENV)
    try:
        F8.POLICY, F8.GUARD, F8._MINL_ENV = "calib", "lazy", "0"
        F8._CACHES.clear()
        m(x, mask)
        F8.overflow_peak()
        return m(x, mask), F8.overflow_peak()
    finally:
        F8.POLICY, F8.GUARD, F8._MINL_ENV = old
        F8._CACHES.clear()


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("n", [128, 256, 384])
def test_d512_planes_accuracy(outgoing: bool, n: int) -> None:
    _needs_h100()
    torch.manual_seed(0)
    m, ref = _module(outgoing)
    x = torch.randn(1, n, n, 512, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[0, n - 37 :] = False
    with torch.no_grad():
        y32 = ref(x.float(), mask)
        y, peak = _route(m, x, mask)
        y16 = m(x, mask)
    assert torch.isfinite(y.float()).all()
    assert peak <= 448.0
    assert _rel(y - x, y32 - x.float()) < 0.06
    assert _rel(y16 - x, y32 - x.float()) < 0.02
    assert not torch.equal(y, y16)  # the FP8 route ran


def test_d512_gates() -> None:
    """One direction at L % 128 == 0 only; bidirectional D512 and other lengths keep the bf16 default route."""
    _needs_h100()
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    m, _ = _module(True)
    old = (F8.POLICY, F8._MINL_ENV)
    try:
        F8.POLICY, F8._MINL_ENV = "calib", "0"
        x = torch.zeros(1, 256, 256, 512, device="cuda", dtype=torch.bfloat16)
        assert F8.serves(m, x, None, False)
        assert not F8.serves(m, x, None, True)
        assert not F8.serves(m, torch.zeros(1, 200, 200, 512, device="cuda", dtype=torch.bfloat16), None, False)
        meta = lambda b, n: torch.empty(b, n, n, 512, device="meta", dtype=torch.bfloat16)  # (only the shape is read)
        assert F8.serves(m, meta(1, 2048), None, False)
        assert not F8.serves(m, meta(2, 256), None, False)  # B > 1: not measured
        assert not F8.serves(m, meta(1, 2304), None, False)  # beyond the measured L <= 2048
        F8.D512 = False
        assert not F8.serves(m, x, None, False)
        F8.D512 = True
    finally:
        F8.POLICY, F8._MINL_ENV = old
        F8.D512 = True


def test_d512_default_untouched_when_off() -> None:
    """FP8 off (POLICY None): the default route's output does not depend on the FP8 module state."""
    _needs_h100()
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    assert F8.POLICY is None
    torch.manual_seed(2)
    m, _ = _module(True)
    n = 256
    x = torch.randn(1, n, n, 512, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        a = m(x, mask)
        _route(m, x, mask)
        b = m(x, mask)
    assert torch.equal(a, b)


def test_d512_graph_replay_equals_eager() -> None:
    _needs_h100()
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    torch.manual_seed(3)
    m, _ = _module(True)
    n = 256
    x = torch.randn(1, n, n, 512, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    old = (F8.POLICY, F8.GUARD, F8._MINL_ENV)
    try:
        F8.POLICY, F8.GUARD, F8._MINL_ENV = "calib", "lazy", "0"
        F8._CACHES.clear()
        F8.GUARD_STATS["capture_fallbacks"] = 0
        with torch.no_grad():
            m(x, mask)
            m(x, mask)
            eager = m(x, mask)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                m(x, mask)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = m(x, mask)
            g.replay()
            torch.cuda.synchronize()
        assert F8.GUARD_STATS["capture_fallbacks"] == 0
        assert torch.equal(out, eager)
    finally:
        F8.POLICY, F8.GUARD, F8._MINL_ENV = old
        F8._CACHES.clear()

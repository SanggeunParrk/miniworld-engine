"""H100, 3.0.0 exploration (opt-in, off by default): the e4m3 TriMul planes at the n64-block widths (D64 / D384, ``tmn_kernels.cuh`` TMN_K1_F8 n64
store) and B > 1 through the one-sample route per sample. Accuracy bounds only (the e4m3 storage error is ~3.7 % of the update); the bf16
default route is not touched (POLICY None)."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _modules(d, outgoing, bidirectional=False):
    from foldjet_engine.modules.triangle import BidirectionalTriangleMultiplication, TriangleMultiplication

    if bidirectional:
        m = BidirectionalTriangleMultiplication(d, d_hidden=d, master_weights=True, backend="cuda").cuda().eval()
        ref = BidirectionalTriangleMultiplication(d, d_hidden=d, backend="pytorch").cuda().eval()
    else:
        m = TriangleMultiplication(d, outgoing=outgoing, master_weights=True, backend="cuda").cuda().eval()
        ref = TriangleMultiplication(d, outgoing=outgoing, backend="pytorch").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
        ref.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
    return m, ref


def _route(m, x, mask):
    """y of the module with the FP8 route on (calibrated on a first eager call, every min-L gate off), restored afterwards."""
    from foldjet_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    old = (F8.POLICY, F8.GUARD, F8._MINL_ENV)
    try:
        F8.POLICY, F8.GUARD, F8._MINL_ENV = "calib", "lazy", "0"
        F8._CACHES.clear()
        m(x, mask)
        F8.overflow_peak()  # (reset: the calibration call)
        return m(x, mask), F8.overflow_peak()
    finally:
        F8.POLICY, F8.GUARD, F8._MINL_ENV = old
        F8._CACHES.clear()


@pytest.mark.parametrize("d", [64, 384])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("n", [100, 256, 300])
def test_n64_block_widths(d: int, outgoing: bool, n: int) -> None:
    _needs_h100()
    torch.manual_seed(0)
    m, ref = _modules(d, outgoing)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[0, n - 37 :] = False
    with torch.no_grad():
        y32 = ref(x.float(), mask)
        y, _ = _route(m, x, mask)
        y16 = m(x, mask)
    assert torch.isfinite(y.float()).all()
    assert _rel(y - x, y32 - x.float()) < 0.06
    assert not torch.equal(y, y16)  # the FP8 route ran (the bf16 default differs)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("outgoing", [True, False])
def test_samples_equal_one_sample_calls(d: int, outgoing: bool) -> None:
    """B > 1 runs the one-sample route per sample with the module's static scales: each sample's rows equal a B = 1 call of the same module."""
    _needs_h100()
    from foldjet_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    torch.manual_seed(1)
    n = 256
    m, _ = _modules(d, outgoing)
    x = torch.randn(3, n, n, d, device="cuda").bfloat16()
    mask = torch.ones(3, n, dtype=torch.bool, device="cuda")
    mask[1, n - 11 :] = False
    old = (F8.POLICY, F8.GUARD, F8._MINL_ENV)
    try:
        F8.POLICY, F8.GUARD, F8._MINL_ENV = "calib", "sync", "0"
        F8._CACHES.clear()
        with torch.no_grad():
            m(x[:1], mask[:1])  # calibration on sample 0
            y = m(x, mask)
            ys = [m(x[b : b + 1], mask[b : b + 1]) for b in range(3)]
    finally:
        F8.POLICY, F8.GUARD, F8._MINL_ENV = old
        F8._CACHES.clear()
    for b in range(3):
        assert torch.equal(y[b : b + 1], ys[b])


@pytest.mark.parametrize("d", [64, 384])
def test_n64_block_widths_bidirectional(d: int) -> None:
    _needs_h100()
    torch.manual_seed(3)
    n = 256
    m, ref = _modules(d, True, bidirectional=True)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[0, n - 5 :] = False
    with torch.no_grad():
        y32 = ref(x.float(), mask)
        y, peak = _route(m, x, mask)
    assert torch.isfinite(y.float()).all()
    assert peak <= 448.0
    assert _rel(y - x, y32 - x.float()) < 0.06

"""H100: the e4m3 TriMul planes (``h100_b200style_fp8``) on RoseTTAFold3's ``TriangleMultiplicationRF3``.

RF3's module hands the kernels ``to_left.weight * (1 / n)`` -- a new tensor in every call (``_LeftScaled``). The FP8 route keys its static
calibration on the parameter and applies the factor to the contraction (the channel scale ``csc``), so:

* one calibration per module (no state per call), and a CUDA-graph capture after the eager warm-up takes the FP8 route (no capture fallback),
* the result stays inside the e4m3 storage error of the fp64 reference (about the plain module's), for any real length (masks),
* a second call with another real length (another factor) needs no recalibration and stays accurate.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _rel(a, b) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@pytest.fixture
def fp8_calib():
    from foldjet_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    old = (F8.POLICY, F8.GUARD, F8._MINL_ENV)
    F8.POLICY, F8.GUARD, F8._MINL_ENV = "calib", "lazy", "0"
    F8._CACHES.clear()
    yield F8
    F8.POLICY, F8.GUARD, F8._MINL_ENV = old
    F8._CACHES.clear()


def _make(d: int, outgoing: bool, seed: int = 0):
    from foldjet_engine.modules.triangle import TriangleMultiplication

    torch.manual_seed(seed)
    m = TriangleMultiplication(d, outgoing=outgoing, master_weights=True, backend="cuda", model="rf3").cuda().eval()
    ref = TriangleMultiplication(d, outgoing=outgoing, backend="pytorch", model="rf3").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
        ref.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
        ref = ref.double()
    return m, ref


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_rf3_fp8_planes_accuracy_and_one_calibration(fp8_calib, d: int, outgoing: bool, masked: bool) -> None:
    _needs_h100()
    F8 = fp8_calib
    n = 256
    m, ref = _make(d, outgoing)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    mask = None
    if masked:
        mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
        mask[0, n - 37 :] = False
    F8.GUARD_STATS.update(calls=0, capture_fallbacks=0)
    with torch.no_grad():
        y_fp8 = m(x, mask)
        y_fp8b = m(x, mask)
        F8.POLICY = None
        y_bf16 = m(x, mask)  # the default bf16 route (FP8 off)
        F8.POLICY = "calib"
        y64 = ref(x.double(), mask)
    assert F8.GUARD_STATS["calls"] == 2
    assert len(F8._CACHES) == 1  # one calibration for the module, not one per call
    assert torch.equal(y_fp8, y_fp8b)
    e8, e16 = _rel(y_fp8 - x, y64 - x.double()), _rel(y_bf16 - x, y64 - x.double())
    assert e8 < 0.06, (e8, e16)  # the e4m3 storage error (the plain module: 4 %, 6x the bf16 floor)


@pytest.mark.parametrize("outgoing", [True, False])
def test_rf3_fp8_capture_takes_the_fp8_route_and_follows_the_length_factor(fp8_calib, outgoing: bool) -> None:
    """Eager warm-up calibrates; a graph captured afterwards runs the FP8 route (no capture fallback), and replaying it after the mask's real length
    changed (the factor is a device value read at replay) gives the right update without a recalibration."""
    _needs_h100()
    F8 = fp8_calib
    d, n = 128, 256
    m, ref = _make(d, outgoing, seed=3)
    x = torch.randn(1, n, n, d, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[0, n - 37 :] = False
    with torch.no_grad():
        for _ in range(2):  # FoldForge's _Graph warm-up
            m(x, mask)
        assert len(F8._CACHES) == 1
        F8.GUARD_STATS.update(calls=0, capture_fallbacks=0)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(x, mask)
        assert F8.GUARD_STATS["capture_fallbacks"] == 0
        assert F8.GUARD_STATS["calls"] == 1
        g.replay()
        torch.cuda.synchronize()
        y_a = out.clone()
        mask[0, n - 90 :] = False  # another real length: another factor, same graph
        g.replay()
        torch.cuda.synchronize()
        y_b = out.clone()
        y64_a = ref(x.double(), torch.cat([torch.ones(1, n - 37), torch.zeros(1, 37)], 1).bool().cuda())
        y64_b = ref(x.double(), mask)
    assert len(F8._CACHES) == 1
    assert _rel(y_a - x, y64_a - x.double()) < 0.06
    assert _rel(y_b - x, y64_b - x.double()) < 0.06
    assert not torch.equal(y_a, y_b)

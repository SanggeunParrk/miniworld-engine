"""H100: the opt-in e4m3 TriMul planes route for the BIDIRECTIONAL block (``h100_b200style_fp8``, direction 0: first d_hidden channels outgoing, second
incoming; two single-direction K1 + e4m3 contraction launches into one ``tri``, one K3 over 2 d channels with per-channel scales): the update stays at the
single-direction e4m3 level (about 4 % of the fp32 reference update), the calibrated route is capturable, below the min-L gate the bf16 default runs
bit-identically, and the pow2 scale has a floor."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _module(d: int):
    from foldjet_engine.modules.triangle import BidirectionalTriangleMultiplication

    torch.manual_seed(0)
    m = BidirectionalTriangleMultiplication(d, master_weights=True, backend="cuda").cuda().eval()
    ref = BidirectionalTriangleMultiplication(d, backend="pytorch").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
        ref.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
    return m, ref


def _leaves(m, x):
    from foldjet_engine.kernels.trimul_inproj.cuda._h100_pack import cast_training_weights

    mats = [m.to_left.weight, m.to_left_gate.weight, m.to_right.weight, m.to_right_gate.weight, m.to_gate.weight, m.to_out.weight]
    return [x, *cast_training_weights(mats), m.ln_pair.weight, m.ln_pair.bias, m.ln_out.weight, m.ln_out.bias]


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("n", [40, 100, 256, 300, 771])
def test_bidirectional_fp8_planes_accuracy(d: int, n: int) -> None:
    _needs_h100()
    from foldjet_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    assert F8.POLICY is None
    m, ref = _module(d)
    with torch.no_grad():
        x = torch.randn(1, n, n, d, device="cuda").bfloat16()
        mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
        mask[0, n - 37 :] = False
        y32 = ref(x.float(), mask)
        y = F8.run(_leaves(m, x), mask, 0)
    assert _rel(y - x, y32 - x.float()) < 0.06
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("d", [128, 256])
def test_bidirectional_fp8_k3_pipe_bit_identical(d: int) -> None:
    """The K3 pipeline macros (K3_EXTRA) on the bidirectional FP8 K3 (h = 2 d, per-channel scales) are bit-identical to the plain K3."""
    _needs_h100()
    from foldjet_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    m, _ = _module(d)
    n = 300
    with torch.no_grad():
        x = torch.randn(1, n, n, d, device="cuda").bfloat16()
        mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
        leaves = _leaves(m, x)
        old = dict(F8.K.K3_EXTRA)
        try:
            F8.K.K3_EXTRA.clear()
            y_plain = F8.run(leaves, mask, 0, margin=16.0)
            F8.K.K3_EXTRA.update(old)
            y_pipe = F8.run(leaves, mask, 0, margin=16.0)
        finally:
            F8.K.K3_EXTRA.clear()
            F8.K.K3_EXTRA.update(old)
    assert torch.equal(y_plain, y_pipe)


@pytest.mark.parametrize("d,n", [(128, 384), (256, 1024)])
def test_bidirectional_calibrated_route_graph_and_guard(d: int, n: int) -> None:
    """Module hook, calibrated scales: eager call = graph replay (static device tensors), the update stays at the e4m3 level, the guard peak is below 448,
    ``rescale_all`` keeps the route finite and consistent, and the bf16 default is untouched once the policy is off."""
    _needs_h100()
    from foldjet_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    m, ref = _module(d)
    old = (F8.POLICY, F8.GUARD)
    try:
        with torch.no_grad():
            x = torch.randn(1, n, n, d, device="cuda").bfloat16()
            mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
            F8.POLICY, F8.GUARD = None, "lazy"
            y_default = m(x, mask)
            F8._CACHES.clear()
            F8.POLICY = "calib"
            y0 = m(x, mask)  # calibrates (eager)
            y1 = m(x, mask)
            assert F8.overflow_peak() < 448.0
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                yg = m(x, mask)
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(y1, yg) and torch.equal(y0, y1)
            y32 = ref(x.float(), mask)
            assert _rel(y1 - x, y32 - x.float()) < 0.06
            assert len(F8._CACHES) == 1 and set(next(iter(F8._CACHES.values()))["bd"]) == {1, 2}
            F8.rescale_all(900.0)  # factor 4: scales x4, K1 weights / K3 operand rebuilt in place
            y2 = m(x, mask)
            assert torch.isfinite(y2.float()).all() and _rel(y2 - x, y32 - x.float()) < 0.08
            F8.POLICY = None
            assert torch.equal(m(x, mask), y_default)
    finally:
        F8.POLICY, F8.GUARD = old
        F8._CACHES.clear()


def test_bidirectional_below_min_l_is_the_default_route() -> None:
    _needs_h100()
    from foldjet_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8

    m, _ = _module(128)
    n = 192  # below the D128 bidirectional gate
    assert n < F8.MIN_L[(128, 0)]
    old = F8.POLICY
    try:
        with torch.no_grad():
            x = torch.randn(1, n, n, 128, device="cuda").bfloat16()
            F8.POLICY = None
            y_default = m(x, None)
            F8._CACHES.clear()
            F8.POLICY = "calib"
            y = m(x, None)
            assert not F8._CACHES
            assert torch.equal(y, y_default)
    finally:
        F8.POLICY = old


def test_pow2_scale_floor() -> None:
    """A channel whose amax is 0 or denormal-sized must not get 2^-99 (saturates the e4m3 cast on any later input); normal channels are unchanged."""
    _needs_h100()
    from foldjet_engine.kernels.trimul_inproj.cuda import _h100_fp8 as F

    amax = torch.tensor([0.0, 1e-38, 1e-12, 1e-6, 1e-3, 0.5, 448.0, 1000.0], device="cuda")
    s = F.pow2_scale(amax)
    assert s[0] == 1.0
    assert (s[1:] >= 2.0**-30).all()
    # unchanged for amax >= 448 * 2^-30: the exact pow2 above amax / 448
    plain = torch.exp2(torch.ceil(torch.log2(amax[3:] / 448.0)))
    assert torch.equal(s[3:], plain)

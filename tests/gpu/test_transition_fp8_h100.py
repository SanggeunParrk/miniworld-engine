"""H100: the experimental e4m3 D128 Transition forward (``kernels/transition/cuda/fp8_sm90a``, opt-in, docs/fp8.md): the FP8 route (D 128, and D 384 / 512) stays inside
the e4m3 storage error of the fp64 reference (about 3.5 % of the output, 8 % at most), the bf16-expand / bf16-squeeze build with the 2.5.12
sigmoid reproduces the 2.5.12 default bytes (the same kernel structure), the guard catches a saturated h scale, and the default route is
untouched while the switch is off (POLICY None)."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _module(n: int, d: int = 128):
    from miniworld_engine.modules.transition import Transition

    m = Transition(d, n, master_weights=True, backend="cuda").cuda().eval()
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


def _variant(F, g1=1, g2=1, sigt=1, slots=4, guard="sync"):
    F.POLICY, F.GUARD, F.G1, F.G2, F.SIGT, F.SLOTS, F.ABL, F.MIN_ROWS = "calib", guard, g1, g2, sigt, slots, 0, 0


def test_perm_k_is_the_fragment_order() -> None:
    """Logical position 4 t + q of an e4m3 k32 fragment reads the physical column a bf16 / fp32 fragment thread t holds at its q-th slot."""
    from miniworld_engine.kernels.transition.cuda.fp8_sm90a import _perm_k

    p = _perm_k(64, "cpu").tolist()
    assert sorted(p) == list(range(64))
    for blk in range(2):
        for half in range(2):
            for t in range(4):
                held = [32 * blk + 16 * half + c for c in (2 * t, 2 * t + 1, 8 + 2 * t, 9 + 2 * t)]
                assert p[32 * blk + 16 * half + 4 * t : 32 * blk + 16 * half + 4 * t + 4] == held


@pytest.mark.parametrize("n", [4, 2])
@pytest.mark.parametrize("rows", [128, 16384, 300 * 128])
@pytest.mark.parametrize("cfg", [(1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)])
def test_fp8_error_vs_fp64(n: int, rows: int, cfg) -> None:
    _needs_h100()
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as F

    assert F.POLICY is None
    torch.manual_seed(0)
    m = _module(n)
    x = torch.randn(1, rows, 128, device="cuda").bfloat16()
    ref = copy.deepcopy(m).double()
    ref.set_backend("pytorch")
    try:
        _variant(F, *cfg)
        with torch.no_grad():
            y = m(x)
            y64 = ref(x.double())
    finally:
        F.POLICY = None
    assert torch.isfinite(y.float()).all()
    err = ((y.double() - y64).abs().max() / y64.abs().max()).item()
    rms = ((y.double() - y64).pow(2).mean().sqrt() / y64.pow(2).mean().sqrt()).item()
    assert err < 0.08 and rms < 0.05, (err, rms)
    assert F.GUARD_STATS["fallbacks"] == 0


@pytest.mark.parametrize("d", [384, 512])
@pytest.mark.parametrize("n", [4, 2])
@pytest.mark.parametrize("g1", [1, 0])
def test_wide_fp8_error_vs_fp64(d: int, n: int, g1: int) -> None:
    """D 384 / 512: e4m3 h + e4m3 squeeze (and the e4m3 expand with G1): inside the e4m3 storage error, finite, no guard fallback."""
    _needs_h100()
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as F

    m = _module(n, d)
    x = torch.randn(1, 300 * 128, d, device="cuda").bfloat16()
    ref = copy.deepcopy(m).double()
    ref.set_backend("pytorch")
    try:
        _variant(F, g1)
        with torch.no_grad():
            y = m(x)
            y64 = ref(x.double())
    finally:
        F.POLICY = None
    assert torch.isfinite(y.float()).all()
    err = ((y.double() - y64).abs().max() / y64.abs().max()).item()
    rms = ((y.double() - y64).pow(2).mean().sqrt() / y64.pow(2).mean().sqrt()).item()
    assert err < 0.08 and rms < 0.05, (err, rms)
    assert F._CACHES[m.expand_a.weight.data_ptr()]["wide"] == n


@pytest.mark.parametrize("n", [4, 2])
def test_bf16_build_reproduces_default(n: int) -> None:
    """G1 = G2 = 0 with the 2.5.12 sigmoid: the same products, roundings and accumulation order as the default kernel -> the same bytes."""
    _needs_h100()
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as F

    m = _module(n)
    x = torch.randn(1, 64 * 128, 128, device="cuda").bfloat16()
    with torch.no_grad():
        y0 = m(x)
        try:
            _variant(F, 0, 0, 0, 2)
            y1 = copy.deepcopy(m)(x)
        finally:
            F.POLICY = None
    assert torch.equal(y0, y1)


def test_guard_rescales_a_saturated_h_scale() -> None:
    _needs_h100()
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as F

    m = _module(4)
    x = torch.randn(1, 4096, 128, device="cuda").bfloat16()
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


def test_default_route_untouched_when_off() -> None:
    _needs_h100()
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as F

    assert F.POLICY is None
    m = _module(4)
    x = torch.randn(1, 2048, 128, device="cuda").bfloat16()
    calls = F.GUARD_STATS["calls"]
    with torch.no_grad():
        m(x)
    assert F.GUARD_STATS["calls"] == calls

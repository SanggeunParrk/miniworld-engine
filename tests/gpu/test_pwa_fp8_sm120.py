"""RTX 5090: the opt-in e4m3 PWA key contraction (``kernels/integrations/pwa_fp8_sm120``, ``fp8.SM120_ROUTES["pwa"]``, docs/fp8.md "RTX 5090"):
``ln_vt8`` against the default ``ln_v`` (every native width, a clipped token tail), ``w_quant``, error against the float64 module at the level of the
H100 routes (judge 0.30-0.38 %, update rms 2.8-3.6 % there; inside the engine tolerance 8e-3) at the native widths and the BIG width, ``safe`` turns it
on, the guard (sync redo, lazy ``take_overflow`` / ``rescale_all`` through ``fp8.check``), graph == eager, a capture before calibration runs the default,
and the default route untouched (bytes) after the route ran."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.gpu]

WIDTHS = [(64, 32, 128), (64, 8, 128), (64, 16, 128), (128, 8, 256), (128, 16, 256), (256, 32, 512)]  # (d_msa, C, d_pair)


def _needs_5090() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("needs an RTX 5090 (sm_120)")


def _case(d: int, c: int, cz: int, s: int, n: int, seed: int = 0):
    from miniworld_engine.modules.msa import MSAPairWeightedAveraging

    m = MSAPairWeightedAveraging(d, cz, n_head=8, d_hidden=c, master_weights=True, backend="cuda").cuda().eval()
    g = torch.Generator(device="cpu").manual_seed(seed + d + c + cz)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    torch.manual_seed(seed)
    x = torch.randn(1, s, n, d, device="cuda").bfloat16()
    z = torch.randn(1, n, n, cz, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[0, torch.tensor([1, n // 2, n - 1])] = False
    return m, (x, z, mask)


@pytest.fixture
def P():
    _needs_5090()
    from miniworld_engine.kernels.integrations import pwa_fp8_sm120 as P

    assert not P.ENABLED  # off unless switched on
    saved = (P.GUARD, P.MIN_SN_ALL, P.BIG_MIN_SN, P.HEADROOM)
    P.ENABLED, P.MIN_SN_ALL, P.BIG_MIN_SN = True, 0, 0
    yield P
    P.ENABLED = False
    P.GUARD, P.MIN_SN_ALL, P.BIG_MIN_SN, P.HEADROOM = saved
    P._VS.clear()
    P._VINV.clear()
    P.take_overflow()


def _run(m, inp):
    with torch.no_grad():
        return m(*inp)


@pytest.mark.parametrize("d,c", [(64, 8), (64, 16), (64, 32), (128, 8), (128, 16)])
@pytest.mark.parametrize("nt", [32, 64])
def test_ln_vt8_matches_ln_v(d: int, c: int, nt: int) -> None:
    _needs_5090()
    from miniworld_engine.kernels.integrations import msa_f8_sm120 as X
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80

    s, n = 70, 208  # 208 = 3 x 64 + 16: a clipped tail tile
    torch.manual_seed(1)
    m = torch.randn(s, n, d, device="cuda").bfloat16()
    lnw = 1.0 + 0.1 * torch.randn(d, device="cuda")
    lnb = 0.1 * torch.randn(d, device="cuda")
    wv = (torch.randn(8 * c, d, device="cuda") * d**-0.5).bfloat16()
    v, _ = sm80.ln_v(m, lnw, lnb, wv, 1e-5, False, 1)  # [8, N, S C] bf16 (the default kernel)
    vinv = torch.tensor([4.0], device="cuda")
    amax = torch.zeros(1, device="cuda")
    vt = X.ln_vt8(m, lnw, lnb, wv, 1e-5, vinv, amax, nt, 2)
    assert vt.shape == (8, s * c, n)
    ref = (v.float() * 4.0).transpose(1, 2)
    got = vt.float()
    # one e4m3 rounding of (nearly) the same value: the default's bf16 v and this kernel's fp32 v differ by a bf16 rounding, so compare within e4m3 steps
    rel = ((got - ref).abs() / ref.abs().clamp_min(2.0**-6)).max().item()
    assert rel < 0.13  # at most one e4m3 step (12.5 % at a power of two): bf16 then e4m3 against fp32 then e4m3 (double rounding)
    assert ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item() < 0.03
    assert amax.item() == pytest.approx(ref.abs().max().item(), rel=0.02)


def test_vt8_y_matches_torch() -> None:
    _needs_5090()
    from miniworld_engine.kernels.integrations import msa_f8_sm120 as X

    s, n = 70, 208  # a clipped token tail
    torch.manual_seed(2)
    y = torch.randn(s * n, 256, device="cuda").bfloat16()
    wv = (torch.randn(256, 256, device="cuda") / 16).bfloat16()
    vinv = torch.tensor([2.0], device="cuda")
    amax = torch.zeros(1, device="cuda")
    vt = X.vt8_y(y, wv, vinv, amax, s, n)
    v = (y.float() @ wv.float().t()) * 2.0  # [S N, 256]
    ref = v.view(s, n, 8, 32).permute(2, 0, 3, 1).reshape(8, s * 32, n)
    got = vt.float()
    assert ((got - ref.clamp(-448, 448)).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item() < 0.03
    assert ((got - ref).abs() / ref.abs().clamp_min(2.0**-6)).max().item() < 0.07
    assert amax.item() == pytest.approx(ref.abs().max().item(), rel=1e-3)


def test_w_quant() -> None:
    _needs_5090()
    from miniworld_engine.kernels.integrations import msa_f8_sm120 as X

    w = torch.rand(8, 208, 208, device="cuda").bfloat16()
    assert torch.equal(X.w_quant(w).float(), (w.float() * 256).to(torch.float8_e4m3fn).float())


@pytest.mark.parametrize("d,c,cz", WIDTHS)
@pytest.mark.parametrize("s,n", [(256, 384), (33, 400)])
def test_error_vs_fp64(P, d: int, c: int, cz: int, s: int, n: int) -> None:
    m, inp = _case(d, c, cz, s, n)
    with torch.no_grad():
        ref = copy.deepcopy(m).double().set_backend("pytorch")
        y64 = ref(*[t.double() if t.is_floating_point() else t for t in inp])
        del ref
    P.ENABLED = False
    y0 = _run(m, inp)
    P.ENABLED = True
    calls = P.STATS["calls"] + P.STATS["wide_calls"]
    y1 = _run(m, inp)
    assert P.STATS["calls"] + P.STATS["wide_calls"] > calls  # the route ran
    assert torch.isfinite(y1).all()
    x = inp[0].double()

    def errs(y):
        dlt = y.double() - y64
        return (dlt.pow(2).mean().sqrt() / (y64 - x).pow(2).mean().sqrt()).item(), (dlt.abs().max() / y64.abs().max()).item()

    (u0, j0), (u1, j1) = errs(y0), errs(y1)
    assert j1 < 8e-3  # the engine tolerance (H100 routes: judge 0.30-0.38 %)
    assert u1 < 0.06 and u1 < 2.5 * u0  # H100: FP8 update rms 2.8-3.6 % against the bf16 2.0-3.1 %
    assert P.overflow_peak() <= 448.0


@pytest.mark.parametrize("d,c,cz", [(64, 32, 128), (256, 32, 512)])
def test_default_untouched_after_route(P, d: int, c: int, cz: int) -> None:
    m, inp = _case(d, c, cz, 64, 256)
    P.ENABLED = False
    y0 = _run(m, inp)
    P.ENABLED = True
    _run(m, inp)
    P.ENABLED = False
    assert torch.equal(_run(m, inp), y0)


def test_safe_preset_and_disable(P) -> None:
    from miniworld_engine import fp8

    P.ENABLED = False
    assert fp8._modules()[fp8.SLOTS.index("pwa")] is P
    fp8.enable("safe")
    try:
        assert P.ENABLED and fp8.status()["pwa"] is True
    finally:
        fp8.disable()
    assert not P.ENABLED


@pytest.mark.parametrize("d,c,cz", [(64, 32, 128), (256, 32, 512)])
def test_sync_guard_redo(P, d: int, c: int, cz: int) -> None:
    m, inp = _case(d, c, cz, 128, 256, seed=3)
    P.ENABLED = False
    y0 = _run(m, inp)
    P.ENABLED, P.GUARD, P.HEADROOM = True, "sync", 2.0**-6  # calibrated 64x too small: v saturates
    retries = P.STATS["retries"]
    y1 = _run(m, inp)
    assert P.STATS["retries"] > retries
    assert ((y1.float() - y0.float()).abs().max() / y0.float().abs().max()).item() < 0.01


@pytest.mark.parametrize("d,c,cz", [(64, 8, 128), (256, 32, 512)])
def test_lazy_guard_through_check_and_graph(P, d: int, c: int, cz: int) -> None:
    from miniworld_engine import fp8

    m, inp = _case(d, c, cz, 128, 256, seed=4)
    P.ENABLED = False
    y0 = _run(m, inp)
    P.ENABLED, P.GUARD, P.HEADROOM = True, "lazy", 2.0**-6
    _run(m, inp)  # calibrates too small, saturates silently
    hit = False
    for _ in range(4):  # each check widens 4x: 64x takes three rounds
        if not fp8.check():
            break
        hit = True
        _run(m, inp)
    assert hit and not fp8.check()
    y_eager = _run(m, inp)
    assert ((y_eager.float() - y0.float()).abs().max() / y0.float().abs().max()).item() < 0.01
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_g = _run(m, inp)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y_g, y_eager)


def test_capture_before_calibration_runs_default(P) -> None:
    m, inp = _case(64, 32, 128, 128, 256, seed=7)
    P.ENABLED = False
    y0 = _run(m, inp)
    P.ENABLED = True
    calls = P.STATS["calls"]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_g = _run(m, inp)
    g.replay()
    torch.cuda.synchronize()
    assert P.STATS["calls"] == calls
    assert torch.equal(y_g, y0)


def test_gates(P) -> None:
    P.MIN_SN_ALL = None
    saved = dict(P.MIN_SN)
    try:
        P.MIN_SN[(64, 32)] = 1 << 18
        assert not P.serves(64, 32, 256, 512)
        assert P.serves(64, 32, 1024, 512)
        assert not P.serves(64, 32, 1024, 520)  # N % 16
    finally:
        P.MIN_SN.clear()
        P.MIN_SN.update(saved)

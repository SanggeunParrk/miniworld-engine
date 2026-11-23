"""RTX 5090: the opt-in e4m3 OuterProductMean product (``kernels/integrations/opm_fp8_sm120``, ``fp8.SM120_ROUTES["opm"]``, docs/fp8.md "RTX 5090"):
the transposing e4m3 cast against torch, error against the float64 module at the level of the H100 ``bound`` route (judge 0.6-1.4 % there, tol
max(PyTorch bf16, 8e-3) up to 1.72 x), ``safe`` / ``engine-fp8`` turn it on and ``disable`` off, the guard (``calib`` with a too-small headroom:
sync redo; lazy ``overflow_peak`` / ``rescale_all`` through ``fp8.check``), graph == eager, a capture before calibration runs the default, gates,
and the default route untouched (bytes) after the route ran."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_5090() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("needs an RTX 5090 (sm_120)")


def _case(cm: int, cz: int, length: int, depth: int, after: bool = False, seed: int = 0):
    from miniworld_engine.modules.msa import OuterProductMean

    m = OuterProductMean(cm, d_hidden=32, d_pair=cz, normalize_before_proj=not after, master_weights=True, backend="cuda").cuda().eval()
    g = torch.Generator(device="cpu").manual_seed(seed + cm + cz)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    torch.manual_seed(seed)
    x = torch.randn(1, depth, length, cm, device="cuda").bfloat16()
    mask = torch.ones(1, depth, length, dtype=torch.bool, device="cuda")
    mask[:, depth - depth // 8 :, : length // 2] = False
    return m, x, mask


@pytest.fixture
def F():
    _needs_5090()
    from miniworld_engine.kernels.integrations import opm_fp8_sm120 as F

    assert F.POLICY is None  # off unless switched on
    saved = (F.GUARD, F.MIN_S, F.HEADROOM)
    F.POLICY, F.MIN_S = "bound", 0
    yield F
    F.POLICY = None
    F.GUARD, F.MIN_S, F.HEADROOM = saved
    F._CACHES.clear()
    F.overflow_peak()


def _run(m, x, mask):
    with torch.no_grad():
        return m(x, mask)


def test_transpose_quant_matches_torch() -> None:
    _needs_5090()
    from miniworld_engine.kernels.integrations import msa_f8_sm120 as X

    for b, r, n in ((1, 1000, 32 * 77), (3, 70, 256 * 2 + 8), (8, 208, 1024)):
        x = (torch.randn(b, r, n, device="cuda") * 30).bfloat16()
        q = torch.tensor([0.25], device="cuda")
        amax = torch.zeros(1, device="cuda")
        y = X.transpose_quant(x, q, amax)
        rp = -(-r // 16) * 16
        assert y.shape == (b, n, rp)
        ref = torch.zeros(b, n, rp, device="cuda")
        ref[:, :, :r] = (x.float() * 0.25).transpose(1, 2).clamp(-448, 448)
        assert torch.equal(y.float(), ref.to(torch.float8_e4m3fn).float())
        assert amax.item() == pytest.approx((x.float().abs().max() * 0.25).item())


@pytest.mark.parametrize(
    "cm,cz,length,depth,after", [(64, 128, 256, 1024, False), (64, 128, 384, 1024, True), (128, 256, 256, 1024, True), (256, 512, 128, 1024, False), (64, 128, 208, 296, False)]
)
def test_error_vs_fp64(F, cm: int, cz: int, length: int, depth: int, after: bool) -> None:
    m, x, mask = _case(cm, cz, length, depth, after)
    with torch.no_grad():
        ref = copy.deepcopy(m).double().set_backend("pytorch")
        y64 = ref(x.double(), mask)
        del ref
    F.POLICY = None
    y0 = _run(m, x, mask)
    F.POLICY = "bound"
    calls = F.GUARD_STATS["calls"]
    y1 = _run(m, x, mask)
    assert F.GUARD_STATS["calls"] > calls  # the route ran
    assert torch.isfinite(y1).all()

    def judge(y):
        return ((y.double() - y64).abs().max() / y64.abs().max()).item()

    def rms(y):
        return ((y.double() - y64).pow(2).mean().sqrt() / y64.pow(2).mean().sqrt()).item()

    # H100 bound route: judge 0.6-1.4 %, rms 0.24-3.5 % (AF3 order: the bias is divided by n too, a small y64)
    assert judge(y1) < 0.025 and judge(y1) < 8 * max(judge(y0), 2e-3)
    assert rms(y1) < 0.05
    assert F.overflow_peak() <= 448.0


def test_default_untouched_after_route(F) -> None:
    m, x, mask = _case(64, 128, 256, 1024)
    F.POLICY = None
    y0 = _run(m, x, mask)
    F.POLICY = "bound"
    _run(m, x, mask)
    F.POLICY = None
    assert torch.equal(_run(m, x, mask), y0)


def test_safe_preset_and_disable(F) -> None:
    from miniworld_engine import fp8

    F.POLICY = None
    assert fp8._modules()[fp8.SLOTS.index("opm")] is F
    fp8.enable("safe")
    try:
        assert F.POLICY == "bound" and fp8.status()["opm"] == "bound"
    finally:
        fp8.disable()
    assert F.POLICY is None


def test_sync_guard_redo(F) -> None:
    m, x, mask = _case(64, 128, 256, 1024)
    F.POLICY = None
    y0 = _run(m, x, mask)
    F.POLICY, F.GUARD, F.HEADROOM = "calib", "sync", 2.0**-6  # the calibrated scale is 64x too small: the cast saturates
    retries = F.GUARD_STATS["retries"]
    y1 = _run(m, x, mask)
    assert F.GUARD_STATS["retries"] > retries
    assert ((y1.float() - y0.float()).abs().max() / y0.float().abs().max()).item() < 0.03


def test_lazy_guard_through_check_and_graph(F) -> None:
    from miniworld_engine import fp8

    m, x, mask = _case(64, 128, 256, 1024)
    F.POLICY = None
    y0 = _run(m, x, mask)
    F.POLICY, F.GUARD, F.HEADROOM = "calib", "lazy", 2.0**-6
    _run(m, x, mask)  # calibrates (too small) and saturates silently
    assert fp8.check()  # read the guard, raised the scales
    assert not fp8.check()
    y_eager = _run(m, x, mask)
    assert ((y_eager.float() - y0.float()).abs().max() / y0.float().abs().max()).item() < 0.03
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_g = _run(m, x, mask)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y_g, y_eager)


def test_capture_before_calibration_runs_default(F) -> None:
    m, x, mask = _case(64, 128, 256, 1024, seed=5)
    F.POLICY = None
    y0 = _run(m, x, mask)
    F.POLICY = "calib"
    fb = F.GUARD_STATS["capture_fallbacks"]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_g = _run(m, x, mask)
    g.replay()
    torch.cuda.synchronize()
    assert F.GUARD_STATS["capture_fallbacks"] > fb
    assert torch.equal(y_g, y0)


@torch.no_grad()
def test_gate(F) -> None:
    F.MIN_S = None
    assert not F.serves(512, 256)
    assert F.serves(1024, 128)
    F.MIN_S = 0
    assert F.serves(16, 16)
    with torch.enable_grad():
        assert not F.serves(4096, 256)  # inference only

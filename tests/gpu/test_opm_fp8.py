"""EXPERIMENTAL OuterProductMean FP8 route (``kernels.integrations.opm_fp8``, opt-in, H100): e4m3 A2 / BT from the wide prologue's FP8 build,
cuBLASLt e4m3 grouped product, the default epilogue.

- off (``POLICY`` None): the route is never taken and no FP8 extension is built;
- the e4m3 operands are the default bf16 A2 / BT within one e4m3 rounding (same mask, same bit rows), masked rows exactly zero;
- the module output against the float64 reference: inside the FP8 error budget (rms < 4 %, max < 8 % of max|y64|; the bf16 default ~0.3 %),
  finite, for both scale policies, both normalisation orders, the row-chunked product and the pair residual;
- ``bound`` scales never saturate; ``calib`` with a forced too-small scale: the sync guard redoes the call, the lazy guard reports the peak and
  ``rescale_all`` raises the scales in place;
- a CUDA-graph capture of a calibrated module replays to the eager output bit for bit."""

import copy

import pytest
import torch

from miniworld_engine.kernels.integrations import opm_fp8 as F
from miniworld_engine.kernels.integrations import opm_train
from miniworld_engine.measure import config as C
from miniworld_engine.modules import OuterProductMean

pytestmark = [pytest.mark.gpu]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.fixture
def fp8(monkeypatch):
    def set_(policy, guard="sync", fast=False):
        monkeypatch.setattr(F, "POLICY", policy)
        monkeypatch.setattr(F, "GUARD", guard)
        monkeypatch.setattr(F, "FAST_ACCUM", fast)
        monkeypatch.setattr(F, "MIN_S", 0)
        F._CACHES.clear()
        F.overflow_peak()

    yield set_
    F._CACHES.clear()


def _case(length, depth, cm=64, cz=128, after=False, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed + length + depth)
    base = OuterProductMean(cm, cz, 32, normalize_before_proj=not after).cuda()
    C.randomize(base, gen)
    inp = base.make_inputs(length=length, samples=1, depth=depth, dtype=torch.bfloat16, device="cuda", generator=gen)
    msa, mask = inp["msa"], inp["mask"]
    mask[:, depth // 3 : depth // 3 + 37] = False
    return base, msa, mask


def _run(module, msa, mask, residual=None):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=msa.dtype != torch.float64):
        return module(msa, mask, residual=residual)


def _gold(base, msa, mask, residual=None):
    g = copy.deepcopy(base).double().set_backend("pytorch")
    return _run(g, msa.double(), mask, None if residual is None else residual.double())


def _err(y, gold):
    d = y.double() - gold
    return float(d.pow(2).mean().sqrt() / gold.pow(2).mean().sqrt()), float(d.abs().max() / gold.abs().max())


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
def test_off_by_default(fp8, monkeypatch) -> None:
    fp8(None)
    base, msa, mask = _case(128, 512)
    calls = []
    monkeypatch.setattr(F, "inference", lambda *a, **k: calls.append(1))
    _run(copy.deepcopy(base).set_backend("cuda"), msa, mask)
    assert not calls and not any(k.endswith("_fp8") or "_fp8_" in k for k in opm_train._EXT)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
@pytest.mark.parametrize("cm", [64, 128])
def test_operands_are_one_e4m3_rounding(cm) -> None:
    torch.manual_seed(0)
    cz = 128 if cm == 64 else 256
    s_, n = 1024, 128
    m = torch.randn(s_, n, cm, device="cuda", dtype=torch.bfloat16)
    mk = torch.rand(s_, n, device="cuda") > 0.1
    gam, bet = 1 + 0.1 * torch.randn(cm, device="cuda"), 0.1 * torch.randn(cm, device="cuda")
    wa, wb = ((torch.randn(32, cm, device="cuda") * cm**-0.5).to(torch.bfloat16) for _ in range(2))
    e16 = opm_train._ext(None, native=True, tma=False, cm=cm, cz=cz)
    e8 = opm_train._ext(None, native=True, tma=False, cm=cm, cz=cz, fp8=True)
    a2, bt, _, mk16, bits = e16.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False)
    s = F.bound_scales(gam, bet, wa, wb).cuda()
    amax = torch.zeros(2, device="cuda")
    a8, b8, _, mk8, bits8 = e8.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False, f8q=(1 / s).contiguous(), f8amax=amax)
    assert torch.equal(mk16, mk8) and torch.equal(bits, bits8)
    assert float(amax.max()) <= F.E4M3_MAX
    for q, ref, sc in ((a8, a2, s[0]), (b8, bt, s[1])):
        deq, ref = q.float() * sc, ref.float()
        assert torch.all(deq[ref == 0] == 0)
        # e4m3: 3 mantissa bits, |x - q(x)| <= 2^-4 |x| for normal values (+ the bf16 rounding the default stores); subnormal floor
        assert torch.all((deq - ref).abs() <= ref.abs() * (2**-4 + 2**-8) + sc * 2**-9)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
@pytest.mark.parametrize("policy", ["bound", "calib"])
@pytest.mark.parametrize(
    ("length", "depth", "cm", "cz", "after", "residual", "chunk_rows"),
    [
        (256, 2048, 64, 128, False, False, 0),
        (192, 4096, 64, 128, True, True, 0),
        (256, 8192, 64, 128, True, False, 64),
        (128, 2048, 128, 256, True, False, 0),
        (128, 1024, 256, 512, False, False, 0),
    ],
)
def test_module_error_budget(fp8, monkeypatch, policy, length, depth, cm, cz, after, residual, chunk_rows) -> None:
    fp8(policy)
    base, msa, mask = _case(length, depth, cm, cz, after)
    res = torch.randn((1, length, length, cz), device="cuda").to(torch.bfloat16) if residual else None
    if chunk_rows:
        monkeypatch.setattr(F, "_O_CHUNK_BYTES", chunk_rows * length * 32 * 32 * 2)
    gold = _gold(base, msa, mask, res)
    mod = copy.deepcopy(base).set_backend("cuda")
    y = _run(mod, msa, mask, res)
    if policy == "calib":
        assert F._CACHES, "the first call calibrates"
        y = _run(mod, msa, mask, res)
    assert torch.isfinite(y).all()
    fp8(None)
    y_def = _run(copy.deepcopy(base).set_backend("cuda"), msa, mask, res)
    upd = gold - (res.double() if res is not None else 0)
    rms, mx = _err(y.double() - (res.double() if res is not None else 0), upd)
    rms_d, _ = _err(y_def.double() - (res.double() if res is not None else 0), upd)
    assert rms < 0.04 and mx < 0.08, (rms, mx)
    assert rms > rms_d  # (the FP8 route did run)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
def test_guard(fp8) -> None:
    base, msa, mask = _case(128, 1024)
    fp8("calib", guard="sync")
    mod = copy.deepcopy(base).set_backend("cuda")
    y0 = _run(mod, msa, mask)
    (cache,) = F._CACHES.values()
    ok = cache["s"].clone()
    F._scale_up(cache, 2.0**-8)  # far too small: the cast would saturate
    y1 = _run(mod, msa, mask)
    assert F.GUARD_STATS["retries"] >= 1 and torch.all(cache["s"] >= ok * 2.0**-8 * 2)
    assert _err(y1.double(), y0.double())[0] < 0.05
    fp8("calib", guard="lazy")
    mod2 = copy.deepcopy(base).set_backend("cuda")
    _run(mod2, msa, mask)
    (cache,) = F._CACHES.values()
    F._scale_up(cache, 2.0**-8)
    F.overflow_peak()
    _run(mod2, msa, mask)
    peak = F.overflow_peak()
    assert peak > F.E4M3_MAX
    F.rescale_all(peak)
    _run(mod2, msa, mask)
    assert F.overflow_peak() <= F.E4M3_MAX


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
@pytest.mark.parametrize("policy", ["bound", "calib"])
def test_graph_replay(fp8, policy) -> None:
    fp8(policy, guard="lazy")
    base, msa, mask = _case(256, 2048)
    mod = copy.deepcopy(base).set_backend("cuda")
    eager = _run(mod, msa, mask)
    eager = _run(mod, msa, mask)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run(mod, msa, mask)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _run(mod, msa, mask)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    assert F.overflow_peak() <= F.E4M3_MAX

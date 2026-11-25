"""EXPERIMENTAL OuterProductMean FP8 TRAINING route (``kernels.integrations.opm_fp8_train``, opt-in, H100): e4m3 A2 / BT + cuBLASLt e4m3
grouped product in the forward, dO (e5m2 / e4m3) in both layouts from one dgrad launch and the two backward products on cuBLASLt FP8.

- off (``MODE`` None): the route is never taken, the default gradients are unchanged after an FP8 step (no leaked state);
- the dO^T plane is the dO plane transposed, bit for bit; dequantised dO is the default bf16 dO within one FP8 rounding;
- every gradient and the output against float64 PyTorch: inside the FP8 budget, finite, no saturation (bound scales);
- deterministic: two steps give the same bits."""

import copy

import pytest
import torch

from foldjet_engine.kernels.integrations import opm_fp8_train as F8
from foldjet_engine.kernels.integrations import opm_train
from foldjet_engine.measure import config as C
from foldjet_engine.modules import OuterProductMean

pytestmark = [pytest.mark.gpu]

NAMES = ("ln_msa.weight", "ln_msa.bias", "to_left.weight", "to_right.weight", "to_out.weight", "to_out.bias")


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.fixture
def mode(monkeypatch):
    def set_(m):
        monkeypatch.setattr(F8, "MODE", m)
        monkeypatch.setattr(F8, "MIN_S", 0)
        monkeypatch.setattr(F8, "MIN_L", 0)
        monkeypatch.setattr(F8, "MAX_CZ", 0)
        F8.reset()
        F8.peaks()

    yield set_


def _case(length, depth, cm=64, cz=128, after=False, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed + length + depth)
    base = OuterProductMean(cm, cz, 32, normalize_before_proj=not after).cuda()
    C.randomize(base, gen)
    C.cast(base, "master")  # in place (Module.float)
    inp = base.make_inputs(length=length, samples=1, depth=depth, dtype=torch.bfloat16, device="cuda", generator=gen)
    msa, mask = inp["msa"], inp["mask"]
    mask[:, depth - depth // 8 :, length // 2 :] = False
    gz = torch.randn((1, length, length, cz), device="cuda", generator=gen).to(torch.bfloat16)
    return base, msa, mask, gz


def _grads(mod, msa, mask, gz, gold=False):
    for p in mod.parameters():
        p.grad = None
    x = msa.detach().clone().requires_grad_(True)
    if gold:
        y = mod(x, mask)
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = mod(x, mask)
    y.backward(gz.to(y.dtype))
    p = dict(mod.named_parameters())
    return {"y": y.detach().double(), "msa": x.grad.double(), **{k: p[k].grad.double() for k in NAMES}}


def _gold(base, msa, mask, gz):
    return _grads(copy.deepcopy(base).double().set_backend("pytorch"), msa.double(), mask, gz.double(), gold=True)


def _err(a, g):
    d = a - g
    return float(d.pow(2).mean().sqrt() / g.pow(2).mean().sqrt()), float(d.abs().max() / g.abs().max())


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
def test_off_by_default_and_no_leak(mode) -> None:
    mode(None)
    base, msa, mask, gz = _case(128, 1024)
    mod = copy.deepcopy(base).set_backend("cuda")
    c0 = F8.STATS["calls"]
    g0 = _grads(mod, msa, mask, gz)
    assert F8.STATS["calls"] == c0
    mode("e5m2")
    _grads(mod, msa, mask, gz)
    assert F8.STATS["calls"] == c0 + 1
    mode(None)
    g1 = _grads(mod, msa, mask, gz)
    assert all(torch.equal(g0[k], g1[k]) for k in g0)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
@pytest.mark.parametrize("fmt", [0, 1])
@pytest.mark.parametrize("single", [False, True])
def test_dgrad_planes(fmt, single) -> None:
    n, cz = 128, 128
    ext = opm_train._ext(torch.device("cuda"), native=True, tma=False, cm=64, cz=cz, fp8=True)
    ext0 = opm_train._ext(torch.device("cuda"), native=True, tma=False, cm=64, cz=cz)
    gen = torch.Generator(device="cuda").manual_seed(3)
    dz = torch.randn((n, n, cz), device="cuda", generator=gen).to(torch.bfloat16)
    norm = torch.randint(1, 50, (n, n), device="cuda", generator=gen).float()
    norm = torch.maximum(norm, norm.t()).contiguous()
    wo = (torch.randn((cz, 1024), device="cuda", generator=gen) / 32).to(torch.bfloat16)
    bw = wo.t().contiguous()
    bw2 = torch.cat([bw, bw.view(32, 32, cz).transpose(0, 1).reshape(1024, cz)]).contiguous()
    s = torch.tensor([2.0**-8], device="cuda")
    amax = torch.zeros(1, device="cuda")
    d8, d8t, dzp, dbo = ext.opm_dgrad_f8(dz, norm, bw2, n, n, 0, 1.0 / s, amax, fmt, single)
    dO, dzp0, dbo0 = ext0.opm_dgrad(dz, norm, bw, n, n, 0)
    assert torch.equal(d8t.view(torch.uint8), d8.view(torch.uint8).t())
    assert torch.equal(dzp, dzp0) and torch.equal(dbo, dbo0)
    deq = d8.float() * s
    ref = dO.float()
    rel = float((deq - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
    assert rel < (0.08 if fmt else 0.04), rel
    assert 0 < float(amax) <= (57344.0 if fmt else 448.0)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
@pytest.mark.parametrize(
    "length,depth,cm,cz,after,fmt",
    [
        (128, 1024, 64, 128, True, "e5m2"),
        (128, 1024, 64, 128, True, "e4m3"),
        (192, 2048, 64, 128, False, "e5m2"),
        (128, 1024, 128, 256, True, "e5m2"),
        (128, 512, 256, 512, False, "e5m2"),
    ],
)
def test_gradients_against_fp64(mode, length, depth, cm, cz, after, fmt) -> None:
    base, msa, mask, gz = _case(length, depth, cm, cz, after)
    gold = _gold(base, msa, mask, gz)
    mod = copy.deepcopy(base).set_backend("cuda")
    mode(fmt)
    c0 = F8.STATS["calls"]
    _grads(mod, msa, mask, gz)  # first call: the dO scale seeds from the bound; the next ones use the amax history (delayed)
    g = _grads(mod, msa, mask, gz)
    assert F8.STATS["calls"] == c0 + 2
    peaks = F8.peaks()
    assert peaks[0] <= 448 and peaks[1] <= 448 and peaks[2] <= (57344 if fmt == "e5m2" else 448)
    for k, v in g.items():
        assert torch.isfinite(v).all(), k
        rms, mx = _err(v, gold[k])
        # measured (docs/fp8_training.md): e4m3 dO ~3.5 % rms, e5m2 ~6 % (one rounding of a 3- / 2-bit mantissa); the bf16 default ~0.5 %
        rb, mb = (0.05, 0.10) if fmt == "e4m3" else (0.09, 0.15)
        assert rms < rb and mx < mb, (k, rms, mx)
    g2 = _grads(mod, msa, mask, gz)
    assert all(torch.equal(g[k], g2[k]) for k in g)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
@pytest.mark.parametrize("cm,bias", [(64, False), (128, True)])
def test_weight_scales_match_bound_scales(cm, bias) -> None:
    from foldjet_engine.kernels.integrations import opm_fp8

    cz = 128 if cm == 64 else 256
    ext = opm_train._ext(torch.device("cuda"), native=True, tma=False, cm=cm, cz=cz, fp8=True)
    gen = torch.Generator(device="cuda").manual_seed(cm)
    lnw = 1 + 0.1 * torch.randn(cm, device="cuda", generator=gen)
    lnb = 0.2 * torch.randn(cm, device="cuda", generator=gen)
    wa, wb = (torch.randn((32, cm), device="cuda", generator=gen) / cm**0.5 for _ in range(2))
    pb = 0.1 * torch.randn(64, device="cuda", generator=gen) if bias else None
    sc = ext.opm_f8_wscale(lnw, lnb, wa.to(torch.bfloat16), wb.to(torch.bfloat16), pb)
    ref = opm_fp8.bound_scales(lnw, lnb, wa, wb, pb)
    assert torch.equal(sc[:2], ref) and torch.equal(sc[2:], 1.0 / ref)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
def test_prologue_transposed_planes() -> None:
    """Round 2: the prologue's [s, (i,c)] copies are the byte transposes of its [(i,c), s] planes, and its other outputs do not change."""
    ext = opm_train._ext(torch.device("cuda"), native=True, tma=False, cm=64, cz=128, fp8=True)
    gen = torch.Generator(device="cuda").manual_seed(11)
    s, n = 512, 128
    m = torch.randn((s, n, 64), device="cuda", generator=gen).to(torch.bfloat16)
    mk = torch.rand((s, n), device="cuda", generator=gen) > 0.1
    gam, bet = 1 + 0.1 * torch.randn(64, device="cuda", generator=gen), 0.1 * torch.randn(64, device="cuda", generator=gen)
    wa, wb = ((torch.randn((32, 64), device="cuda", generator=gen) / 8).to(torch.bfloat16) for _ in range(2))
    q = torch.full((2,), 64.0, device="cuda")
    am0 = torch.zeros(3, device="cuda")
    am1 = torch.zeros(3, device="cuda")
    r0 = ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, True, None, True, f8q=q, f8amax=am0)
    r1 = ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, True, None, True, f8q=q, f8amax=am1, f8t=True)
    assert len(r0) == 5 and len(r1) == 7
    for a, b in zip(r0, r1[:5], strict=True):
        assert torch.equal(a.view(torch.uint8) if a.element_size() == 1 else a, b.view(torch.uint8) if b.element_size() == 1 else b)
    assert torch.equal(r1[5].view(torch.uint8), r1[0].view(torch.uint8).t())
    assert torch.equal(r1[6].view(torch.uint8), r1[1].view(torch.uint8).t())


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
def test_gate_and_delayed_scale(mode, monkeypatch) -> None:
    """Round 2: the shape gate (MIN_L opt-in, d_pair <= 384 by default); the delayed dO scale seeds from the bound, then tracks the amax."""
    msa = torch.zeros((1, 1024, 128, 64), device="cuda", dtype=torch.bfloat16)
    mode("e4m3")
    monkeypatch.setattr(F8, "MIN_L", 256)  # (opt-in gate: the default is 0; round 2 serves L 128)
    monkeypatch.setattr(F8, "MAX_CZ", 256)
    assert not F8.serves(msa, 64, 128)  # L 128
    assert F8.serves(torch.zeros((1, 1024, 256, 64), device="cuda", dtype=torch.bfloat16), 64, 128)
    assert not F8.serves(torch.zeros((1, 1024, 256, 256), device="cuda", dtype=torch.bfloat16), 256, 512)  # d_pair 512: 1.03x in round 2
    monkeypatch.setattr(F8, "MIN_L", 0)  # round 3: the per-d_pair length table (MAX_CZ 384 = the default gate)
    monkeypatch.setattr(F8, "MAX_CZ", 384)
    z = lambda n, c: torch.zeros((1, 1024, n, c), device="cuda", dtype=torch.bfloat16)
    assert F8.serves(z(128, 64), 64, 128) and not F8.serves(z(128, 128), 128, 256) and F8.serves(z(256, 128), 128, 256)
    assert not F8.serves(z(256, 128), 128, 384) and F8.serves(z(384, 128), 128, 384)
    monkeypatch.setattr(F8, "MAX_CZ", 256)
    base, msa, mask, gz = _case(128, 1024)
    mod = copy.deepcopy(base).set_backend("cuda")
    monkeypatch.setattr(F8, "MIN_L", 0)
    _grads(mod, msa, mask, gz)
    site = next(iter(F8._SITES.values()))
    assert site.seeded and float(site.st[3]) == 1.0
    sd0 = float(site.sd)
    _grads(mod, msa, mask, gz)
    _grads(mod, msa, mask, gz)
    assert float(site.st[3]) == 3.0 and float(site.st[4]) == 0.0
    peak = float(site.amax)
    assert 0 < peak <= 448
    assert float(site.sd) <= sd0  # the measured amax is below the bound's (4-6x loose), so the scale never grows


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
@pytest.mark.parametrize("ostore", ["recompute", "fp8"])
def test_o_storage_modes(mode, monkeypatch, ostore) -> None:
    """Round 4: O kept for dW_o as 'recompute' (the FP8 product redone: gradients bit-identical to keeping bf16 O, far fewer saved bytes)
    or 'fp8' (e4m3 x 2^k: only dW_out moves, within 2x of the bf16-O error)."""
    base, msa, mask, gz = _case(192, 1024, after=True)
    mod = copy.deepcopy(base).set_backend("cuda")
    gold = _gold(base, msa, mask, gz)
    mode("e4m3")
    monkeypatch.setattr(F8, "OSTORE", "bf16")
    _grads(mod, msa, mask, gz)  # seeds the delayed scale
    ref = _grads(mod, msa, mask, gz)
    monkeypatch.setattr(F8, "OSTORE", ostore)
    got = _grads(mod, msa, mask, gz)
    for k, v in got.items():
        assert torch.isfinite(v).all()
        if ostore == "recompute" or k != "to_out.weight":
            assert torch.equal(v, ref[k]), k
        else:
            assert _err(v, gold[k])[0] < 2 * _err(ref[k], gold[k])[0] + 1e-3

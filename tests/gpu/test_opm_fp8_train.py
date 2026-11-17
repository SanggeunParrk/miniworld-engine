"""EXPERIMENTAL OuterProductMean FP8 TRAINING route (``kernels.integrations.opm_fp8_train``, opt-in, H100): e4m3 A2 / BT + cuBLASLt e4m3
grouped product in the forward, dO (e5m2 / e4m3) in both layouts from one dgrad launch and the two backward products on cuBLASLt FP8.

- off (``MODE`` None): the route is never taken, the default gradients are unchanged after an FP8 step (no leaked state);
- the dO^T plane is the dO plane transposed, bit for bit; dequantised dO is the default bf16 dO within one FP8 rounding;
- every gradient and the output against float64 PyTorch: inside the FP8 budget, finite, no saturation (bound scales);
- deterministic: two steps give the same bits."""

import copy

import pytest
import torch

from miniworld_engine.kernels.integrations import opm_fp8_train as F8
from miniworld_engine.kernels.integrations import opm_train
from miniworld_engine.measure import config as C
from miniworld_engine.modules import OuterProductMean

pytestmark = [pytest.mark.gpu]

NAMES = ("ln_msa.weight", "ln_msa.bias", "to_left.weight", "to_right.weight", "to_out.weight", "to_out.bias")


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.fixture
def mode(monkeypatch):
    def set_(m):
        monkeypatch.setattr(F8, "MODE", m)
        monkeypatch.setattr(F8, "MIN_S", 0)
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
def test_dgrad_planes(fmt) -> None:
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
    d8, d8t, dzp, dbo = ext.opm_dgrad_f8(dz, norm, bw2, n, n, 0, 1.0 / s, amax, fmt)
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
    g = _grads(mod, msa, mask, gz)
    assert F8.STATS["calls"] == c0 + 1
    peaks = F8.peaks()
    assert peaks[0] <= 448 and peaks[1] <= 448 and peaks[2] <= (57344 if fmt == "e5m2" else 448)
    for k, v in g.items():
        assert torch.isfinite(v).all(), k
        rms, mx = _err(v, gold[k])
        assert rms < 0.06 and mx < 0.12, (k, rms, mx)
    g2 = _grads(mod, msa, mask, gz)
    assert all(torch.equal(g[k], g2[k]) for k in g)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
@pytest.mark.parametrize("cm,bias", [(64, False), (128, True)])
def test_weight_scales_match_bound_scales(cm, bias) -> None:
    from miniworld_engine.kernels.integrations import opm_fp8

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

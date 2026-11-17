"""EXPERIMENTAL opt-in PWA FP8 route (``kernels.integrations.pwa_fp8``, H100): e4m3 w and v in the key contraction (cuBLASLt), v written
K-major as e4m3 by ``ln_vt8``. Off by default: the default route does not change; on: finite, within the engine tolerance at these shapes,
graph replay = eager, the saturation guard fires and ``rescale_all`` clears it."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import MSAPairWeightedAveraging

pytestmark = [pytest.mark.gpu]

_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.fixture
def f8():
    from miniworld_engine.kernels.integrations import pwa_fp8 as F8

    saved = (F8.ENABLED, F8.GUARD, F8.O8, F8.MIN_N, F8.MIN_S)
    F8.GUARD, F8.MIN_N, F8.MIN_S = "lazy", 0, 0
    yield F8
    F8.ENABLED, F8.GUARD, F8.O8, F8.MIN_N, F8.MIN_S = saved
    F8.take_overflow()


def _case(d_pair: int, n: int, s: int):
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = MSAPairWeightedAveraging(64, d_pair, 8, d_hidden=32).cuda().eval()
    C.randomize(base, gen)
    inp = base.make_inputs(length=n, samples=1, depth=s, dtype=torch.bfloat16, device="cuda", generator=gen)
    return base, inp


@pytest.mark.skipif(not _H100, reason="H100")
def test_ln_vt8_matches_ln_vg(f8) -> None:
    from miniworld_engine.kernels.integrations import pwa_train as P

    torch.manual_seed(0)
    s, n = 96, 256
    m = torch.randn(s, n, 64, device="cuda").bfloat16()
    lw, lb = 1 + 0.1 * torch.randn(64, device="cuda"), 0.1 * torch.randn(64, device="cuda")
    wv = (torch.randn(256, 64, device="cuda") / 8).bfloat16()
    v, y = P._k(True)["lnvg"].ln_vg(m, lw, lb, wv, 1e-5, 2, 3, 1)
    amax = torch.zeros(1, device="cuda")
    vt, y8 = f8._ext().ln_vt8(m, lw, lb, wv, 1e-5, torch.full((1,), 8.0, device="cuda"), amax, *f8.LNVT8_CFG)
    assert torch.equal(y, y8)
    vd = vt.float().transpose(1, 2) / 8.0
    ref = v.float()
    assert float((vd - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()) < 0.04
    assert abs(float(amax) - 8 * float(ref.abs().max())) <= 0.02 * float(amax)
    w = torch.softmax(torch.randn(8, n, n, device="cuda"), -1).bfloat16()
    assert torch.equal(f8._ext().w_quant(w).view(torch.uint8), (w.float() * 256).to(torch.float8_e4m3fn).view(torch.uint8))


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize("o8", [False, True])
@pytest.mark.parametrize(("d_pair", "n", "s"), [(128, 512, 64), (128, 256, 32), (256, 384, 32)])
def test_route_accuracy_and_default_unchanged(f8, o8: bool, d_pair: int, n: int, s: int) -> None:
    base, inp = _case(d_pair, n, s)
    kern = copy.deepcopy(base).set_backend("cuda")
    with torch.no_grad():
        gold = copy.deepcopy(base).double()(inp["msa"].double(), inp["pair"].double(), inp["mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = base(**inp).double()
        before = kern(**inp)
        f8.ENABLED, f8.O8 = True, o8
        got = kern(**inp)
        again = kern(**inp)
        f8.ENABLED = False
        after = kern(**inp)
    assert not f8.take_overflow()
    assert torch.equal(before, after)  # off again: the default route, unchanged
    assert torch.isfinite(got).all() and torch.equal(got, again)
    scale = gold.abs().max()
    err_ref = float((ref - gold).abs().max() / scale)
    err = float((got.double() - gold).abs().max() / scale)
    assert err <= max(err_ref, 8e-3), f"fp8 err {err:.3e}, pytorch {err_ref:.3e}"


@pytest.mark.skipif(not _H100, reason="H100")
def test_graph_replay_and_guard(f8) -> None:
    base, inp = _case(128, 512, 64)
    kern = copy.deepcopy(base).set_backend("cuda")
    f8.ENABLED = True
    with torch.no_grad():
        eager = kern(**inp)  # calibrates
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            kern(**inp)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = kern(**inp)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
        assert not f8.take_overflow()
        f8.rescale_all(2.0**-8)  # scale far too small: the e4m3 cast saturates, the guard reports it
        g.replay()
        assert f8.take_overflow()
        f8.rescale_all(2.0**8)  # back: the captured graph follows the in-place scale
        g.replay()
        torch.cuda.synchronize()
        assert not f8.take_overflow()
        assert torch.equal(out, eager)

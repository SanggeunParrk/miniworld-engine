"""EXPERIMENTAL opt-in PWA FP8 route at the ``pwa_wide`` widths (``kernels.integrations.pwa_fp8.forward_wide``, H100): 8 heads x 8 / 16
(AF3 / OpenFold3 / Protenix-v1 64 x 8, Protenix-v2 / OpenDDE 128 x 8, ESMFold2 128 x 16). ``ln_vt8`` built at the width writes v as e4m3
K-major (no y; N a multiple of 16), cuBLASLt e4m3 contraction, the default ``gate_out_w`` pass. Off by default: the default route does not
change; on: finite, within the engine tolerance, graph replay = eager, the saturation guard fires and ``rescale_all`` clears it."""

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

    saved = (F8.ENABLED, F8.GUARD, F8.WIDE_MIN_N, F8.WIDE_MIN_SN, F8.BIG_N_MULT, F8.BIG_MIN_SN)
    F8.GUARD, F8.WIDE_MIN_N, F8.WIDE_MIN_SN, F8.BIG_N_MULT, F8.BIG_MIN_SN = "lazy", 0, 0, 16, 0
    yield F8
    F8.ENABLED, F8.GUARD, F8.WIDE_MIN_N, F8.WIDE_MIN_SN, F8.BIG_N_MULT, F8.BIG_MIN_SN = saved
    F8.take_overflow()


def _case(d_msa: int, d_pair: int, c: int, n: int, s: int):
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = MSAPairWeightedAveraging(d_msa, d_pair, 8, d_hidden=c).cuda().eval()
    C.randomize(base, gen)
    inp = base.make_inputs(length=n, samples=1, depth=s, dtype=torch.bfloat16, device="cuda", generator=gen)
    return base, inp


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize(("d_msa", "c"), [(64, 8), (128, 8), (128, 16), (256, 32)])
def test_ln_vt8_wide_matches_ln_vg_w(f8, d_msa: int, c: int) -> None:
    from miniworld_engine.kernels.integrations import pwa_wide as PW

    torch.manual_seed(0)
    s, n = 70, 208  # S not a multiple of 64, N a multiple of 16 but not of 64 (the clipped token tail)
    m = torch.randn(s, n, d_msa, device="cuda").bfloat16()
    lw, lb = 1 + 0.1 * torch.randn(d_msa, device="cuda"), 0.1 * torch.randn(d_msa, device="cuda")
    wv = (torch.randn(8 * c, d_msa, device="cuda") * d_msa**-0.5).bfloat16()
    v, _, _ = PW._value(d_msa, c).ln_vg_w(m, lw, lb, wv, 1e-5, False)  # head-major [H, N, S*C]
    amax = torch.zeros(1, device="cuda")
    vt, y = f8._ext(d_msa, c).ln_vt8(m, lw, lb, wv, 1e-5, torch.full((1,), 8.0, device="cuda"), amax, *f8.wide_cfg(d_msa, c), False)
    assert y.numel() == 0 and vt.shape == (8, s * c, n)
    vd = vt.float().transpose(1, 2) / 8.0
    ref = v.float()
    assert float((vd - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()) < 0.04
    assert abs(float(amax) - 8 * float(ref.abs().max())) <= 0.02 * float(amax)


@pytest.mark.skipif(not _H100, reason="H100")
@pytest.mark.parametrize(
    ("d_msa", "d_pair", "c", "n", "s"),
    [
        (64, 128, 8, 400, 96),
        (64, 128, 8, 777, 33),
        (128, 256, 8, 384, 64),
        (128, 256, 16, 272, 64),
        (128, 384, 8, 256, 32),
        (256, 512, 32, 384, 64),  # BIG, N % 64 == 0: y kept by ln_vt8
        (256, 512, 32, 400, 33),  # BIG, N % 16 only: gate_w normalises m again
    ],
)
def test_route_accuracy_and_default_unchanged(f8, d_msa: int, d_pair: int, c: int, n: int, s: int) -> None:
    base, inp = _case(d_msa, d_pair, c, n, s)
    kern = copy.deepcopy(base).set_backend("cuda")
    with torch.no_grad():
        gold = copy.deepcopy(base).double()(inp["msa"].double(), inp["pair"].double(), inp["mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = base(**inp).double()
        before = kern(**inp)
        f8.ENABLED = True
        n_vs = len(f8._VS)
        got = kern(**inp)
        assert len(f8._VS) == n_vs + 1  # the route ran (it calibrated this module's v scale)
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
    base, inp = _case(64, 128, 8, 384, 128)
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
    f8.ENABLED = False


def test_gates_wide() -> None:
    from miniworld_engine.kernels.integrations import pwa_fp8 as F8

    saved = (F8.ENABLED, F8.WIDE_MIN_N, F8.WIDE_MIN_SN, F8.BIG_N_MULT, F8.BIG_MIN_SN)
    try:
        F8.ENABLED, F8.WIDE_MIN_N, F8.WIDE_MIN_SN, F8.BIG_N_MULT, F8.BIG_MIN_SN = True, 256, 3 << 16, 64, 1 << 14
        assert F8.serves_wide(64, 8, 512, 384) and F8.serves_wide(64, 8, 1024, 256) and F8.serves_wide(128, 16, 1024, 784)
        assert F8.serves_wide(64, 8, 256, 768) and not F8.serves_wide(64, 8, 128, 768)  # S * N >= 3 * 2^16
        assert not F8.serves_wide(64, 8, 512, 256) and not F8.serves_wide(64, 8, 256, 384)
        assert not F8.serves_wide(64, 8, 4096, 192)  # short
        assert not F8.serves_wide(64, 8, 1024, 392)  # N % 16 (the caller pads)
        assert F8.serves_wide(256, 32, 1024, 768) and F8.serves_wide(256, 32, 64, 256)  # the BIG build (IntelliFold2): own gates
        assert not F8.serves_wide(256, 32, 1024, 400) and not F8.serves_wide(256, 32, 32, 256)  # N % 64, S * N >= 2^14
        F8.ENABLED = False
        assert not F8.serves_wide(64, 8, 4096, 768)
    finally:
        F8.ENABLED, F8.WIDE_MIN_N, F8.WIDE_MIN_SN, F8.BIG_N_MULT, F8.BIG_MIN_SN = saved


@pytest.mark.skipif(not _H100, reason="H100")
def test_ln_vt8_big_keeps_y(f8) -> None:
    """BIG (256 x 8 x 32): with store_y the y bytes are ``ln_vg_w``'s y, and v matches its v."""
    from miniworld_engine.kernels.integrations import pwa_wide as PW

    torch.manual_seed(0)
    s, n, d, c = 70, 192, 256, 32
    m = torch.randn(s, n, d, device="cuda").bfloat16()
    lw, lb = 1 + 0.1 * torch.randn(d, device="cuda"), 0.1 * torch.randn(d, device="cuda")
    wv = (torch.randn(8 * c, d, device="cuda") * d**-0.5).bfloat16()
    v, y_ref, _ = PW._value(d, c).ln_vg_w(m, lw, lb, wv, 1e-5, True)
    amax = torch.zeros(1, device="cuda")
    vt, y = f8._ext(d, c).ln_vt8(m, lw, lb, wv, 1e-5, torch.full((1,), 8.0, device="cuda"), amax, *f8.wide_cfg(d, c), True)
    assert torch.equal(y, y_ref)
    vd = vt.float().transpose(1, 2) / 8.0
    ref = v.float()
    assert float((vd - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()) < 0.04

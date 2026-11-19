"""B200 token DiT e4m3 expand route (``conditioned_transition.cuda.token_dit_fp8_b200``, opt-in, default OFF): route on / off, accuracy against the fp64 gold next
to the bf16 default route, saturation stress (input x 0.05 / x 20), graph == eager, route stats, row gate."""

import copy
import dataclasses
import re

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) only")

YAML = "modules: [TokenDiTProtenix, TokenDiTChai1, TokenDiTRoseTTAFold3, TokenDiTBoltz2]\nbackends: [pytorch, cuda]\nexternal: []\nrows: []\ndtypes: []\nmodes: []\nlengths: [128]\ndepths: min\nsamples: all\ndevice: cuda\nseed: 0\nfixed: {bfloat16: 8.0e-3, master: 8.0e-3, float32: 1.0e-5}\n"


def _case(tmp_path, module, L, S=5):
    from miniworld_engine.measure import config as C

    p = tmp_path / "c.yaml"
    p.write_text(YAML)
    cfg = C.load(str(p))
    for case in C.cases(cfg):
        lab = C.describe(case, cfg)["label"]
        if case.module == module and case.mode == "eval" and case.dtype == "master" and "d_pair=128" in lab or (module == "TokenDiTChai1" and case.module == module and case.mode == "eval" and case.dtype == "master"):
            return dataclasses.replace(case, length=L, samples=S)
    raise AssertionError(module)


class Rig:
    def __init__(self, case):
        from miniworld_engine.measure import config as C, speed as S

        dev = torch.device("cuda")
        self.m = copy.deepcopy(C.build(case, device=dev))
        with S.unpadded_masks():
            self.inp = C.inputs(case, self.m, device=dev)
        self.case, self.C, self.S = case, C, S
        self.step = S.make_step(self.m, self.inp, backend="cuda", train=False, autocast=C.autocast(case), seed=0, compile_mode=None)
        self.m64 = copy.deepcopy(self.m).double()
        self.inp64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in self.inp.items()}
        self.xk = next(k for k, v in self.inp.items() if torch.is_tensor(v) and v.is_floating_point() and v.ndim == 4)

    def gold(self, scale=1.0):
        inp = dict(self.inp64)
        inp[self.xk] = inp[self.xk] * scale
        st = self.S.make_step(self.m64, inp, backend="pytorch", train=False, autocast=None, seed=0, compile_mode=None)
        return st().double(), inp[self.xk].double()

    def run(self, scale=1.0):
        x = self.inp[self.xk]
        keep = x.clone()
        x.mul_(scale)
        try:
            o = self.step()
            torch.cuda.synchronize()
            return (o if torch.is_tensor(o) else o[0]).float().clone()
        finally:
            x.copy_(keep)


def _err(y, gold, x):
    d = y.double() - gold
    return float(d.pow(2).mean().sqrt() / (gold - x).pow(2).mean().sqrt()), float(d.abs().max() / gold.abs().max())


@pytest.fixture
def f8():
    from miniworld_engine.kernels.conditioned_transition.cuda import token_dit_fp8 as F

    old = (F.PLACES, F.MIN_ROWS)
    yield F
    F.PLACES, F.MIN_ROWS = old


@pytest.mark.parametrize("module", ["TokenDiTProtenix", "TokenDiTChai1"])
def test_route_accuracy_stress_and_stats(tmp_path, f8, module):
    rig = Rig(_case(tmp_path, module, 384))
    for scale in (1.0, 0.05, 20.0):
        f8.PLACES, f8.MIN_ROWS = frozenset(), 0
        y0 = rig.run(scale)
        gold, x = rig.gold(scale)
        e0 = _err(y0, gold, x)
        f8.PLACES, f8.MIN_ROWS = f8._parse("expand"), 0
        f8.STATS.update({k: 0 for k in f8.STATS})
        y1 = rig.run(scale)
        e1 = _err(y1, gold, x)
        assert f8.STATS["expand"] >= 1 and f8.STATS["declined"] == 0, f8.STATS
        assert torch.isfinite(y1).all()
        assert not torch.equal(y0, y1)  # the route took the e4m3 expand
        # the H100 route's figures: ~5.4 % update rms at input scale 1 (6.2 % Chai-1), diluted at x 20; the bf16 route 0.7 %
        if scale == 20.0:  # the update is a few % of the stream here: judge form (max error over max |y64|) and rms within the H100 route's 9-13 %
            assert e1[1] < 0.01 and e1[0] < 0.2, (scale, e0, e1)
        else:
            assert e1[0] < 0.09 and e1[1] < 0.15, (scale, e0, e1)
            assert e1[0] / max(e0[0], 1e-9) < 15, (scale, e0, e1)
        print(module, scale, "bf16 default (rms, max):", e0, "e4m3 expand:", e1, "ratio", e1[0] / e0[0])


@pytest.mark.parametrize("module", ["TokenDiTProtenix", "TokenDiTChai1"])
def test_off_and_gated_are_bit_equal(tmp_path, f8, module):
    from miniworld_engine.kernels.conditioned_transition.cuda import token_dit_fp8 as F

    rig = Rig(_case(tmp_path, module, 128))  # M = 640: below the B200 row gate
    f8.PLACES, f8.MIN_ROWS = frozenset(), None
    y0 = rig.run()
    f8.PLACES = f8._parse("expand")
    f8.STATS.update({k: 0 for k in f8.STATS})
    y1 = rig.run()
    assert F.MIN_ROWS_B200["expand"] > 640 and f8.STATS["calls"] == 0 and f8.STATS["declined"] >= 1, f8.STATS
    assert torch.equal(y0, y1)


def test_rf3_and_up_blocks_are_not_served(tmp_path, f8):
    for module in ("TokenDiTRoseTTAFold3", "TokenDiTBoltz2"):
        rig = Rig(_case(tmp_path, module, 384))
        f8.PLACES, f8.MIN_ROWS = frozenset(), 0
        y0 = rig.run()
        f8.PLACES = f8._parse("expand")
        f8.STATS.update({k: 0 for k in f8.STATS})
        y1 = rig.run()
        assert f8.STATS["calls"] == 0 and torch.equal(y0, y1), (module, f8.STATS)


def test_graph_equals_eager(tmp_path, f8):
    from miniworld_engine.kernels import _capture

    rig = Rig(_case(tmp_path, "TokenDiTProtenix", 384))
    f8.PLACES, f8.MIN_ROWS = f8._parse("expand"), 0
    ye = rig.run()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with _capture.static_weights(), _capture.static_inputs():
        with torch.cuda.stream(side):
            for _ in range(3):
                rig.step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=side):
            og = rig.step()
    og = og if torch.is_tensor(og) else og[0]
    g.replay()
    torch.cuda.synchronize()
    a = og.float().clone()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(a, og.float())  # replays are deterministic
    gold, x = rig.gold()
    assert _err(a, gold, x)[0] < 0.09 and _err(ye, gold, x)[0] < 0.09
    assert (a.double() - ye.double()).abs().max() / ye.abs().max() < 0.02

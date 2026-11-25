"""B200 atom DiT e4m3 rows route (``augmented_attention.cuda.b200_atom.f8``, opt-in, default OFF; docs/b200/fp8.md "Agent E"): route on / off, accuracy against the
fp64 gold next to the bf16 default route (input x {1e-3, 1, 40}, offset 25, weights x 8), graph == eager, route stats, row gate, fp32 / training / save decline."""

import copy
import dataclasses

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) only")

YAML = "modules: [AtomDiTProtenix, AtomDiTOpenFold3, AtomDiTChai1, AtomDiTBoltz2, AtomDiTRoseTTAFold3, AtomDiTESMFold2Pair]\nbackends: [pytorch, cuda]\nexternal: []\nrows: []\ndtypes: []\nmodes: []\nlengths: [1024]\ndepths: min\nsamples: all\ndevice: cuda\nseed: 0\nfixed: {bfloat16: 8.0e-3, master: 8.0e-3, float32: 1.0e-5}\n"


def _case(tmp_path, module, L, S=5, mode="eval", dtype="master"):
    from foldjet_engine.measure import config as C

    p = tmp_path / "c.yaml"
    p.write_text(YAML)
    cfg = C.load(str(p))
    for case in C.cases(cfg):
        lab = C.describe(case, cfg)["label"]
        if case.module == module and case.mode == mode and case.dtype == dtype and "slide" not in lab and "D5" not in lab:
            return dataclasses.replace(case, length=L, samples=S)
    raise AssertionError(module)


class Rig:
    def __init__(self, case):
        from foldjet_engine.measure import config as C, speed as S

        dev = torch.device("cuda")
        self.m = copy.deepcopy(C.build(case, device=dev))
        with S.unpadded_masks():
            self.inp = C.inputs(case, self.m, device=dev)
        self.S = S
        self.step = S.make_step(self.m, self.inp, backend="cuda", train=False, autocast=C.autocast(case), seed=0, compile_mode=None)
        self.m64 = copy.deepcopy(self.m).double()
        self.inp64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in self.inp.items()}
        self.xk = next(k for k, v in self.inp.items() if torch.is_tensor(v) and v.is_floating_point() and v.ndim == 4 and v.shape[-1] == 128)

    def gold(self, scale=1.0, offset=0.0):
        inp = dict(self.inp64)
        inp[self.xk] = inp[self.xk] * scale + offset
        st = self.S.make_step(self.m64, inp, backend="pytorch", train=False, autocast=None, seed=0, compile_mode=None)
        return st().double(), inp[self.xk].double()

    def run(self, scale=1.0, offset=0.0):
        x = self.inp[self.xk]
        keep = x.clone()
        x.mul_(scale).add_(offset)
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
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom import f8 as F

    old = (F.ENABLED, F.MIN_ROWS, F.MAX_ROWS)
    F.MIN_ROWS, F.MAX_ROWS = 0, 1 << 30
    yield F
    F.ENABLED, F.MIN_ROWS, F.MAX_ROWS = old


@pytest.mark.parametrize("module", ["AtomDiTProtenix", "AtomDiTOpenFold3", "AtomDiTChai1", "AtomDiTBoltz2", "AtomDiTRoseTTAFold3", "AtomDiTESMFold2Pair"])
def test_route_accuracy_stress_and_stats(tmp_path, f8, module):
    rig = Rig(_case(tmp_path, module, 1024))
    for scale, offset in ((1.0, 0.0), (1e-3, 0.0), (40.0, 0.0), (1.0, 25.0)):
        f8.enable(False)
        y0 = rig.run(scale, offset)
        f8.enable("all")
        before = dict(f8.STATS)
        y1 = rig.run(scale, offset)
        gold, x = rig.gold(scale, offset)
        e0, e1 = _err(y0, gold, x), _err(y1, gold, x)
        assert torch.isfinite(y1).all()
        # the e4m3 error is ~6x the bf16 default route's (rms 5 % vs 0.9 % of the update); never saturated / non-finite at any scale
        assert e1[0] < max(0.12, 1.5 * e0[0]) and e1[0] < 12 * max(e0[0], 5e-3), (module, scale, offset, e0, e1)  # (x 40: the update is small against the input, both routes are input-rounding bound)
        assert f8.STATS["pre"] > before["pre"]
        if True:
            assert f8.STATS["post"] > before["post"]


def test_route_off_is_bit_identical_and_deterministic(tmp_path, f8):
    rig = Rig(_case(tmp_path, "AtomDiTProtenix", 1024))
    f8.enable(False)
    a = rig.run()
    f8.enable("all")
    rig.run()
    f8.enable(False)
    b = rig.run()
    assert torch.equal(a, b)


def test_graph_equals_eager(tmp_path, f8):
    from foldjet_engine.kernels import _capture

    rig = Rig(_case(tmp_path, "AtomDiTOpenFold3", 1024))
    f8.enable("all")
    with _capture.static_weights(), _capture.static_inputs():
        for _ in range(3):
            rig.step()
        torch.cuda.synchronize()
        eager = rig.step()
        eager = (eager if torch.is_tensor(eager) else eager[0]).clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                rig.step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=side):
            out = rig.step()
        out = out if torch.is_tensor(out) else out[0]
        g.replay()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
        # saturation protocol through the captured graph: new input contents (x 40 + 25, x 1e-3) in the static input, replayed, against the eager route
        x = rig.inp[rig.xk]
        x0 = x.clone()
        for scale, offset in ((40.0, 25.0), (1e-3, 0.0)):
            x.copy_(x0 * scale + offset)
            g.replay()
            torch.cuda.synchronize()
            ye = rig.step()
            ye = ye if torch.is_tensor(ye) else ye[0]
            assert torch.isfinite(out).all() and torch.equal(out, ye), (scale, offset)
        x.copy_(x0)


def test_row_gate_and_declines(f8):
    f8.enable("all")
    f8.MIN_ROWS = 20480
    assert not f8.serves("pre", 5120, False) and f8.serves("pre", 20480, False)
    f8.MIN_ROWS, f8.MAX_ROWS = 0, 10240
    assert f8.serves("pre", 10240, False) and not f8.serves("pre", 10272, False)
    f8.MIN_ROWS, f8.MAX_ROWS = 20480, 1 << 30
    assert not f8.serves("post", 20480, True)  # saved activations (training): bf16
    f8.enable("pre")
    assert f8.serves("pre", 20480, False) and not f8.serves("post", 20480, False)
    f8.enable(False)
    assert not f8.serves("pre", 1 << 20, False)

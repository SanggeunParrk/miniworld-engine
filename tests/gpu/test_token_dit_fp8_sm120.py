"""RTX 5090 (sm_120a) token DiT e4m3 route (``conditioned_transition.cuda.token_dit_fp8_sm120``, ``fp8.SM120_ROUTES["token_dit"]``; opt-in, default
OFF): accuracy against the fp64 gold next to the bf16 default for every token DiT class the H100 route covers (Protenix / MiniWorld default block,
Boltz-2 up gate, RoseTTAFold3 parallel block, Chai-1 identity AdaLN), stress inputs (x 0.05 / x 20), the opt-in placements, the default untouched
byte for byte, the row gate, graph == eager, the ``safe`` preset and ``route_stats``."""

import copy
import dataclasses
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="RTX 5090 (sm_120a) only")

YAML = (
    "modules: [{m}]\nbackends: [pytorch, cuda]\nexternal: []\nrows: []\ndtypes: []\nmodes: []\nlengths: [128]\ndepths: min\nsamples: all\n"
    "device: cuda\nseed: 0\nfixed: {{bfloat16: 8.0e-3, master: 8.0e-3, float32: 1.0e-5}}\n"
)
CLASSES = ["TokenDiTProtenix", "TokenDiTMiniWorld", "TokenDiTBoltz2", "TokenDiTRoseTTAFold3", "TokenDiTChai1"]


def _case(module, L, S=5):
    from foldjet_engine.measure import config as C

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(YAML.format(m=module))
    cfg = C.load(f.name)
    best = None
    for c in C.cases(cfg):
        if c.module == module and c.mode == "eval" and c.dtype == "master":
            lab = C.describe(c, cfg)["label"]
            if best is None or ("d_pair=128" in lab and "d_pair=128" not in best[1]):
                best = (c, lab)
    assert best is not None, module
    c = best[0]
    stream = c.stream + "_shared_cond" if c.stream == "token_single" else c.stream
    return dataclasses.replace(c, length=L, samples=S, stream=stream)


class Rig:
    def __init__(self, case):
        from foldjet_engine.measure import config as C
        from foldjet_engine.measure import speed as S

        dev = torch.device("cuda")
        self.m = copy.deepcopy(C.build(case, device=dev))
        with S.unpadded_masks():
            self.inp = C.inputs(case, self.m, device=dev)
        self.S = S
        self.step = S.make_step(self.m, self.inp, backend="cuda", train=False, autocast=C.autocast(case), seed=0, compile_mode=None)
        self.xk = next(k for k, v in self.inp.items() if torch.is_tensor(v) and v.is_floating_point() and v.ndim == 4)

    def gold(self, scale=1.0):
        m64 = copy.deepcopy(self.m).double()
        inp = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in self.inp.items()}
        inp[self.xk] = inp[self.xk] * scale
        st = self.S.make_step(m64, inp, backend="pytorch", train=False, autocast=None, seed=0, compile_mode=None)
        with torch.no_grad():
            y = st()
        return (y if torch.is_tensor(y) else y[0]).double(), inp[self.xk].double()

    def run(self, scale=1.0):
        x = self.inp[self.xk]
        keep = x.clone()
        x.mul_(scale)
        try:
            with torch.no_grad():
                o = self.step()
            torch.cuda.synchronize()
            return (o if torch.is_tensor(o) else o[0]).float().clone()
        finally:
            x.copy_(keep)


def _err(y, gold, x):
    """(update rms, judge): rms(y - y64) / rms(y64 - x), max|y - y64| / max|y64|."""
    d = y.double() - gold
    return float(d.pow(2).mean().sqrt() / (gold - x).pow(2).mean().sqrt()), float(d.abs().max() / gold.abs().max())


@pytest.fixture
def f8():
    from foldjet_engine.kernels.conditioned_transition.cuda import token_dit_fp8_sm120 as F

    old = (F.PLACES, F.MIN_ROWS)
    yield F
    F.PLACES, F.MIN_ROWS = old


def _on(F, places):
    F.PLACES, F.MIN_ROWS = F._parse(places), 0
    F.STATS.update({k: 0 for k in F.STATS})


@pytest.mark.parametrize("module", CLASSES)
def test_expand_accuracy_stress_and_stats(f8, module):
    rig = Rig(_case(module, 384))
    for scale in (1.0, 0.05, 20.0):
        f8.PLACES = frozenset()
        y0 = rig.run(scale)
        gold, x = rig.gold(scale)
        e0 = _err(y0, gold, x)
        _on(f8, "expand")
        y1 = rig.run(scale)
        e1 = _err(y1, gold, x)
        assert f8.STATS["expand"] >= 1 and f8.STATS["calls"] >= 1 and f8.STATS["declined"] == 0, f8.STATS
        assert torch.isfinite(y1).all() and not torch.equal(y0, y1)
        # the H100 route: update rms 5.4 % (Chai-1 6.2 %), judge 3-4e-2 at input x 1 / x 0.05; at x 20 the update is a few % of the stream
        if scale == 20.0:
            assert e1[1] < 0.01 and e1[0] < 0.2, (module, scale, e0, e1)
        else:
            assert e1[0] < 0.09 and e1[1] < 0.08, (module, scale, e0, e1)
            assert e1[0] / max(e0[0], 1e-9) < 15, (module, scale, e0, e1)
        print(module, scale, "default (upd rms, judge):", e0, "e4m3 expand:", e1)


@pytest.mark.parametrize("module", ["TokenDiTProtenix", "TokenDiTMiniWorld", "TokenDiTBoltz2", "TokenDiTChai1", "TokenDiTRoseTTAFold3"])
def test_opt_in_placements(f8, module):
    """qkvg / out / squeeze (opt-in, not in safe): served where the H100 route serves them, error bounded (the H100 all-four column: 6.6 % / 4.4e-2)."""
    rig = Rig(_case(module, 384))
    gold, x = rig.gold()
    _on(f8, "all")
    y = rig.run()
    e = _err(y, gold, x)
    want = {
        "TokenDiTProtenix": {"qkvg", "out", "expand", "squeeze"},
        "TokenDiTMiniWorld": {"qkvg", "out", "expand", "squeeze"},
        "TokenDiTBoltz2": {"qkvg", "out", "expand", "squeeze"},
        "TokenDiTChai1": {"out", "expand", "squeeze"},  # no gating query: no qkvg (its AdaLN scale is unbounded, the H100 rule)
        "TokenDiTRoseTTAFold3": {"expand"},  # the parallel block with the k-q LayerNorm: expand only (the H100 rule)
    }[module]
    assert {k for k in ("qkvg", "out", "expand", "squeeze") if f8.STATS[k]} == want, (module, f8.STATS)
    assert torch.isfinite(y).all() and e[0] < 0.11 and e[1] < 0.09, (module, e)
    print(module, "all four (upd rms, judge):", e)


@pytest.mark.parametrize("module", ["TokenDiTProtenix", "TokenDiTBoltz2"])
def test_default_untouched_and_gate(f8, module):
    """With the route off the block is the default byte for byte, also after e4m3 calls; below the row gate the route declines (bit-equal)."""
    rig = Rig(_case(module, 128))  # M = 640
    f8.PLACES = frozenset()
    y0 = rig.run()
    _on(f8, "all")
    y1 = rig.run()
    assert not torch.equal(y0, y1)
    f8.PLACES = frozenset()
    assert torch.equal(rig.run(), y0)
    f8.PLACES, f8.MIN_ROWS = f8._parse("all"), 641
    f8.STATS.update({k: 0 for k in f8.STATS})
    assert torch.equal(rig.run(), y0)
    assert f8.STATS["calls"] == 0 and f8.STATS["declined"] >= 1, f8.STATS


@pytest.mark.parametrize("module", ["TokenDiTProtenix", "TokenDiTChai1"])
def test_graph_equals_eager(f8, module):
    from foldjet_engine.kernels import _capture

    rig = Rig(_case(module, 256))
    _on(f8, "all")
    y_e = rig.run()
    with torch.no_grad(), _capture.static_weights(), _capture.static_inputs():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            rig.step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o = rig.step()
        g.replay()
        g.replay()
        torch.cuda.synchronize()
    y_g = (o if torch.is_tensor(o) else o[0]).float()
    assert torch.equal(y_g, y_e)


def test_safe_preset_and_route_stats(f8):
    from foldjet_engine import fp8

    assert fp8.arch() == "sm120"
    snap = fp8._snapshot()
    try:
        fp8.enable("safe")
        assert f8.PLACES == frozenset({"expand"}) and fp8.status()["token_dit"] == ["expand"]
        f8.MIN_ROWS = 0
        fp8.reset_route_stats()
        Rig(_case("TokenDiTProtenix", 256)).run()
        st = fp8.route_stats()["token_dit"]
        assert st["calls"] >= 1 and st["expand"] >= 1, st
        fp8.disable()
        assert f8.PLACES == frozenset()
    finally:
        fp8._restore(snap)

"""RTX 5090 (sm_120a) atom DiT e4m3 route (``dit_variants.cuda.atom_fp8_sm120``, ``fp8.SM120_ROUTES["atom_dit"]``; opt-in, default OFF): accuracy
against the fp64 gold next to the bf16 default for the atom DiT classes of the H100 route (Protenix: chained K / V AdaLN, OpenFold3: parallel,
Boltz-2: up gate, RoseTTAFold3: q / k LayerNorm, Chai-1: slid window), the parts (pre / post), stress inputs, the default untouched byte for byte,
graph == eager, training not served, the ``safe`` preset and ``route_stats``."""

import copy
import dataclasses
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="RTX 5090 (sm_120a) only")

YAML = (
    "modules: [{m}]\nbackends: [pytorch, cuda]\nexternal: []\nrows: []\ndtypes: []\nmodes: []\nlengths: min\ndepths: min\nsamples: all\n"
    "device: cuda\nseed: 0\nfixed: {{bfloat16: 8.0e-3, master: 8.0e-3, float32: 1.0e-5}}\n"
)
CASES = [("AtomDiTProtenix", None), ("AtomDiTOpenFold3", None), ("AtomDiTBoltz2", None), ("AtomDiTRoseTTAFold3", None), ("AtomDiTChai1", "key_window=slide")]


def _case(module, N, S=5, pick=None, shared=True):
    from miniworld_engine.measure import config as C

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(YAML.format(m=module))
    cfg = C.load(f.name)
    best = None
    for c in C.cases(cfg):
        if c.module != module or c.mode != "eval" or c.dtype != "master" or c.stream != "atom_single":
            continue
        lab = C.describe(c, cfg)["label"]
        if (pick is None and ("key_window" in lab or "mask_form" in lab)) or (pick is not None and pick not in lab):
            continue
        best = best or c
    assert best is not None, module
    return dataclasses.replace(best, length=N, samples=S, stream="atom_single_shared_cond" if shared else "atom_single")


class Rig:
    def __init__(self, case):
        from miniworld_engine.measure import config as C
        from miniworld_engine.measure import speed as S

        dev = torch.device("cuda")
        self.m = copy.deepcopy(C.build(case, device=dev))
        with S.unpadded_masks():
            self.inp = C.inputs(case, self.m, device=dev)
        self.S, self.C, self.case = S, C, case
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
    """(update rms, rms_rel, judge): rms(y - y64) / rms(y64 - x), ||y - y64|| / ||y64||, max|y - y64| / max|y64|."""
    d = y.double() - gold
    return (
        float(d.pow(2).mean().sqrt() / (gold - x).pow(2).mean().sqrt()),
        float(d.pow(2).sum().sqrt() / gold.pow(2).sum().sqrt()),
        float(d.abs().max() / gold.abs().max()),
    )


@pytest.fixture
def f8():
    from miniworld_engine.kernels.dit_variants.cuda import atom_fp8_sm120 as F

    old = (F.FP8_ROWS, F.MIN_ROWS)
    F.MIN_ROWS = 0
    yield F
    F.FP8_ROWS, F.MIN_ROWS = old


def _on(F, parts):
    F.enable_fp8(parts)
    F.STATS.update({k: 0 for k in F.STATS})


@pytest.mark.parametrize("module,pick", CASES)
def test_accuracy_parts_and_stress(f8, module, pick):
    rig = Rig(_case(module, 1024, pick=pick))
    for scale in (1.0, 0.05, 20.0):
        f8.enable_fp8(False)
        y0 = rig.run(scale)
        gold, x = rig.gold(scale)
        e0 = _err(y0, gold, x)
        for parts in ("pre", "post", "1"):
            _on(f8, parts)
            y1 = rig.run(scale)
            e1 = _err(y1, gold, x)
            want = {"pre": ("pre",), "post": ("post",), "1": ("pre", "post")}[parts]
            assert f8.STATS["calls"] >= 1 and all(f8.STATS[p] >= 1 for p in want) and f8.STATS["fallbacks"] == 0, f8.STATS
            assert torch.isfinite(y1).all() and not torch.equal(y0, y1)
            # the H100 route (pre + post): rms_rel 2.9 % (Chai-1 slide 5.6 %), judge 5e-2 (Chai-1 1e-1)
            tol_rel, tol_judge = (0.08, 0.15) if module == "AtomDiTChai1" else (0.04, 0.08)
            if scale == 0.05:  # the stream is small: the update dominates y, rms_rel ~ the update error
                tol_rel, tol_judge = 0.12, max(tol_judge, 0.12)
            assert e1[1] < tol_rel and e1[2] < tol_judge and e1[0] < 0.14, (module, parts, scale, e0, e1)
            print(module, parts, scale, "default (upd rms, rms_rel, judge):", e0, "e4m3:", e1)


def test_default_untouched_training_and_rows(f8):
    rig = Rig(_case("AtomDiTOpenFold3", 1024))
    f8.enable_fp8(False)
    y0 = rig.run()
    _on(f8, "1")
    assert not torch.equal(rig.run(), y0)
    f8.enable_fp8(False)
    assert torch.equal(rig.run(), y0)
    _on(f8, "1")
    f8.MIN_ROWS = 5 * 1024 + 1  # below the row gate: the default, counted
    assert torch.equal(rig.run(), y0) and f8.STATS["why_rows"] >= 1 and f8.STATS["calls"] == 0, f8.STATS
    f8.MIN_ROWS = 0
    # a call with a gradient (training) is not served: the default autograd block, counted as why_save
    f8.STATS.update({k: 0 for k in f8.STATS})
    inp = dict(rig.inp)
    inp[rig.xk] = inp[rig.xk].detach().clone().requires_grad_(True)
    rig.m.set_backend("cuda")
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = rig.m(**inp)
        (out if torch.is_tensor(out) else out[0]).float().sum().backward()
    assert f8.STATS["calls"] == 0 and f8.STATS["why_save"] >= 1, f8.STATS


@pytest.mark.parametrize("module,pick", [("AtomDiTProtenix", None), ("AtomDiTChai1", "key_window=slide")])
def test_graph_equals_eager(f8, module, pick):
    from miniworld_engine.kernels import _capture

    rig = Rig(_case(module, 1024, pick=pick))
    _on(f8, "1")
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
    assert torch.equal((o if torch.is_tensor(o) else o[0]).float(), y_e)


def test_per_sample_conditioning(f8):
    """A conditioning per sample (the table [A N, 8 d]) is served too."""
    rig = Rig(_case("AtomDiTOpenFold3", 1024, shared=False))
    gold, x = rig.gold()
    _on(f8, "1")
    e = _err(rig.run(), gold, x)
    assert f8.STATS["calls"] >= 1 and e[1] < 0.04, (f8.STATS, e)


def test_safe_preset_and_route_stats(f8):
    from miniworld_engine import fp8

    assert fp8.arch() == "sm120" and fp8.SM120_ROUTES["atom_dit"] == f8.__name__
    snap = fp8._snapshot()
    try:
        fp8.enable("safe")
        assert f8.FP8_ROWS == frozenset({"pre", "post"}) and fp8.status()["atom_dit"] == ["post", "pre"]
        fp8.reset_route_stats()
        Rig(_case("AtomDiTProtenix", 1024)).run()
        st = fp8.route_stats()["atom_dit"]
        assert st["calls"] >= 1 and st["pre"] >= 1 and st["post"] >= 1, st
        fp8.disable()
        assert f8.FP8_ROWS == frozenset()
    finally:
        fp8._restore(snap)

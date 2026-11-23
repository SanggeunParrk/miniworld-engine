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

    old = (F.FP8_ROWS, F.MIN_ROWS, F.ESM_MIN_ROWS)
    F.MIN_ROWS = F.ESM_MIN_ROWS = 0
    yield F
    F.FP8_ROWS, F.MIN_ROWS, F.ESM_MIN_ROWS = old


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


@pytest.mark.parametrize("module,pick", [("AtomDiTOpenFold3", None), ("AtomDiTProtenix", None)])
def test_compiled_block_is_finite_and_equals_eager(f8, module, pick):
    """Regression (5090-int, FoldForge AF3 5I28 n = 4 under torch.compile gave non-finite coordinates): the e4m3 block is ONE opaque op, so a
    compiled step runs the eager launches: finite and at the eager e4m3 call (the compiled glue around the op may round differently)."""
    import torch._dynamo

    rig = Rig(_case(module, 1024, pick=pick))
    _on(f8, "1")
    y_e = rig.run()
    torch._dynamo.reset()
    step = rig.S.make_step(rig.m, rig.inp, backend="cuda", train=False, autocast=rig.C.autocast(rig.case), seed=0, compile_mode="default")
    with torch.no_grad():
        for _ in range(2):
            o = step()
    torch.cuda.synchronize()
    y_c = (o if torch.is_tensor(o) else o[0]).float()
    assert f8.STATS["calls"] >= 1 and torch.isfinite(y_c).all()
    assert ((y_c - y_e).abs().max() / y_e.abs().max()).item() < 2e-2, (y_c - y_e).abs().max()  # the glue around the op may fuse differently


# ------------------------------------------------------------------------------------------------------------ ESMFold2 (the H100 RMS option)
def _esm(n: int, a: int, shared: bool = True):
    """``AtomDiTESMFold2Pair`` (the H100 test's block: fp32 parameters, bf16 inputs, 40 masked atoms) and its float64 PyTorch twin."""
    from miniworld_engine.modules import AtomDiTESMFold2Pair

    torch.manual_seed(0)
    m = AtomDiTESMFold2Pair(backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 1 + torch.randn_like(p) * 0.1)
    g = torch.Generator(device="cuda").manual_seed(1)
    single = torch.randn(a, 1, n, 128, device="cuda", generator=g).bfloat16()
    cond = torch.randn(1 if shared else a, 1, n, 128, device="cuda", generator=g).bfloat16().expand(a, 1, n, 128)
    pair = torch.randn(1, (n + 31) // 32, 32, 192, 16, device="cuda", generator=g).bfloat16()
    mask = torch.zeros(1, n, dtype=torch.bool, device="cuda")
    mask[:, : n - 40] = True
    angle = 3 * torch.randn(1, n, 16, device="cuda", generator=g)
    return m, (single, cond, pair, mask), (angle.cos(), angle.sin())


def _esm_gold(m, inputs, rope):
    ref = copy.deepcopy(m).set_backend("pytorch").double()
    with torch.no_grad():
        return ref(*(t.double() if t.is_floating_point() else t for t in inputs), rope=tuple(t.double() for t in rope))


def _esm_run(m, inputs, rope):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return m(*inputs, rope=rope).float().clone()


def _esm_err(y, y64, x, real):
    """(update rms, rms_rel, judge) over the real atoms."""
    y, y64, x = y[:, :, :real].double(), y64[:, :, :real], x[:, :, :real].double()
    return _err(y, y64, x)


@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("parts", ["1", "pre", "post"])
def test_esmfold2_accuracy_and_parts(f8, n, parts):
    """ESMFold2's atom block on the e4m3 rows (RMS mode of ``atom_f8_sm120``): finite, the e4m3 route really ran and was counted, the error
    against the float64 module inside the H100 RMS route's bound (rms_rel < 6 %, the H100 test's; H100 measured 3.8e-2 / 6.4e-2 rms / max)."""
    m, inputs, rope = _esm(n, 5)
    y64 = _esm_gold(m, inputs, rope)
    f8.enable_fp8(False)
    y0 = _esm_run(m, inputs, rope)
    _on(f8, parts)
    y1 = _esm_run(m, inputs, rope)
    want = {"pre": ("pre",), "post": ("post",), "1": ("pre", "post")}[parts]
    assert f8.STATS["calls"] >= 1 and all(f8.STATS[p] >= 1 for p in want) and f8.STATS["fallbacks"] == 0, f8.STATS
    e0, e1 = _esm_err(y0, y64, inputs[0], n - 40), _esm_err(y1, y64, inputs[0], n - 40)
    print("esmfold2", n, parts, "default (upd rms, rms_rel, judge):", e0, "e4m3:", e1)
    assert torch.isfinite(y1).all() and not torch.equal(y0, y1)
    assert e1[1] < 0.06 and e1[2] < 0.1 and e1[1] > e0[1], (e0, e1)
    f8.enable_fp8(False)
    assert torch.equal(_esm_run(m, inputs, rope), y0)  # switched off: the default again, byte for byte


def test_esmfold2_per_sample_rows_gate_training(f8):
    """A per-sample conditioning is served (the H100 RMS route takes a shared one only); below the row gate and with a gradient the default
    runs byte for byte (counted as ``why_rows``), a forward with a gradient never reaches the e4m3 block."""
    m, inputs, rope = _esm(1024, 3, shared=False)
    y64 = _esm_gold(m, inputs, rope)
    f8.enable_fp8(False)
    y0 = _esm_run(m, inputs, rope)
    _on(f8, "1")
    e1 = _esm_err(_esm_run(m, inputs, rope), y64, inputs[0], 1024 - 40)
    assert f8.STATS["calls"] >= 1 and e1[1] < 0.06, (f8.STATS, e1)
    _on(f8, "1")
    f8.ESM_MIN_ROWS = 3 * 1024 + 1  # the ESMFold2 gate (ESM_MIN_ROWS), not the other classes' MIN_ROWS
    assert torch.equal(_esm_run(m, inputs, rope), y0) and f8.STATS["why_rows"] >= 1 and f8.STATS["calls"] == 0, f8.STATS
    f8.ESM_MIN_ROWS = 0
    f8.STATS.update({k: 0 for k in f8.STATS})
    x = inputs[0].detach().clone().requires_grad_(True)
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(x, *inputs[1:], rope=rope).float().sum().backward()
    assert f8.STATS["calls"] == 0, f8.STATS  # the training block (_Block autograd) never reaches the inference op


def test_esmfold2_graph_equals_eager_and_safe(f8):
    """A CUDA-graph replay equals the eager e4m3 call bit for bit; ``fp8.enable("safe")`` turns the ESMFold2 rows on (``route_stats``)."""
    from miniworld_engine import fp8
    from miniworld_engine.kernels import _capture

    m, inputs, rope = _esm(1024, 5)
    _on(f8, "1")
    y_e = _esm_run(m, inputs, rope)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(*inputs, rope=rope)

    with torch.no_grad(), _capture.static_weights(), _capture.static_inputs():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o = step()
        g.replay()
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(o.float(), y_e)
    snap = fp8._snapshot()
    try:
        fp8.enable("safe")
        fp8.reset_route_stats()
        _esm_run(m, inputs, rope)
        st = fp8.route_stats()["atom_dit"]
        assert st["calls"] >= 1 and st["pre"] >= 1 and st["post"] >= 1, st
    finally:
        fp8._restore(snap)

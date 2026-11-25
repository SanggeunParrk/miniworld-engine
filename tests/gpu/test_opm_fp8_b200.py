"""B200 OuterProductMean e4m3 product (``integrations.opm_fp8_b200``, opt-in ``MINIWORLD_OPM_FP8=bound``, default OFF): route on / off, accuracy against the fp64 gold next to the
bf16 route (large-weight stress), graph == eager, route stats, the depth gate."""

import copy
import dataclasses

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) only")

YAML = "modules: [OuterProductMean]\nbackends: [pytorch, cuda]\nexternal: []\nrows: []\ndtypes: []\nmodes: []\nlengths: [128]\ndepths: all\nsamples: all\ndevice: cuda\nseed: 0\nfixed: {bfloat16: 8.0e-3, master: 8.0e-3, float32: 1.0e-5}\n"


def _rig(tmp_path, dims, L, D, tag):
    from foldjet_engine.measure import config as C, speed as S

    p = tmp_path / "o.yaml"
    p.write_text(YAML)
    cfg = C.load(str(p))
    for case in C.cases(cfg):
        lab = C.describe(case, cfg)["label"]
        if dims in lab and tag in lab and case.mode == "eval" and case.dtype == "master":
            case = dataclasses.replace(case, length=L, depth=D)
            dev = torch.device("cuda")
            m = copy.deepcopy(C.build(case, device=dev))
            with S.unpadded_masks():
                inp = C.inputs(case, m, device=dev)
            step = S.make_step(m, inp, backend="cuda", train=False, autocast=C.autocast(case), seed=0, compile_mode=None)
            return m, inp, step, S, C, case
    raise AssertionError((dims, tag))


def _gold(m, inp, S):
    m64 = copy.deepcopy(m).double()
    i64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inp.items()}
    st = S.make_step(m64, i64, backend="pytorch", train=False, autocast=None, seed=0, compile_mode=None)
    g = st()
    return (g if torch.is_tensor(g) else g[0]).double()


def _err(y, g):
    d = y.double() - g
    return float(d.pow(2).mean().sqrt() / g.pow(2).mean().sqrt()), float(d.abs().max() / g.abs().max())


@pytest.fixture
def route():
    from foldjet_engine.kernels.integrations import opm_fp8 as H, opm_fp8_b200 as B

    old = (H.POLICY, B.MIN_S)
    B._CACHES.clear()
    yield H, B
    H.POLICY, B.MIN_S = old
    B._CACHES.clear()


def _run(step):
    o = step()
    torch.cuda.synchronize()
    return (o if torch.is_tensor(o) else o[0]).float().clone()


@pytest.mark.parametrize("dims,tag", [("d_msa=64;d_pair=128", "normalize_before_proj=False)"), ("d_msa=64;d_pair=128", "projection_bias=True"),
                                       ("d_msa=64;d_pair=128", "count_offset"), ("d_msa=128;d_pair=256", "normalize_before_proj=False)")])
def test_accuracy_and_stats(tmp_path, route, dims, tag):
    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, dims, 256, 1024, tag)
    H.POLICY = None
    y0 = _run(step)
    gold = _gold(m, inp, S)
    H.POLICY, B.MIN_S = "bound", None
    B.STATS.update({k: 0 for k in B.STATS})
    y1 = _run(step)
    e0, e1 = _err(y0, gold), _err(y1, gold)
    assert B.STATS["calls"] == 1, B.STATS
    assert not torch.equal(y0, y1) and torch.isfinite(y1).all()
    # H100 route: judge 0.6-1.4 % at depth 1024-4096 (bf16 0.3-0.6 %); here the same operand rounding
    assert e1[1] < 0.03 and e1[0] < 0.06, (e0, e1)
    print(dims, tag, "bf16 route (rms, max):", e0, "e4m3:", e1)


def test_large_weight_stress(tmp_path, route):
    """LN weights x 8, biases x 4 (the bound policy's scales follow the weights: nothing saturates)."""
    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, "d_msa=64;d_pair=128", 256, 1024, "normalize_before_proj=False)")
    with torch.no_grad():
        m.ln_msa.weight.mul_(8.0)
        m.ln_msa.bias.mul_(4.0)
    H.POLICY = None
    y0 = _run(step)
    gold = _gold(m, inp, S)
    H.POLICY, B.MIN_S = "bound", None
    B._CACHES.clear()
    y1 = _run(step)
    e0, e1 = _err(y0, gold), _err(y1, gold)
    assert torch.isfinite(y1).all() and e1[1] < 0.03, (e0, e1)
    print("stress bf16", e0, "e4m3", e1)


def test_below_gate_and_off_are_bit_equal(tmp_path, route):
    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, "d_msa=64;d_pair=128", 256, 128, "normalize_before_proj=False)")
    H.POLICY = None
    y0 = _run(step)
    H.POLICY, B.MIN_S = "bound", None
    B.STATS.update({k: 0 for k in B.STATS})
    y1 = _run(step)
    assert B.STATS["calls"] == 0 and torch.equal(y0, y1)


def test_graph_equals_eager(tmp_path, route):
    from foldjet_engine.kernels import _capture

    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, "d_msa=64;d_pair=128", 256, 1024, "normalize_before_proj=False)")
    H.POLICY, B.MIN_S = "bound", None
    ye = _run(step)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with _capture.static_weights(), _capture.static_inputs():
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=side):
            og = step()
    og = og if torch.is_tensor(og) else og[0]
    g.replay()
    torch.cuda.synchronize()
    a = og.float().clone()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(a, og.float()) and torch.equal(a, ye)


# ---------------------------------------------------------------------------------------------------- wide prologue: e4m3 A2 / BT written by the drain (``opm_wprologue_f8``, agent G)
@pytest.mark.parametrize("dims,tag", [("d_msa=128;d_pair=384", "normalize_before_proj=False)"), ("d_msa=256;d_pair=512", "normalize_before_proj=False)"),
                                       ("d_msa=64;d_pair=128", "normalize_before_proj=True;projection_bias=True"),
                                       ("d_msa=128;d_pair=256", "count_offset")])
def test_wide_prologue_accuracy_and_stress(tmp_path, route, dims, tag):
    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, dims, 256, 1024, tag)
    for stress in (1.0, 8.0):  # LN weights x 8, biases x 4: the bound policy's scales follow the weights
        with torch.no_grad():
            if stress != 1.0:
                m.ln_msa.weight.mul_(stress)
                m.ln_msa.bias.mul_(4.0)
        H.POLICY, B.MIN_S = None, None
        y0 = _run(step)
        gold = _gold(m, inp, S)
        H.POLICY, B.MIN_S = "bound", None
        B._CACHES.clear()
        B.STATS.update({k: 0 for k in B.STATS})
        y1 = _run(step)
        e0, e1 = _err(y0, gold), _err(y1, gold)
        assert B.STATS["calls"] == 1, B.STATS
        assert torch.isfinite(y1).all() and not torch.equal(y0, y1)
        assert e1[1] < 0.04 and e1[0] < 0.08, (dims, tag, stress, e0, e1)
        print("OPM wide", dims, tag, "stress", stress, "bf16 route (rms, max):", e0, "e4m3:", e1)


def test_wide_graph_equals_eager(tmp_path, route):
    from foldjet_engine.kernels import _capture

    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, "d_msa=128;d_pair=256", 256, 1024, "normalize_before_proj=False)")
    H.POLICY, B.MIN_S = "bound", None
    ye = _run(step)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with _capture.static_weights(), _capture.static_inputs():
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=side):
            og = step()
    og = og if torch.is_tensor(og) else og[0]
    g.replay()
    torch.cuda.synchronize()
    a = og.float().clone()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(a, og.float()) and torch.equal(a, ye)


# ---------------------------------------------------------------------------------------------------- fused e4m3 kernel (S <= 512, ``opm_fused_infer_f8``) and the Chai-1 grouped product (agent G)
@pytest.mark.parametrize("dims,tag,L,D", [("d_msa=64;d_pair=128", "normalize_before_proj=False)", 768, 384), ("d_msa=64;d_pair=128", "projection_bias=True", 768, 512),
                                          ("d_msa=64;d_pair=128", "normalize_before_proj=True)", 512, 256)])
def test_fused_e4m3_accuracy_and_stress(tmp_path, route, dims, tag, L, D):
    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, dims, L, D, tag)
    for stress in (1.0, 8.0):
        with torch.no_grad():
            if stress != 1.0:
                m.ln_msa.weight.mul_(stress)
                m.ln_msa.bias.mul_(4.0)
        H.POLICY, B.MIN_S = None, None
        y0 = _run(step)
        gold = _gold(m, inp, S)
        H.POLICY = "bound"
        B._CACHES.clear()
        B.STATS.update({k: 0 for k in B.STATS})
        y1 = _run(step)
        e0, e1 = _err(y0, gold), _err(y1, gold)
        assert B.STATS["calls"] == 1, B.STATS  # the fused kernel ran (counted by serves_fused)
        assert torch.isfinite(y1).all() and not torch.equal(y0, y1)
        assert e1[1] < 0.04 and e1[0] < 0.08, (dims, tag, stress, e0, e1)
        print("OPM fused e4m3", dims, tag, L, D, "stress", stress, "bf16 route (rms, max):", e0, "e4m3:", e1)


def test_fused_e4m3_below_gate_is_bit_equal(tmp_path, route):
    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, "d_msa=64;d_pair=128", 512, 128, "normalize_before_proj=False)")  # padded S 128: the bf16 fused kernel
    H.POLICY = None
    y0 = _run(step)
    H.POLICY, B.MIN_S = "bound", None
    B.STATS.update({k: 0 for k in B.STATS})
    y1 = _run(step)
    assert B.STATS["calls"] == 0 and torch.equal(y0, y1)


def _graph_replay_equals(step, ye):
    from foldjet_engine.kernels import _capture

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with _capture.static_weights(), _capture.static_inputs():
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=side):
            og = step()
    og = og if torch.is_tensor(og) else og[0]
    g.replay()
    torch.cuda.synchronize()
    a = og.float().clone()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(a, og.float()) and torch.equal(a, ye)


def test_fused_e4m3_graph_equals_eager(tmp_path, route):
    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, "d_msa=64;d_pair=128", 768, 384, "normalize_before_proj=False)")
    H.POLICY, B.MIN_S = "bound", None
    _graph_replay_equals(step, _run(step))


@pytest.mark.parametrize("tag", ["normalize_before_proj=False)", "normalize_before_proj=True)"])
def test_chai_grouped_e4m3_accuracy_gate_and_graph(tmp_path, route, tag):
    """Chai-1's grouped OPM (8 x 8, product LayerNorm): the e4m3 grouped product from the gate (N >= 384, S >= 1024), error against fp64 next to the bf16 route's,
    below the gate bit-equal, graph == eager."""
    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, "d_msa=64;d_pair=256;d_hidden=64;groups=8", 384, 1024, tag)
    H.POLICY, B.MIN_S = None, None
    y0 = _run(step)
    gold = _gold(m, inp, S)
    H.POLICY = "bound"
    B._CACHES.clear()
    B._CHAI_CACHES.clear()
    B.STATS.update({k: 0 for k in B.STATS})
    ye = _run(step)
    e0, e1 = _err(y0, gold), _err(ye, gold)
    assert B.STATS["calls"] == 1, B.STATS
    assert torch.isfinite(ye).all() and not torch.equal(y0, ye)
    assert e1[1] < 0.06 and e1[0] < 0.08, (e0, e1)  # the product LayerNorm (eps 0.1) does not hide the operand rounding: ~5x the bf16 route
    print("Chai grouped e4m3", tag, "bf16 route (rms, max):", e0, "e4m3:", e1)
    _graph_replay_equals(step, ye)
    m2, inp2, step2, S2, C2, case2 = _rig(tmp_path, "d_msa=64;d_pair=256;d_hidden=64;groups=8", 256, 512, tag)  # below the gate
    H.POLICY = None
    z0 = _run(step2)
    H.POLICY = "bound"
    B.STATS.update({k: 0 for k in B.STATS})
    z1 = _run(step2)
    assert B.STATS["calls"] == 0 and torch.equal(z0, z1)


# ---------------------------------------------------------------------------------------------------- input stress {x scale} x {offset} (agent G): the bound-policy scales cannot saturate (LayerNorm removes scale and offset)
def _stress_inputs(inp, xs, off):
    out = dict(inp)
    for k, v in inp.items():
        if torch.is_tensor(v) and v.is_floating_point() and v.dim() == 4:
            out[k] = (v.float() * xs + off).to(v.dtype)
    return out


@pytest.mark.parametrize("xs,off", [(1e-3, 0.0), (40.0, 0.0), (1.0, 25.0), (40.0, 25.0)])
@pytest.mark.parametrize("dims,tag,L,D", [("d_msa=128;d_pair=256", "normalize_before_proj=False)", 256, 1024), ("d_msa=64;d_pair=128", "normalize_before_proj=False)", 768, 384),
                                          ("d_msa=64;d_pair=256;d_hidden=64;groups=8", "normalize_before_proj=False)", 384, 1024)])
def test_input_stress_scale_offset(tmp_path, route, xs, off, dims, tag, L, D):
    """x scale {1e-3, 40} and offset 25 on the module's float inputs: finite, no saturation (static bound scales), error against fp64 next to the bf16 route's."""
    H, B = route
    m, inp, step, S, C, case = _rig(tmp_path, dims, L, D, tag)
    inp2 = _stress_inputs(inp, xs, off)
    step2 = S.make_step(m, inp2, backend="cuda", train=False, autocast=C.autocast(case), seed=0, compile_mode=None)
    H.POLICY, B.MIN_S = None, None
    y0 = _run(step2)
    gold = _gold(m, inp2, S)
    H.POLICY = "bound"
    B._CACHES.clear()
    B._CHAI_CACHES.clear()
    B.STATS.update({k: 0 for k in B.STATS})
    y1 = _run(step2)
    e0, e1 = _err(y0, gold), _err(y1, gold)
    assert B.STATS["calls"] == 1, B.STATS
    assert torch.isfinite(y1).all()
    print("OPM input stress", dims, "x", xs, "off", off, "bf16 route (rms, max):", e0, "e4m3:", e1)
    assert e1[1] < 0.15 and e1[1] < 8 * e0[1] + 0.05, (e0, e1)


def test_wide_builds_gate_from_padded_depth_256(tmp_path, route):
    """d_msa 128 / c_z 256 has no bf16 fused kernel: the e4m3 route serves from padded S 256 (0.89-0.95), S 128 stays bf16."""
    H, B = route
    for depth, served in ((128, 0), (256, 1)):
        m, inp, step, S, C, case = _rig(tmp_path, "d_msa=128;d_pair=256", 384, depth, "normalize_before_proj=False)")
        H.POLICY, B.MIN_S = None, None
        y0 = _run(step)
        H.POLICY = "bound"
        B.STATS.update({k: 0 for k in B.STATS})
        y1 = _run(step)
        assert B.STATS["calls"] == served, (depth, B.STATS)
        assert torch.equal(y0, y1) == (not served)

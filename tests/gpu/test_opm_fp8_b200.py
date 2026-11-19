"""B200 OuterProductMean e4m3 product (``integrations.opm_fp8_b200``, opt-in ``MINIWORLD_OPM_FP8=bound``, default OFF): route on / off, accuracy against the fp64 gold next to the
bf16 route (large-weight stress), graph == eager, route stats, the depth gate."""

import copy
import dataclasses

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) only")

YAML = "modules: [OuterProductMean]\nbackends: [pytorch, cuda]\nexternal: []\nrows: []\ndtypes: []\nmodes: []\nlengths: [128]\ndepths: all\nsamples: all\ndevice: cuda\nseed: 0\nfixed: {bfloat16: 8.0e-3, master: 8.0e-3, float32: 1.0e-5}\n"


def _rig(tmp_path, dims, L, D, tag):
    from miniworld_engine.measure import config as C, speed as S

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
    from miniworld_engine.kernels.integrations import opm_fp8 as H, opm_fp8_b200 as B

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
    m, inp, step, S, C, case = _rig(tmp_path, "d_msa=64;d_pair=128", 256, 384, "normalize_before_proj=False)")
    H.POLICY = None
    y0 = _run(step)
    H.POLICY, B.MIN_S = "bound", None
    B.STATS.update({k: 0 for k in B.STATS})
    y1 = _run(step)
    assert B.STATS["calls"] == 0 and torch.equal(y0, y1)


def test_graph_equals_eager(tmp_path, route):
    from miniworld_engine.kernels import _capture

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

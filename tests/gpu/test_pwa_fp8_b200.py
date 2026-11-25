"""B200 MSAPairWeightedAveraging e4m3 contraction (``integrations.pwa_fp8_b200``, opt-in ``MINIWORLD_PWA_FP8=1``, default OFF): route on / off, accuracy against the fp64 gold next
to the bf16 route, graph == eager, route stats, the shape gate."""

import copy
import dataclasses

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) only")

YAML = "modules: [MSAPairWeightedAveraging]\nbackends: [pytorch, cuda]\nexternal: []\nrows: []\ndtypes: []\nmodes: []\nlengths: [128]\ndepths: all\nsamples: all\ndevice: cuda\nseed: 0\nfixed: {bfloat16: 8.0e-3, master: 8.0e-3, float32: 1.0e-5}\n"


def _rig(tmp_path, dims, L, D):
    from foldjet_engine.measure import config as C, speed as S

    p = tmp_path / "p.yaml"
    p.write_text(YAML)
    cfg = C.load(str(p))
    for case in C.cases(cfg):
        lab = C.describe(case, cfg)["label"]
        if dims in lab and case.mode == "eval" and case.dtype == "master":
            case = dataclasses.replace(case, length=L, depth=D)
            dev = torch.device("cuda")
            m = copy.deepcopy(C.build(case, device=dev))
            with S.unpadded_masks():
                inp = C.inputs(case, m, device=dev)
            return m, inp, S.make_step(m, inp, backend="cuda", train=False, autocast=C.autocast(case), seed=0, compile_mode=None), S
    raise AssertionError(dims)


def _gold(m, inp, S):
    m64 = copy.deepcopy(m).double()
    i64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inp.items()}
    g = S.make_step(m64, i64, backend="pytorch", train=False, autocast=None, seed=0, compile_mode=None)()
    return (g if torch.is_tensor(g) else g[0]).double()


def _err(y, g):
    d = y.double() - g
    return float(d.pow(2).mean().sqrt() / g.pow(2).mean().sqrt()), float(d.abs().max() / g.abs().max())


def _run(step):
    o = step()
    torch.cuda.synchronize()
    return (o if torch.is_tensor(o) else o[0]).float().clone()


@pytest.fixture
def route():
    from foldjet_engine.kernels.integrations import pwa_fp8 as H, pwa_fp8_b200 as B

    old = (H.ENABLED, B.MIN_N)
    B._CACHES.clear()
    yield H, B
    H.ENABLED, B.MIN_N = old
    B._CACHES.clear()


def test_accuracy_stress_and_stats(tmp_path, route):
    H, B = route
    m, inp, step, S = _rig(tmp_path, "d_hidden=32;p_drop=0.15", 1280, 512)
    H.ENABLED = False
    y0 = _run(step)
    gold = _gold(m, inp, S)
    for stress in (1.0, 8.0):  # LN weights / biases x 8: the bound policy's scale follows the weights, nothing saturates
        with torch.no_grad():
            if stress != 1.0:
                m.ln_msa.weight.mul_(stress)
                m.ln_msa.bias.mul_(stress) if hasattr(m, "ln_msa") and m.ln_msa.bias is not None else None
        if stress != 1.0:
            H.ENABLED = False
            y0 = _run(step)
            gold = _gold(m, inp, S)
        H.ENABLED, B.MIN_N = True, None
        B._CACHES.clear()
        B.STATS.update({k: 0 for k in B.STATS})
        y1 = _run(step)
        e0, e1 = _err(y0, gold), _err(y1, gold)
        assert B.STATS["calls"] == 1, B.STATS
        assert torch.isfinite(y1).all() and not torch.equal(y0, y1)
        assert e1[1] < 0.02 and e1[0] < 0.02, (stress, e0, e1)  # H100 route: judge 0.30-0.37 % (bf16 0.28 %)
        print("PWA stress", stress, "bf16 route (rms, max):", e0, "e4m3:", e1)


def test_off_and_below_gate_are_bit_equal(tmp_path, route):
    H, B = route
    m, inp, step, S = _rig(tmp_path, "d_hidden=32;p_drop=0.15", 512, 512)  # N^2 S below the gate
    H.ENABLED = False
    y0 = _run(step)
    H.ENABLED, B.MIN_N = True, None
    B.STATS.update({k: 0 for k in B.STATS})
    y1 = _run(step)
    assert B.STATS["calls"] == 0 and torch.equal(y0, y1)
    m8, inp8, step8, _ = _rig(tmp_path, "d_hidden=8;p_drop=0.25", 2048, 256)  # head dim 8: never served
    H.ENABLED = False
    z0 = _run(step8)
    H.ENABLED = True
    z1 = _run(step8)
    assert torch.equal(z0, z1)


def test_graph_equals_eager(tmp_path, route):
    from foldjet_engine.kernels import _capture

    H, B = route
    m, inp, step, S = _rig(tmp_path, "d_hidden=32;p_drop=0.15", 1280, 512)
    H.ENABLED, B.MIN_N = True, None
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


# ---------------------------------------------------------------------------------------------------- wide builds: the fused e4m3 v (``ln_vg_w_f8``, agent G)
WIDE_CASES = [  # (label fragment, N, depth): each above the build's work gate (MIN_WORK_WIDE)
    ("d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_drop=0.25", 1536, 1024),
    ("d_msa=128;d_pair=256;n_head=8;d_hidden=16;p_drop=0.25", 1280, 1024),
    ("d_msa=64;d_pair=256;n_head=8;d_hidden=32;p_drop=0.25", 768, 1024),
    ("d_msa=256;d_pair=512;n_head=8;d_hidden=32;p_drop=0.25", 768, 1024),
]


@pytest.mark.parametrize("dims,L,D", WIDE_CASES)
def test_wide_fused_accuracy_and_stress(tmp_path, route, dims, L, D):
    H, B = route
    m, inp, step, S = _rig(tmp_path, dims, L, D)
    for stress in (1.0, 8.0):  # LN weights / biases x 8: the bound policy's scale follows the weights
        with torch.no_grad():
            if stress != 1.0:
                m.ln_msa.weight.mul_(stress)
                if m.ln_msa.bias is not None:
                    m.ln_msa.bias.mul_(stress)
        H.ENABLED, B.MIN_N = False, None
        y0 = _run(step)
        gold = _gold(m, inp, S)
        H.ENABLED = True
        B._CACHES.clear()
        B.STATS.update({k: 0 for k in B.STATS})
        y1 = _run(step)
        e0, e1 = _err(y0, gold), _err(y1, gold)
        assert B.STATS["calls"] == 1, B.STATS  # contract8 ran (the fused wide route)
        assert torch.isfinite(y1).all() and not torch.equal(y0, y1)
        assert e1[1] < 0.02 and e1[0] < 0.02, (dims, stress, e0, e1)
        print("PWA wide", dims, L, D, "stress", stress, "bf16 route (rms, max):", e0, "e4m3:", e1)


def test_wide_below_gate_is_bit_equal(tmp_path, route):
    H, B = route
    m, inp, step, S = _rig(tmp_path, "d_msa=128;d_pair=256;n_head=8;d_hidden=8;p_drop=0.25", 1280, 512)  # 8 x 8 below its work gate
    H.ENABLED = False
    y0 = _run(step)
    H.ENABLED, B.MIN_N = True, None
    B.STATS.update({k: 0 for k in B.STATS})
    y1 = _run(step)
    assert B.STATS["calls"] == 0 and torch.equal(y0, y1)


def test_wide_graph_equals_eager(tmp_path, route):
    from foldjet_engine.kernels import _capture

    H, B = route
    m, inp, step, S = _rig(tmp_path, "d_msa=128;d_pair=256;n_head=8;d_hidden=16;p_drop=0.25", 1280, 1024)
    H.ENABLED, B.MIN_N = True, None
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


# ---------------------------------------------------------------------------------------------------- input stress {x scale} x {offset} (agent G)
@pytest.mark.parametrize("xs,off", [(1e-3, 0.0), (40.0, 0.0), (1.0, 25.0), (40.0, 25.0)])
@pytest.mark.parametrize("dims,L,D", [("d_msa=128;d_pair=256;n_head=8;d_hidden=16;p_drop=0.25", 1280, 1024), ("d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_drop=0.25", 1536, 1024)])
def test_wide_input_stress_scale_offset(tmp_path, route, xs, off, dims, L, D):
    """x scale {1e-3, 40} and offset 25 on the float inputs: finite, static bound scales cannot saturate, error against fp64 next to the bf16 route's."""
    H, B = route
    m, inp, step, S = _rig(tmp_path, dims, L, D)
    inp2 = dict(inp)
    for k, v in inp.items():
        if torch.is_tensor(v) and v.is_floating_point() and v.dim() == 4:
            inp2[k] = (v.float() * xs + off).to(v.dtype)
    from foldjet_engine.measure import config as C

    step2 = S.make_step(m, inp2, backend="cuda", train=False, autocast=None, seed=0, compile_mode=None)
    H.ENABLED, B.MIN_N = False, None
    y0 = _run(step2)
    gold = _gold(m, inp2, S)
    H.ENABLED = True
    B._CACHES.clear()
    B.STATS.update({k: 0 for k in B.STATS})
    y1 = _run(step2)
    e0, e1 = _err(y0, gold), _err(y1, gold)
    assert B.STATS["calls"] == 1, B.STATS
    assert torch.isfinite(y1).all()
    print("PWA input stress", dims[:40], "x", xs, "off", off, "bf16 route (rms, max):", e0, "e4m3:", e1)
    assert e1[1] < 0.05 and e1[1] < 8 * e0[1] + 0.02, (e0, e1)


# ---------------------------------------------------------------------------------------------------- the grouped one-launch contraction (``pwa8_forward``, agent G)
@pytest.mark.parametrize("n,sc", [(128, 1024), (256, 8192), (768, 16384), (1280, 8192)])
def test_grouped_kernel_equals_per_head_gemms(route, n, sc):
    """pwa8_forward (all heads, one tcgen05 e4m3 launch) == the eight per-head cuBLASLt `_scaled_mm`s bit for bit (same fp32 accumulation, the scale applied before the bf16 rounding)."""
    H, B = route
    E4 = torch.float8_e4m3fn
    torch.manual_seed(2)
    w8 = (torch.rand(8, n, n, device="cuda") * 2).to(E4)
    v8 = torch.randn(8, n, sc, device="cuda").to(E4)
    sw, sv = torch.tensor(1.0 / 256.0, device="cuda"), torch.tensor(0.25, device="cuda")
    dq = (sw * sv).reshape(1).contiguous()
    o_g = torch.empty(8, n, sc, device="cuda", dtype=torch.bfloat16)
    B._ext_grouped().pwa8_forward(w8, v8, o_g, dq)
    o_h = torch.empty_like(o_g)
    for k in range(8):
        torch._scaled_mm(w8[k], v8[k], scale_a=sw, scale_b=sv, out_dtype=torch.bfloat16, out=o_h[k])
    torch.cuda.synchronize()
    assert torch.equal(o_g, o_h)


def test_grouped_route_equals_streams_route(tmp_path, route):
    H, B = route
    m, inp, step, S = _rig(tmp_path, "d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_drop=0.25", 1280, 1024)  # per-head work 1.3e10: the grouped kernel serves
    H.ENABLED, B.MIN_N = True, 0
    old = (B.GROUPED, B.HEAD_STREAMS)
    try:
        B.GROUPED = True
        y_g = _run(step)
        B.GROUPED = False
        y_s = _run(step)
        B.GROUPED, B.HEAD_STREAMS = False, 1
        y_1 = _run(step)
    finally:
        B.GROUPED, B.HEAD_STREAMS = old
    assert torch.equal(y_g, y_s) and torch.equal(y_g, y_1)

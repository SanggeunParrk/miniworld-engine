"""The fp32 path of the fused SWA atom DiT block on B200 (``kernels/swa_dit/cuda/sm100/tf32_fwd``): every forward stage on the
hand-written TF32 tensor-core kernels, no Triton. Checked against an fp64 reference (a banded twin of ``reference.py``, so A = 48 at
S = 4096 fits) and against the Triton fp32 path on the same inputs (``MINIWORLD_SWA_DIT_TF32=0``): the output no worse than Triton's,
the training saves (all fp32) each close to their fp64 meaning, the kernels that ran (read from a CUDA-graph capture: the
three TF32 kernels, no Triton kernel), the op's fake against the launch, CUDA-graph capture, no register spills, the hoisted modulation on mod_fwd_tf32, and the
Triton fallback off the 128-atom grid, and ``torch.compile(fullgraph=True)`` of an SWADiTBlock stack (inference and a
training step) against eager. ``-k fwd`` selects the forward tests; ``test_bwd_*`` needs ``cuda/sm100/tf32_bwd``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.kernels.swa_dit import dispatch
from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_fwd
from miniworld_engine.kernels.swa_dit.interface import FP32_EPS, swa_dit_block, swa_dit_hoist_modulation
from miniworld_engine.kernels.swa_dit.reference import swa_dit_block_reference, swa_dit_hoist_modulation_reference

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100)"),
]
DEV, F32, F64 = "cuda", torch.float32, torch.float64
C, H, D, HIDDEN, HW = 128, 4, 32, 256, 64
SAVES = ("out", "Qh", "Kh", "Vh", "G", "O", "lse", "q1", "X", "PQ", "PK", "Att", "Y", "FF")
TF32_KERNELS = ("swa_qkvg_fwd_tf32_sm100", "swa_attn_fwd_tf32_sm100", "swa_ffn_fwd_tf32_sm100")
SHAPES = [(1, 1024), (1, 4096), (5, 1024), (5, 4096), (48, 1024), (48, 4096)]   # (A, S) at B = 1: inference A 1 / 5, training A 48


def _case(a, b, s, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(DEV, F32)

    n = a * b
    leaves = [r(n, s, C), r(b, s, C), r(6 * C, C, scale=0.05), r(3 * C, C, scale=C ** -0.5), r(C, C, scale=C ** -0.5),
              r(C, C, scale=C ** -0.5), r(2 * HIDDEN, C, scale=C ** -0.5), r(C, HIDDEN, scale=HIDDEN ** -0.5)]
    angles = torch.randn(b, s, C // H // 2, generator=g) * 3.0
    lengths = [s, max(1, s - 17), 5, s - 1] * n                       # ragged, front-packed, one row shorter than the window
    return leaves, angles.cos().to(DEV), angles.sin().to(DEV), torch.tensor(lengths[:n], dtype=torch.int32, device=DEV)


def _mod32(c_base, wmod):
    """The block's modulation, exact (fp64) and handed to every path in fp32: the block comparisons leave the modulation GEMM out."""
    return swa_dit_hoist_modulation_reference(c_base.double(), wmod.double()).float()


def _rel(a, e):
    a, e = a.double(), e.double()
    return float((a - e).norm() / e.norm().clamp_min(1e-30))


def _rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + FP32_EPS)


def _ref64(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, b, hw=HW):
    """``reference.swa_dit_block_reference`` in fp64 with the window attention computed per 128-query block (keys [i0 - hw, i0 + 128 +
    hw)), returning the intermediates the TF32 forward saves under their names (+ ``valid`` [N, S], the rows below seqused)."""
    q, mod, cos, sin, wqkv, wg, wo, wu, wd = (t.to(F64) for t in (q, mod, cos, sin, wqkv, wg, wo, wu, wd))
    n, s, _ = q.shape
    a = n // b
    m = n * s

    def per_row(t):
        return t.reshape(b, s, t.shape[-1]).repeat(a, 1, 1)

    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = per_row(mod).chunk(6, dim=-1)
    cs, sn = per_row(cos).unsqueeze(2), per_row(sin).unsqueeze(2)

    def rope(t):
        t1, t2 = t[..., : D // 2], t[..., D // 2:]
        return torch.cat([t1 * cs - t2 * sn, t2 * cs + t1 * sn], dim=-1)

    x = _rms(q) * (1 + scale_a) + shift_a
    pq, pk, pv = (x @ wqkv.t()).chunk(3, dim=-1)
    g = x @ wg.t()
    qh = rope(_rms(pq.reshape(n, s, H, D))).transpose(1, 2).contiguous()      # [n, H, s, D]
    kh = rope(_rms(pk.reshape(n, s, H, D))).transpose(1, 2).contiguous()
    vh = pv.reshape(n, s, H, D).transpose(1, 2).contiguous()
    pos = torch.arange(s, device=q.device)
    valid = pos.view(1, s) < seqused.to(torch.long).view(n, 1)
    o = torch.zeros_like(qh)
    lse = torch.zeros(n, H, s, device=q.device, dtype=F64)
    for i0 in range(0, s, 128):
        j0, j1 = max(i0 - hw, 0), min(i0 + 128 + hw, s)
        sc = torch.einsum("nhid,nhjd->nhij", qh[:, :, i0:i0 + 128], kh[:, :, j0:j1]) * D ** -0.5
        band = (pos[i0:i0 + 128].view(-1, 1) - pos[j0:j1].view(1, -1)).abs() <= hw
        sc = sc.masked_fill(~(band.view(1, 1, *band.shape) & valid[:, j0:j1].view(n, 1, 1, -1)), float("-inf"))
        lb = torch.logsumexp(sc, dim=-1)
        lb = torch.where(torch.isfinite(lb), lb, torch.zeros_like(lb))         # rows without a valid key: p = 0, lse 0
        o[:, :, i0:i0 + 128] = torch.exp(sc - lb.unsqueeze(-1)) @ vh[:, :, j0:j1]
        lse[:, :, i0:i0 + 128] = lb
    o = o * valid.view(n, 1, s, 1)
    orow = o.transpose(1, 2).reshape(n, s, C)
    att = (torch.sigmoid(g) * orow) @ wo.t()
    q1 = q + gate_a * att
    y = _rms(q1) * (1 + scale_f) + shift_f
    ffn = (F.silu(y @ wu[:HIDDEN].t()) * (y @ wu[HIDDEN:].t())) @ wd.t()
    out = q1 + gate_f * ffn
    rows = {"G": g, "O": orow, "q1": q1, "X": x, "PQ": pq, "PK": pk, "Att": att, "Y": y, "FF": ffn}
    return {"out": out, "Qh": qh, "Kh": kh, "Vh": vh, "lse": lse, "valid": valid, **{k: v.reshape(m, C) for k, v in rows.items()}}


def _profiled_names(fn, expected=()):
    """(fn's eager result, the kernels one call launches). The kernels come from a CUDA-graph capture of the call
    (``tests.cuda_graph_nodes``), not the profiler: in long pytest processes it drops the records of driver-API launches
    (the engine's PDL kernels) while keeping cuBLAS / ATen ones. ``expected`` is kept for the callers; it is not used."""
    from tests.cuda_graph_nodes import launched_kernels

    out = fn()
    return out, launched_kernels(fn)


def _no_triton(names) -> list[str]:
    return [nm for nm in names if "triton" in nm.lower() or nm.startswith("_swa_")]


@pytest.fixture
def tf32_spy(monkeypatch):
    calls = []
    for name in ("block_fwd_tf32", "mod_fwd_tf32"):
        orig = getattr(tf32_fwd, name)
        monkeypatch.setattr(tf32_fwd, name, (lambda o, nm: (lambda *a, **k: calls.append(nm) or o(*a, **k)))(orig, name))
    return calls


def test_fwd_kernels_build_without_spills():
    stats = tf32_fwd.kernels_tf32(torch.cuda.current_device()).stats()
    print("TF32 forward kernels (registers, local bytes):", stats)
    for name, (regs, lmem) in stats.items():
        assert lmem == 0, f"{name} spills: {lmem} B of local memory per thread at {regs} registers"


def test_fwd_banded_reference_matches_the_dense_one():
    leaves, cos, sin, seqused = _case(3, 1, 384, seed=7)
    q, c_base, wmod, *w = leaves
    mod = _mod32(c_base, wmod)
    dense = swa_dit_block_reference(*(t.double() for t in (q, mod, cos, sin)), seqused, *(t.double() for t in w), 1, half_window=HW)
    assert _rel(_ref64(q, mod, cos, sin, seqused, *w, 1)["out"], dense) < 1e-12


@pytest.mark.parametrize(("b", "s"), [(1, 1024), (1, 4096), (3, 384)])
def test_fwd_modulation_runs_on_tf32(b, s, tf32_spy):
    leaves, _, _, _ = _case(1, b, s, seed=3)
    c_base, wmod = leaves[1], leaves[2]
    with torch.no_grad():
        got = swa_dit_hoist_modulation(c_base, wmod)
    assert tf32_spy == ["mod_fwd_tf32"], tf32_spy
    truth = swa_dit_hoist_modulation_reference(c_base.double(), wmod.double())
    e = _rel(got, truth)
    print(f"modulation B{b} S{s}: TF32 {e:.2e}")
    assert got.dtype == F32 and got.shape == (b * s, 6 * C) and torch.isfinite(got).all()
    assert e < 1e-3, f"modulation TF32 {e:.2e}"


@pytest.mark.parametrize(("a", "s"), SHAPES)
def test_fwd_inference_matches_the_reference(a, s, tf32_spy, monkeypatch):
    b = 1
    leaves, cos, sin, seqused = _case(a, b, s, seed=1)
    q, c_base, wmod, *w = leaves
    mod = _mod32(c_base, wmod)
    ref = _ref64(q, mod, cos, sin, seqused, *w, b)["out"]
    with torch.no_grad():
        swa_dit_block(q, mod, cos, sin, seqused, *w, b, half_window=HW)            # build / load / weight forms outside the profile
        tf32_spy.clear()
        got, names = _profiled_names(lambda: swa_dit_block(q, mod, cos, sin, seqused, *w, b, half_window=HW), TF32_KERNELS)
        assert tf32_spy and set(tf32_spy) == {"block_fwd_tf32"}, tf32_spy   # the eager call, its warm-up and its capture
        missing = [k for k in TF32_KERNELS if not any(k in nm for nm in names)]
        no_triton = _no_triton(names)
        monkeypatch.setenv("MINIWORLD_SWA_DIT_TF32", "0")
        tf32_spy.clear()
        triton = swa_dit_block(q, mod, cos, sin, seqused, *w, b, half_window=HW)
        assert tf32_spy == [], "MINIWORLD_SWA_DIT_TF32=0 did not keep the Triton path"
    assert got.dtype == F32 and got.shape == q.shape and torch.isfinite(got).all()
    ef, et = _rel(got, ref), _rel(triton, ref)
    print(f"inference A{a} S{s}: TF32 {ef:.2e} Triton fp32 {et:.2e}")
    # plain TF32 everywhere (the Triton path runs its FFN as tf32x3): ~1.2x Triton's error is the expected price
    assert ef < 5e-4, f"TF32 {ef:.2e} against fp64"
    assert ef <= 1.5 * et + 1e-5, f"TF32 {ef:.2e} vs Triton fp32 {et:.2e}"
    # the kernels that ran (checked after the accuracy, so a profiler hiccup cannot hide an accuracy result)
    assert no_triton == [], no_triton
    assert missing == [], f"TF32 kernels absent from the profile after retries: {missing}; saw {sorted(names)}"


@pytest.mark.parametrize(("a", "s"), SHAPES)
def test_fwd_training_saves_match_the_reference(a, s):
    b = 1
    leaves, cos, sin, seqused = _case(a, b, s, seed=2)
    q, c_base, wmod, *w = leaves
    mod = _mod32(c_base, wmod)
    cs, sn = cos.reshape(-1, D // 2).contiguous(), sin.reshape(-1, D // 2).contiguous()
    ref = _ref64(q, mod, cos, sin, seqused, *w, b)
    with torch.no_grad():
        saved = tf32_fwd.block_fwd_tf32(q, mod, cs, sn, seqused, *w, b, True)
        alone = tf32_fwd.block_fwd_tf32(q, mod, cs, sn, seqused, *w, b, False)
    n, m = a * b, a * b * s
    shapes = {"out": (n, s, C), "Qh": (n, H, s, D), "Kh": (n, H, s, D), "Vh": (n, H, s, D), "lse": (n, H, s)}
    assert len(saved) == len(SAVES) and len(alone) == 1
    torch.testing.assert_close(saved[0], alone[0], atol=0, rtol=0)         # the saves do not change the result
    valid = ref["valid"]
    worst = []
    for name, t in zip(SAVES, saved, strict=True):
        assert t.dtype == F32 and tuple(t.shape) == shapes.get(name, (m, C)) and t.is_contiguous(), name
        assert torch.isfinite(t).all(), name
        if name == "lse":                                                  # rows below seqused (padding rows: a convention)
            e = _rel(t[valid.view(n, 1, s).expand(n, H, s)], ref[name][valid.view(n, 1, s).expand(n, H, s)])
        else:
            e = _rel(t, ref[name])
        worst.append((e, name))
        assert e < 2e-3, f"{name}: {e:.2e} against fp64"
    assert not saved[SAVES.index("O")].view(n, s, C)[~valid].any(), "the attention output of padding rows is not zero"
    x = saved[SAVES.index("X")]
    assert torch.equal(x, tf32_fwd._round_tf32(x)), "X is not the TF32-rounded MMA operand"
    print(f"saves A{a} S{s} worst:", sorted(worst, reverse=True)[:3])


@pytest.mark.parametrize("save", [False, True])
def test_fwd_fake_matches_the_launch(save):
    leaves, cos, sin, seqused = _case(5, 1, 256, seed=4)
    q, c_base, wmod, *w = leaves
    args = (q, _mod32(c_base, wmod), cos.reshape(-1, D // 2).contiguous(), sin.reshape(-1, D // 2).contiguous(), seqused, *w, 1, HW,
            FP32_EPS, save)
    with torch.no_grad():
        real = dispatch.swa_dit_block_fwd(*args)
    fake = dispatch._swa_dit_block_fwd_fake(*args)
    assert [(tuple(t.shape), t.dtype) for t in real] == [(tuple(t.shape), t.dtype) for t in fake]
    if save and dispatch._tf32_ready(q.device, True):
        assert real[1].dtype == F32, "a training forward with the TF32 backward available did not take the TF32 path"


def test_fwd_inference_is_cuda_graph_capturable():
    leaves, cos, sin, seqused = _case(5, 1, 256, seed=5)
    q, c_base, wmod, *w = leaves

    def fused():
        return swa_dit_block(q, swa_dit_hoist_modulation(c_base, wmod), cos, sin, seqused, *w, 1, half_window=HW)

    with torch.no_grad():
        eager = fused()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fused()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fused()
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, atol=0, rtol=0)


def test_fwd_off_the_128_grid_keeps_the_triton_path(tf32_spy):
    leaves, cos, sin, seqused = _case(2, 1, 200, seed=6)
    q, c_base, wmod, *w = leaves
    with torch.no_grad():
        out = swa_dit_block(q, _mod32(c_base, wmod), cos, sin, seqused, *w, 1, half_window=HW)
    assert tf32_spy == [] and torch.isfinite(out).all()


# ------------------------------------------------------------------------------------------------ forward + backward (needs tf32_bwd)
NAMES = ("out", "q", "c_base", "w_adaln", "w_qkv", "w_gate", "w_out", "w_up", "w_down")
WEIGHT_GRADS = ("w_adaln", "w_qkv", "w_gate", "w_out", "w_up", "w_down")


def _train(fn, leaves, dy, *args):
    live = [t.detach().clone().requires_grad_() for t in leaves]
    out = fn(live, *args)
    out.backward(dy.to(out.dtype))
    return [out.detach(), *(t.grad for t in live)]


def _fused(leaves, cos, sin, seqused, b):
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    return swa_dit_block(q, swa_dit_hoist_modulation(c_base, wmod), cos, sin, seqused, wqkv, wg, wo, wu, wd, b, half_window=HW)


def _reference(leaves, cos, sin, seqused, b):
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    mod = swa_dit_hoist_modulation_reference(c_base, wmod)
    return swa_dit_block_reference(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, b, half_window=HW)


@pytest.mark.parametrize(("a", "b", "s"), [(3, 2, 256), (48, 1, 256), (48, 1, 512)])
def test_bwd_training_matches_the_triton_path(a, b, s, tf32_spy, monkeypatch):
    leaves, cos, sin, seqused = _case(a, b, s)
    if not dispatch._tf32_ready(leaves[0].device, True):
        pytest.skip("cuda/sm100/tf32_bwd is not available")
    dy = torch.randn(a * b, s, C, device=DEV)
    truth = _train(_reference, [t.double() for t in leaves], dy.double(), cos.double(), sin.double(), seqused, b)
    fused = _train(_fused, leaves, dy, cos, sin, seqused, b)
    assert "block_fwd_tf32" in tf32_spy and "mod_fwd_tf32" in tf32_spy, tf32_spy
    monkeypatch.setenv("MINIWORLD_SWA_DIT_TF32", "0")
    tf32_spy.clear()
    triton = _train(_fused, leaves, dy, cos, sin, seqused, b)
    assert tf32_spy == [], "MINIWORLD_SWA_DIT_TF32=0 did not keep the Triton path"
    worst = []
    for name, f, t, e in zip(NAMES, fused, triton, truth, strict=True):
        ef, et = _rel(f, e), _rel(t, e)
        worst.append((ef / max(et, 1e-9), name, ef, et))
        assert torch.isfinite(f).all() and f.dtype == t.dtype, name
        if name in WEIGHT_GRADS:                                           # TF32 GEMMs (unit roundoff 4.9e-4) vs Triton's IEEE fp32 ones
            assert ef < 1e-3 and ef <= max(1.5 * et, 1e-3), f"{name}: TF32 {ef:.2e} vs Triton fp32 {et:.2e}"
        else:
            assert ef < 1.5 * et + 1e-4, f"{name}: TF32 {ef:.2e} vs Triton fp32 {et:.2e}"
    print(f"training A{a} B{b} S{s} worst:", sorted(worst, reverse=True)[:3])


# ------------------------------------------------------------------------------------------------ torch.compile(fullgraph=True)
def _stack(n_blocks, seed):
    """n_blocks fp32 SWADiTBlocks on the engine implementation, the adaLN projection drawn (zero-init would hide both branches)."""
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.swa_dit import SWADiTBlock

    torch.manual_seed(seed)
    blocks = torch.nn.ModuleList(SWADiTBlock(C, C, H, half_window=HW, implementation=ImplementationType.MINIWORLD)
                                 for _ in range(n_blocks)).to(DEV, F32)
    with torch.no_grad():
        for blk in blocks:
            blk.adaln_modulation[1].weight.normal_(std=0.05)
    return blocks


def _stack_inputs(a, s, seed):
    """x [A, S, C], the augment-invariant conditioning [1, S, C] and the attention params of front-packed ragged rows."""
    from miniworld_engine.modules.swa_atom_attention import build_attention_params

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(a, s, C, generator=g).to(DEV)
    c_base = torch.randn(1, s, C, generator=g).to(DEV)
    angles = torch.randn(1, s, D // 2, generator=g).to(DEV) * 3.0
    lengths = torch.tensor(([s, s - 17, 5, s - 1] * a)[:a], device=DEV)
    valid = torch.arange(s, device=DEV).view(1, s) < lengths.view(a, 1)
    return x, c_base, build_attention_params(angles.cos(), angles.sin(), valid, num_aug=a)


def _run_stack(blocks, x, c_base, ap):
    for blk in blocks:
        x = blk.forward_hoisted(x, c_base, ap)
    return x


def _tf32_names(names):
    return {k for k in (*TF32_KERNELS, "swa_mod_fwd_tf32_sm100") if any(k in nm for nm in names)}


def test_fwd_compile_fullgraph_inference_matches_eager(tf32_spy):
    blocks = _stack(2, seed=11)
    x, c_base, ap = _stack_inputs(5, 256, seed=12)
    torch._dynamo.reset()
    compiled = torch.compile(_run_stack, fullgraph=True)
    with torch.no_grad():
        eager = _run_stack(blocks, x, c_base, ap)
        assert set(tf32_spy) == {"block_fwd_tf32", "mod_fwd_tf32"}, tf32_spy
        tf32_spy.clear()
        got = compiled(blocks, x, c_base, ap)
        assert set(tf32_spy) == {"block_fwd_tf32", "mod_fwd_tf32"}, tf32_spy
        names = launched_kernels_of(lambda: compiled(blocks, x, c_base, ap))
    torch.testing.assert_close(got, eager, atol=1e-5, rtol=1e-5)
    assert _tf32_names(names) == {*TF32_KERNELS, "swa_mod_fwd_tf32_sm100"}, names
    assert _no_triton(names) == [], _no_triton(names)


def test_fwd_compile_fullgraph_training_matches_eager(tf32_spy, monkeypatch):
    x, c_base, ap = _stack_inputs(5, 256, seed=14)
    if not dispatch._tf32_ready(x.device, True):
        pytest.skip("cuda/sm100/tf32_bwd is not available")
    import copy

    from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_bwd

    bwd_calls = []
    for name in ("block_bwd_tf32", "mod_bwd_tf32"):
        orig = getattr(tf32_bwd, name)
        monkeypatch.setattr(tf32_bwd, name, (lambda o, nm: (lambda *a, **k: bwd_calls.append(nm) or o(*a, **k)))(orig, name))
    blocks = _stack(2, seed=13)
    twin = copy.deepcopy(blocks)
    dy = torch.randn_like(x)

    def step(mods, fn):
        xx, cc = x.clone().requires_grad_(), c_base.clone().requires_grad_()
        for p in mods.parameters():
            p.grad = None
        out = fn(mods, xx, cc, ap)
        out.backward(dy)
        return [out.detach(), xx.grad, cc.grad, *(p.grad for p in mods.parameters())]

    torch._dynamo.reset()
    compiled = torch.compile(_run_stack, fullgraph=True)
    eager = step(blocks, _run_stack)
    tf32_spy.clear()
    bwd_calls.clear()
    got = step(twin, compiled)
    assert set(tf32_spy) == {"block_fwd_tf32", "mod_fwd_tf32"}, tf32_spy
    assert set(bwd_calls) == {"block_bwd_tf32", "mod_bwd_tf32"}, bwd_calls
    for i, (g, e) in enumerate(zip(got, eager, strict=True)):
        assert g is not None and e is not None, i
        torch.testing.assert_close(g, e, atol=1e-5, rtol=1e-5, msg=lambda m, i=i: f"output / gradient #{i}: {m}")
    # the training forward's kernels (with the autograd graph recorded), from a CUDA-graph capture of the compiled call
    xx, cc = x.clone().requires_grad_(), c_base.clone().requires_grad_()
    names = launched_kernels_of(lambda: compiled(twin, xx, cc, ap))
    assert _tf32_names(names) == {*TF32_KERNELS, "swa_mod_fwd_tf32_sm100"}, names
    assert _no_triton(names) == [], _no_triton(names)


def launched_kernels_of(fn):
    from tests.cuda_graph_nodes import launched_kernels

    return launched_kernels(fn)

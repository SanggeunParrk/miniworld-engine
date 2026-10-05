"""The fused SWA atom DiT block on A100 (``kernels/swa_dit/cuda/sm80``): the forward stages (qkvg, window attention, out-projection + FFN) and the backward stages (FFN, out-projection,
window attention, qkvg; the row stages for a conditioning per sample, for one shared by a multiple of 16 samples and, with the modulation expanded to a row per sample, for any other A) on the sm_80 kernels, the adaLN modulation (``mod_fwd`` / ``mod_bwd``), the output
and every gradient (q, the five block weights, and the conditioning and adaLN weight through the hoisted modulation) no worse against the fp32 reference than the Triton path's, the switch that
keeps the Triton stages, CUDA-graph capture, and the atom counts that are not a multiple of 16 (a ragged tail tile)."""

from __future__ import annotations

import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.swa_dit.cuda import sm80
from miniworld_engine.kernels.swa_dit.interface import (
    swa_dit_block,
    swa_dit_hoist_modulation,
)
from miniworld_engine.kernels.swa_dit.reference import (
    swa_dit_block_reference,
    swa_dit_hoist_modulation_reference,
)
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.swa_atom_attention import build_attention_params
from miniworld_engine.modules.swa_dit import SWADiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="A100 (sm_80)"),
]
DEV, BF = "cuda", torch.bfloat16
C, H, HIDDEN, HW = 128, 4, 256, 64


def _case(a, b, s, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(DEV, BF)

    n = a * b
    leaves = [r(n, s, C), r(b, s, C), r(6 * C, C, scale=0.05), r(3 * C, C, scale=C ** -0.5), r(C, C, scale=C ** -0.5),
              r(C, C, scale=C ** -0.5), r(2 * HIDDEN, C, scale=C ** -0.5), r(C, HIDDEN, scale=HIDDEN ** -0.5)]
    angles = torch.randn(b, s, C // H // 2, generator=g) * 3.0
    lengths = [s, max(1, s - 17), 5, s - 1] * n                       # ragged, front-packed, one row shorter than the window
    return leaves, angles.cos().to(DEV), angles.sin().to(DEV), torch.tensor(lengths[:n], dtype=torch.int32, device=DEV)


def _fused(leaves, cos, sin, seqused, b):
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    return swa_dit_block(q, swa_dit_hoist_modulation(c_base, wmod), cos, sin, seqused, wqkv, wg, wo, wu, wd, b, half_window=HW)


def _reference(leaves, cos, sin, seqused, b):
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    mod = swa_dit_hoist_modulation_reference(c_base, wmod)
    return swa_dit_block_reference(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, b, half_window=HW)


def _train(fn, leaves, dy, *args):
    live = [t.detach().clone().requires_grad_() for t in leaves]
    out = fn(live, *args)
    out.backward(dy.to(out.dtype))
    return [out.detach(), *(t.grad for t in live)]


def _rel(a, e):
    a, e = a.double(), e.double()
    return float((a - e).norm() / e.norm().clamp_min(1e-30))


NAMES = ("out", "q", "c_base", "w_adaln", "w_qkv", "w_gate", "w_out", "w_up", "w_down")


#: the stage functions of ``sm80`` the tests watch (a name here is recorded when the stage runs)
STAGES = ("block_fwd", "ffn_bwd", "oproj_bwd", "qkvg_bwd", "attn_bwd")


@pytest.fixture
def spy(monkeypatch):
    calls = []
    for name in STAGES:
        orig = getattr(sm80, name)
        monkeypatch.setattr(sm80, name, (lambda o, n: (lambda *a, **k: calls.append(n) or o(*a, **k)))(orig, name))
    return calls


# (A, B, S): A = 1 (the conditioning per sample) / 16 / 32 / 48 (shared by a multiple of 16 samples) or 3 / 5 / 8 (the modulation expanded to one row per sample, its gradient summed), B = 1 / 2 / 3 / 4,
# S off the 16 grid (200) and on it: every one runs the five sm_80 stages
@pytest.mark.parametrize(("a", "b", "s"), [(3, 2, 256), (8, 2, 128), (5, 1, 200), (48, 1, 256), (16, 2, 200), (32, 1, 130), (1, 4, 256), (1, 3, 200), (1, 2, 1040), (7, 3, 130)])
def test_training_matches_the_triton_path(a, b, s, spy, monkeypatch):
    leaves, cos, sin, seqused = _case(a, b, s)
    dy = torch.randn(a * b, s, C, device=DEV)
    truth = _train(_reference, [t.float() for t in leaves], dy, cos, sin, seqused, b)
    fused = _train(_fused, leaves, dy, cos, sin, seqused, b)
    assert sorted(set(spy)) == ["attn_bwd", "block_fwd", "ffn_bwd", "oproj_bwd", "qkvg_bwd"], spy
    monkeypatch.setenv("MINIWORLD_SWA_DIT_SM80", "0")
    spy.clear()
    triton = _train(_fused, leaves, dy, cos, sin, seqused, b)
    assert spy == [], "the switch did not keep the Triton path"
    worst = []
    for name, f, t, e in zip(NAMES, fused, triton, truth, strict=True):
        ef, et = _rel(f, e), _rel(t, e)
        worst.append((ef / max(et, 1e-6), name, ef, et))
        assert torch.isfinite(f).all(), name
        assert f.dtype == t.dtype, name
        assert ef < 1.5 * et + 3e-3, f"{name}: sm_80 {ef:.2e} vs Triton {et:.2e}"
    print(f"training A{a} B{b} S{s} worst:", sorted(worst, reverse=True)[:2])


@pytest.mark.parametrize(("a", "b", "s"), [(1, 1, 256), (5, 1, 1024), (1, 3, 384), (5, 2, 200), (2, 1, 33)])
def test_inference_matches_the_triton_path(a, b, s, spy, monkeypatch):
    leaves, cos, sin, seqused = _case(a, b, s, seed=1)
    with torch.no_grad():
        truth = _reference([t.float() for t in leaves], cos, sin, seqused, b)
        fused = _fused(leaves, cos, sin, seqused, b)
        assert sorted(set(spy)) == ["block_fwd"], spy
        monkeypatch.setenv("MINIWORLD_SWA_DIT_SM80", "0")
        triton = _fused(leaves, cos, sin, seqused, b)
    ef, et = _rel(fused, truth), _rel(triton, truth)
    print(f"inference A{a} B{b} S{s}: sm_80 {ef:.2e} Triton {et:.2e}")
    assert ef < 1.5 * et + 3e-3, f"sm_80 {ef:.2e} vs Triton {et:.2e}"


def test_the_sm80_stages_reproduce_the_triton_stages_closely(monkeypatch):
    """The stages share their rounding points with the Triton ones: the block outputs differ only by the order the products are accumulated in (a 1-ulp flip of a few percent of the
    elements), a relative Frobenius difference far under the bf16 error against the fp32 truth (~3e-3)."""
    leaves, cos, sin, seqused = _case(5, 1, 1024, seed=4)
    with torch.no_grad():
        fused = _fused(leaves, cos, sin, seqused, 1)
        monkeypatch.setenv("MINIWORLD_SWA_DIT_SM80", "0")
        triton = _fused(leaves, cos, sin, seqused, 1)
    rel = _rel(fused, triton)
    differ = float((fused.float() != triton.float()).float().mean())
    print(f"sm_80 vs Triton: relative difference {rel:.2e}, {100 * differ:.2f} % of the elements differ")
    assert rel < 2e-3, f"relative difference {rel:.2e} ({100 * differ:.2f} % of the elements differ)"


def _mod_case(rows, seed):
    g = torch.Generator().manual_seed(seed)
    c = (torch.randn(rows, C, generator=g) * 2).to(DEV, BF)
    w = (torch.randn(6 * C, C, generator=g) * 0.05).to(DEV, BF)
    return c, w


# row counts around the 32-row tiles of the forward and the 64 / 128-row tiles of the backward, and a ragged multi-wave count
@pytest.mark.parametrize("rows", [1, 31, 33, 130, 3072, 15377])
def test_the_modulation_forward_is_the_fp32_gemm(rows, monkeypatch):
    c, w = _mod_case(rows, rows)
    calls = []
    orig = sm80.mod_fwd
    monkeypatch.setattr(sm80, "mod_fwd", lambda *a, **k: calls.append(1) or orig(*a, **k))
    with torch.no_grad():
        fused = swa_dit_hoist_modulation(c.reshape(1, rows, C), w)
        assert calls, "the sm_80 kernel did not serve the modulation"
        monkeypatch.setenv("MINIWORLD_SWA_DIT_SM80", "0")
        truth = swa_dit_hoist_modulation(c.reshape(1, rows, C), w)
    assert fused.dtype == torch.float32
    assert fused.shape == (rows, 6 * C)
    # the same exact bf16 x bf16 products; the sums differ in order, and ~1 in 1e4 of the silu roundings in the last bf16 place (fast exp)
    assert _rel(fused, truth) < 2e-4, _rel(fused, truth)


@pytest.mark.parametrize(("rows", "need_c"), [(1, True), (47, True), (130, False), (3072, True), (15377, True), (45001, True)])
def test_the_modulation_backward_matches_autograd(rows, need_c, monkeypatch):
    c, w = _mod_case(rows, rows + 1)
    g = torch.randn(rows, 6 * C, generator=torch.Generator().manual_seed(rows + 2)).to(DEV)

    def run():
        cc, ww = c.clone().requires_grad_(need_c), w.clone().requires_grad_()
        swa_dit_hoist_modulation(cc.reshape(1, rows, C), ww).backward(g)
        return cc.grad, ww.grad

    dc, dw = run()
    monkeypatch.setenv("MINIWORLD_SWA_DIT_SM80", "0")
    dc_ref, dw_ref = run()
    if need_c:
        assert dc.dtype == BF
        assert dc.shape == (rows, C)
        assert _rel(dc, dc_ref) < 2e-3, _rel(dc, dc_ref)
    else:
        assert dc is None
        assert dc_ref is None
    assert dw.dtype == BF
    assert dw.shape == (6 * C, C)
    assert _rel(dw, dw_ref) < 2e-3, _rel(dw, dw_ref)


def test_the_modulation_saves_the_bf16_silu_and_its_backward_is_deterministic():
    c, w = _mod_case(9001, 7)
    g = torch.randn(9001, 6 * C, device=DEV)
    _, act = sm80.mod_fwd(c, w, True)
    ref = torch.nn.functional.silu(c)
    assert act.dtype == BF
    assert (act != ref).float().mean() < 1e-3          # the framework's bf16 silu (fast exp: ~1 in 1e4 differ in the last place)
    first = sm80.mod_bwd(g, c, act, w)
    second = sm80.mod_bwd(g, c, act, w)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_inference_is_cuda_graph_capturable_and_deterministic():
    leaves, cos, sin, seqused = _case(5, 1, 256, seed=2)
    with torch.no_grad():
        eager = _fused(leaves, cos, sin, seqused, 1)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _fused(leaves, cos, sin, seqused, 1)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _fused(leaves, cos, sin, seqused, 1)
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, atol=0, rtol=0)


def _module_train(block, fn, x, c, ap, dy):
    block = copy.deepcopy(block)
    x, c = (t.detach().clone().requires_grad_() for t in (x, c))
    out = fn(block, x, c, ap)
    out.backward(dy)
    return out.detach(), [x.grad, c.grad, *(p.grad for p in block.parameters())]


@pytest.mark.skipif(settings.current().compile_wrap != "custom_op", reason="fullgraph needs the custom_op launches")
@pytest.mark.parametrize(("a", "b", "hoisted"), [(3, 2, False), (16, 1, True)])
def test_the_module_compiles_fullgraph_on_the_sm80_stages(a, b, hoisted, spy, monkeypatch):
    """``torch.compile(fullgraph=True)`` traces the module without a break (the kernel gate, the modulation autograd Function and the opaque stage ops), the compiled forward and backward run
    the sm_80 stages (the extension loads at trace time, outside the graph), and they agree with the eager run: only the plain torch ops around the opaque ones are Inductor's."""
    torch.manual_seed(11)
    block = SWADiTBlock(C, C, H, implementation=ImplementationType.MINIWORLD).to(DEV, BF)
    with torch.no_grad():
        block.adaln_modulation[1].weight.normal_(std=0.05)
    s = 200
    g = torch.Generator().manual_seed(5)
    x = torch.randn(a * b, s, C, generator=g).to(DEV, BF)
    c_base = torch.randn(b, s, C, generator=g).to(DEV, BF)
    angles = torch.randn(b, s, C // H // 2, generator=g).to(DEV) * 3.0
    lengths = torch.tensor([s, s - 9, 7][:a * b] + [s] * max(0, a * b - 3), device=DEV)
    valid = torch.arange(s, device=DEV)[None] < lengths[:, None]
    ap = build_attention_params(angles.cos(), angles.sin(), valid, a)
    c = c_base if hoisted else c_base.repeat(a, 1, 1)
    dy = torch.randn(a * b, s, C, device=DEV, dtype=BF)
    run = (lambda m, x_, c_, ap_: m.forward_hoisted(x_, c_, ap_)) if hoisted else (lambda m, x_, c_, ap_: m(x_, c_, ap_))
    build = (lambda m: torch.compile(m.forward_hoisted, fullgraph=True)) if hoisted else (lambda m: torch.compile(m, fullgraph=True))
    assert block.fused_refusal(x, c, ap) is None
    eager, eager_grads = _module_train(block, run, x, c, ap, dy)
    spy.clear()
    compiled, compiled_grads = _module_train(block, lambda m, x_, c_, ap_: build(m)(x_, c_, ap_), x, c, ap, dy)
    assert sorted(set(spy)) == ["attn_bwd", "block_fwd", "ffn_bwd", "oproj_bwd", "qkvg_bwd"], spy
    rel = [_rel(compiled, eager), *(_rel(got, want) for got, want in zip(compiled_grads, eager_grads, strict=True))]
    print(f"compiled vs eager (A{a} B{b} hoisted={hoisted}): out and gradients, relative Frobenius", [f"{r:.1e}" for r in rel])
    assert max(rel) < 1e-4, rel


def test_non_contiguous_weights_give_the_same_bits():
    """The stages read their weights contiguously: weights that arrive as transposed views (same values, other strides) are made contiguous, never refused."""
    leaves, cos, sin, seqused = _case(3, 2, 200, seed=6)
    strided = [t if i < 2 else t.t().contiguous().t() for i, t in enumerate(leaves)]
    assert not any(t.is_contiguous() for t in strided[2:])
    dy = torch.randn(6, 200, C, device=DEV)
    base = _train(_fused, leaves, dy, cos, sin, seqused, 2)
    other = _train(_fused, strided, dy, cos, sin, seqused, 2)
    for name, a, b in zip(NAMES, base, other, strict=True):
        torch.testing.assert_close(b, a, atol=0, rtol=0, msg=lambda m, name=name: f"{name}: {m}")


def test_fp32_and_other_widths_keep_the_triton_stages(spy):
    leaves, cos, sin, seqused = _case(2, 1, 128, seed=5)
    with torch.no_grad():
        _fused([t.float() for t in leaves], cos, sin, seqused, 1)
    assert spy == [], "fp32 operands must not take the bf16 kernels"

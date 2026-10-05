"""The sliding-window attention of ``modules/swa_atom_attention`` on A100 (``kernels/swa_dit`` ``swa_dit_window_attention`` on the sm_80 kernels of ``cuda/sm80``): the kernels against a dense fp32 band
attention on the module's own [N, S, H, D] layout (any strides: the slices of the fused qkv projection go in unchanged, and a head-major plane gives the same bits), the module's output and every
gradient (x and the three projections) against the module's PyTorch path, the gate, CUDA-graph capture, determinism and ``torch.compile(fullgraph=True)``."""

from __future__ import annotations

import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.swa_dit.cuda import sm80
from miniworld_engine.kernels.swa_dit.interface import swa_dit_window_attention
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.swa_atom_attention import (
    SWA3DRoPEAttention,
    build_attention_params,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="A100 (sm_80)"),
]
DEV, BF = "cuda", torch.bfloat16
C, H, D, HW = 128, 4, 32, 64


def _lengths(n, s):
    """Front-packed valid counts: full rows, a short one, a row shorter than the window, one with a single atom, an empty one."""
    return torch.tensor(([s, max(1, s - 17), 5, 1, 0, s - 1] * n)[:n], dtype=torch.int32, device=DEV)


def _band_reference(q, k, v, seqused, d_out=None):
    """Dense fp32 window attention over [N, S, H, D] (rows at or past seqused: output 0), and its autograd gradients for ``d_out``."""
    q, k, v = (t.detach().float().requires_grad_() for t in (q, k, v))
    s = q.shape[1]
    pos = torch.arange(s, device=q.device)
    valid = pos[None, :] < seqused[:, None]
    band = (pos[None, :, None] - pos[None, None, :]).abs() <= HW
    allowed = band & valid[:, :, None] & valid[:, None, :]
    scores = torch.einsum("nihd,njhd->nhij", q, k) * D ** -0.5
    scores = scores.masked_fill(~allowed[:, None], -1e30)                  # finite: a row without a valid key softmaxes to a uniform row that the validity mask below zeroes (no NaN in the backward)
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("nhij,njhd->nihd", p, v) * valid[:, :, None, None]
    if d_out is None:
        return out.detach()
    out.backward(d_out.float())
    return out.detach(), q.grad, k.grad, v.grad


def _rel(a, e):
    a, e = a.double(), e.double()
    return float((a - e).norm() / e.norm().clamp_min(1e-30))


@pytest.mark.parametrize(("n", "s"), [(1, 64), (2, 200), (3, 333), (4, 1040), (6, 130)])
def test_the_kernels_match_the_dense_band_attention(n, s):
    g = torch.Generator().manual_seed(n * 1000 + s)
    q, k, v, d_out = (torch.randn(n, s, H, D, generator=g).to(DEV, BF) for _ in range(4))
    seqused = _lengths(n, s)
    out, lse = sm80.window_fwd(q, k, v, seqused)
    truth, dq_t, dk_t, dv_t = _band_reference(q, k, v, seqused, d_out)
    dq, dk, dv = sm80.window_bwd(q, k, v, out, d_out, lse, seqused)
    errs = {"out": _rel(out, truth), "dq": _rel(dq, dq_t), "dk": _rel(dk, dk_t), "dv": _rel(dv, dv_t)}
    print(f"window kernels N{n} S{s}: relative Frobenius error against the dense fp32 band attention", {k_: f"{e:.1e}" for k_, e in errs.items()})
    assert max(errs.values()) < 8e-3, errs
    pad = torch.arange(s, device=DEV)[None, :] >= seqused[:, None]
    for name, t in (("out", out), ("dq", dq), ("dk", dk), ("dv", dv)):
        assert not t[pad].any(), f"{name}: rows at or past seqused must be exactly 0"


def test_strided_views_and_head_major_planes_give_the_same_bits():
    """q / k / v as slices of a fused [N, S, 3, H, D] projection (row stride 384), contiguous [N, S, H, D], and the fused block's head-major planes: one function of the data."""
    n, s = 3, 333
    g = torch.Generator().manual_seed(7)
    qkv = torch.randn(n, s, 3, H, D, generator=g).to(DEV, BF)
    q, k, v = qkv.unbind(2)
    seqused = _lengths(n, s)
    out0, lse0 = sm80.window_fwd(q, k, v, seqused)
    out1, lse1 = sm80.window_fwd(q.contiguous(), k.contiguous(), v.contiguous(), seqused)
    qh, kh, vh = (t.permute(0, 2, 1, 3).contiguous() for t in (q, k, v))           # head-major [N, H, S, D]
    o2, lse2 = sm80.attn_fwd(qh, kh, vh, seqused)
    torch.testing.assert_close(out1, out0, atol=0, rtol=0)
    torch.testing.assert_close(lse1, lse0, atol=0, rtol=0)
    torch.testing.assert_close(o2.view(n, s, H, D), out0, atol=0, rtol=0)
    torch.testing.assert_close(lse2, lse0, atol=0, rtol=0)
    d_out = torch.randn(n, s, H, D, generator=g).to(DEV, BF)
    g0 = sm80.window_bwd(q, k, v, out0, d_out, lse0, seqused)
    g1 = sm80.window_bwd(q.contiguous(), k.contiguous(), v.contiguous(), out0, d_out, lse0, seqused)
    dvv = (d_out.float() * out0.float()).sum(-1).permute(0, 2, 1).contiguous()
    g2 = sm80.attn_bwd(qh, kh, vh, d_out.view(n * s, C), lse0, dvv, seqused)
    for a, b, c in zip(g0, g1, g2, strict=True):
        torch.testing.assert_close(b, a, atol=0, rtol=0)
        torch.testing.assert_close(c.permute(0, 2, 1, 3), a, atol=1e-2, rtol=2e-2)   # D from torch's fp32 sum here: the same up to the sum's order (rare one-ulp bf16 flips)


def _module(implementation, dtype=BF):
    torch.manual_seed(3)
    return SWA3DRoPEAttention(C, H, HW, implementation=implementation).to(DEV, dtype)


def _case(n, s, dtype=BF, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, s, C, generator=g).to(DEV, dtype)
    angles = torch.randn(1, s, D // 2, generator=g).to(DEV) * 3.0
    valid = torch.arange(s, device=DEV)[None, :] < _lengths(n, s)[:, None]
    ap = build_attention_params(angles.cos(), angles.sin(), valid, n)
    dy = torch.randn(n, s, C, generator=g).to(DEV, dtype)
    return x, ap, dy


def _train(module, x, ap, dy):
    module = copy.deepcopy(module)
    x = x.detach().clone().requires_grad_()
    out = module(x, ap)
    out.backward(dy.to(out.dtype))
    return out.detach(), [x.grad, *(p.grad for p in module.parameters())]


@pytest.fixture
def spy(monkeypatch):
    calls = []
    for name in ("window_fwd", "window_bwd"):
        orig = getattr(sm80, name)
        monkeypatch.setattr(sm80, name, (lambda o, nm: (lambda *a, **kw: calls.append(nm) or o(*a, **kw)))(orig, name))
    return calls


@pytest.mark.parametrize(("n", "s"), [(2, 200), (3, 333), (4, 1040)])
def test_the_module_takes_the_kernels_and_matches_its_pytorch_path(n, s, spy):
    x, ap, dy = _case(n, s)
    ours = _module(ImplementationType.MINIWORLD)
    ref = _module(ImplementationType.PYTORCH)
    ref.load_state_dict(ours.state_dict())
    truth = _module(ImplementationType.PYTORCH, torch.float32)
    truth.load_state_dict({k_: t.float() for k_, t in ours.state_dict().items()})
    out_o, grads_o = _train(ours, x, ap, dy)
    assert sorted(set(spy)) == ["window_bwd", "window_fwd"], spy
    out_r, grads_r = _train(ref, x, ap, dy)
    out_t, grads_t = _train(truth, x.float(), ap, dy.float())
    names = ("x", "w_qkv", "w_gate", "w_out")
    worst = []
    for name, go, gr, gt in zip(("out", *names), (out_o, *grads_o), (out_r, *grads_r), (out_t, *grads_t), strict=True):
        eo, er = _rel(go, gt), _rel(gr, gt)
        worst.append((eo / max(er, 1e-6), name, eo, er))
        assert go.dtype == gr.dtype, name
        assert torch.isfinite(go).all(), name
        assert eo < 1.5 * er + 3e-3, f"{name}: sm_80 {eo:.2e} vs the module's PyTorch path {er:.2e} (both against fp32)"
    print(f"module N{n} S{s}: worst ratio to the PyTorch path", sorted(worst, reverse=True)[:2])


def test_the_gate_and_the_fp32_cast(monkeypatch):
    n, s = 2, 128
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(n, s, H, D, generator=g).to(DEV, BF) for _ in range(3))
    seqused = _lengths(n, s)
    assert swa_dit_window_attention(q, k, v, seqused, H, HW) is not None
    assert swa_dit_window_attention(q, k, v, seqused, 8, HW) is None                  # other head counts and windows are not served
    assert swa_dit_window_attention(q, k, v, seqused, H, 32) is None
    assert swa_dit_window_attention(q[..., :16], k[..., :16], v[..., :16], seqused, H, HW) is None
    out32 = swa_dit_window_attention(q.float(), k.float(), v.float(), seqused, H, HW)
    assert out32.dtype == torch.float32                                                # cast to bf16 inside, back to the caller's dtype
    monkeypatch.setenv("MINIWORLD_SWA_DIT_SM80", "0")
    assert swa_dit_window_attention(q, k, v, seqused, H, HW) is None


def test_inference_is_cuda_graph_capturable_and_deterministic():
    n, s = 3, 256
    x, ap, _ = _case(n, s, seed=2)
    module = _module(ImplementationType.MINIWORLD).eval()
    with torch.no_grad():
        eager = module(x, ap)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            module(x, ap)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = module(x, ap)
        graph.replay()
        torch.cuda.synchronize()
        again = module(x, ap)
    torch.testing.assert_close(out, eager, atol=0, rtol=0)
    torch.testing.assert_close(again, eager, atol=0, rtol=0)


@pytest.mark.skipif(settings.current().compile_wrap != "custom_op", reason="fullgraph needs the custom_op launches")
def test_the_module_compiles_fullgraph(spy):
    n, s = 3, 200
    x, ap, dy = _case(n, s)
    module = _module(ImplementationType.MINIWORLD)
    eager, eager_grads = _train(module, x, ap, dy)
    spy.clear()
    compiled_module = copy.deepcopy(module)
    xc = x.detach().clone().requires_grad_()
    out = torch.compile(compiled_module, fullgraph=True)(xc, ap)
    out.backward(dy)
    assert sorted(set(spy)) == ["window_bwd", "window_fwd"], spy
    got = [out.detach(), xc.grad, *(p.grad for p in compiled_module.parameters())]
    want = [eager, *eager_grads]
    rel = [_rel(a, b) for a, b in zip(got, want, strict=True)]
    print("compiled vs eager: out, dx, parameter gradients, relative Frobenius", [f"{r:.1e}" for r in rel])
    assert max(rel) < 1e-2, rel


def test_the_a100_path_needs_no_flash_install(spy, monkeypatch):
    """With no flash backend at all (``_flash_backend`` None: FlashAttention-2 does not import in this environment), the module's A100 path -- forward and backward -- still runs: the window
    attention is the hand-CUDA kernels', the output gate the hand-CUDA pass, nothing falls through to the flash code (which would raise)."""
    from miniworld_engine.modules.swa_atom_attention import module as swa_module

    monkeypatch.setattr(swa_module, "_flash_backend", lambda device=None: None)
    x, ap, dy = _case(3, 200)
    out, grads = _train(_module(ImplementationType.MINIWORLD), x, ap, dy)
    assert sorted(set(spy)) == ["window_bwd", "window_fwd"], spy
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("dtype", [BF, torch.float32])
def test_the_output_gate_is_one_hand_cuda_pass_and_matches_sigmoid_times_x(dtype, monkeypatch):
    from miniworld_engine.integrations import sigmoid_gate_sm80

    g = torch.Generator().manual_seed(5)
    gate = (torch.randn(3, 200, C, generator=g) * 2).to(DEV, dtype).requires_grad_()
    x = torch.randn(3, 200, C, generator=g).to(DEV, dtype).requires_grad_()
    dy = torch.randn(3, 200, C, generator=g).to(DEV, dtype)
    assert sigmoid_gate_sm80.serves(gate, x)
    out = sigmoid_gate_sm80.sigmoid_gate(gate, x)
    out.backward(dy)
    gt, xt = (t.detach().double().requires_grad_() for t in (gate, x))
    want = torch.sigmoid(gt) * xt
    want.backward(dy.double())
    tol = 1e-2 if dtype is BF else 1e-5                      # fp32: the sigmoid is one tanh.approx (2e-6 relative)
    for got, ref in ((out, want), (gate.grad, gt.grad), (x.grad, xt.grad)):
        assert got.dtype == dtype
        assert _rel(got, ref.detach()) < tol
    assert not sigmoid_gate_sm80.serves(gate, x.to(torch.float32 if dtype is BF else BF)), "mixed dtypes"
    assert not sigmoid_gate_sm80.serves(gate[..., :100], x[..., :100]), "a width that is not a multiple of 8"
    monkeypatch.setenv("MINIWORLD_SIGMOID_GATE_SM80", "0")
    assert not sigmoid_gate_sm80.serves(gate, x)


def test_the_module_takes_the_gate_pass_and_the_switch_keeps_the_triton_one(monkeypatch):
    from miniworld_engine.integrations import sigmoid_gate_sm80

    calls = []
    orig = sigmoid_gate_sm80.sigmoid_gate
    monkeypatch.setattr(sigmoid_gate_sm80, "sigmoid_gate", lambda *a, **k: calls.append(1) or orig(*a, **k))
    x, ap, dy = _case(2, 256)
    out_on, grads_on = _train(_module(ImplementationType.MINIWORLD), x, ap, dy)
    assert calls, "the module did not take the sm_80 gate"
    n = len(calls)
    monkeypatch.setenv("MINIWORLD_SIGMOID_GATE_SM80", "0")
    out_off, grads_off = _train(_module(ImplementationType.MINIWORLD), x, ap, dy)
    assert len(calls) == n, "the switch did not keep the old gate"
    assert _rel(out_on, out_off) < 2e-2
    for a, b in zip(grads_on, grads_off, strict=True):
        assert _rel(a, b) < 3e-2

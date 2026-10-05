"""The A100 hand-CUDA bias-only attention (kernels/bias_only_attention/cuda/sm80.py, the family's ``bias_only_attention`` door): ``softmax(bias) v`` over ``v`` [B, H, L, L, D] with the ``t``
axis sharing one set of weights, forward and the gradients of v and the bias against an fp64 PyTorch reference, no worse than the Triton kernels', ``torch.compile`` equal to eager, CUDA-graph
capture, the gate and the switch."""

import pytest
import torch

from miniworld_engine.kernels.bias_only_attention.cuda import sm80
from miniworld_engine.kernels.bias_only_attention.interface import (
    bias_only_attention,
    triton_bias_only_attention,
)
from miniworld_engine.kernels.bias_only_attention.reference import (
    bias_only_attention_pytorch as reference,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="A100 (sm_80)"),
]
BF = torch.bfloat16


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _case(b, h, length, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    v = torch.randn(b, h, length, length, d, device="cuda", generator=g)
    bias = torch.randn(b, h, length, length, device="cuda", generator=g) * 2
    dy = torch.randn(b, h, length, length, d, device="cuda", generator=g)
    return v, bias, dy


def _run(fn, v, bias, dy, dtype):
    leaves = [t.detach().to(dtype).requires_grad_() for t in (v, bias)]
    out = fn(*leaves)
    out.backward(dy.to(dtype))
    return [out.detach(), *(t.grad for t in leaves)]


def _truth(v, bias, dy):
    leaves = [t.detach().double().requires_grad_() for t in (v, bias)]
    out = reference(*leaves)
    out.backward(dy.double())
    return [out.detach(), *(t.grad for t in leaves)]


@pytest.mark.parametrize(("b", "h", "length", "d"), [(1, 4, 128, 32), (1, 4, 384, 32), (2, 3, 256, 48), (1, 2, 128, 64), (1, 8, 256, 32), (1, 1, 640, 48)])
def test_forward_and_gradients_match_the_reference_as_well_as_triton(b, h, length, d):
    args = _case(b, h, length, d)
    v, bias, _ = args
    assert sm80.serves(v.to(BF), bias.to(BF))
    truth = _truth(*args)
    got = _run(sm80.bias_only_attention_sm80, *args, BF)          # the CUDA path itself (the door keeps Triton for a training call at L = 128)
    triton = _run(triton_bias_only_attention, *args, BF)
    for name, g_, t_, e_ in zip(("out", "dv", "dbias"), got, triton, truth, strict=True):
        assert torch.isfinite(g_).all(), name
        assert g_.dtype is BF, name
        assert g_.shape == t_.shape, name
        assert _rel(g_, e_) <= 1.5 * _rel(t_, e_) + 2e-3, f"{name}: sm_80 {_rel(g_, e_):.2e} vs Triton {_rel(t_, e_):.2e}"


def test_compile_and_graph_replay_match_eager():
    v, bias, dy = _case(1, 4, 256, 32, 3)
    vb, bb = v.to(BF), bias.to(BF)
    eager = bias_only_attention(vb, bb)
    torch.testing.assert_close(torch.compile(bias_only_attention, fullgraph=True)(vb, bb), eager, atol=0, rtol=0)
    sv, sb = vb.clone(), bb.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        bias_only_attention(sv, sb)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = bias_only_attention(sv, sb)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, atol=0, rtol=0)
    # the backward under compile
    got = _run(torch.compile(bias_only_attention, fullgraph=True), v, bias, dy, BF)
    want = _run(bias_only_attention, v, bias, dy, BF)
    for g_, w_ in zip(got, want, strict=True):
        torch.testing.assert_close(g_, w_, atol=0, rtol=0)


def test_a_replay_of_the_backward_is_bit_identical():
    v, bias, dy = _case(1, 4, 256, 48, 5)
    first = _run(bias_only_attention, v, bias, dy, BF)
    second = _run(bias_only_attention, v, bias, dy, BF)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_the_gate_and_the_switch(monkeypatch):
    v, bias, _ = _case(1, 2, 128, 32)
    vb, bb = v.to(BF), bias.to(BF)
    assert sm80.serves(vb, bb)
    assert not sm80.serves(v, bias), "fp32 keeps Triton"
    assert not sm80.serves(vb[:, :, :100, :100], bb[:, :, :100, :100]), "L must be a multiple of 128"
    assert not sm80.serves(torch.randn(1, 2, 128, 128, 40, device="cuda", dtype=BF), bb), "head width 40"
    assert not sm80.serves(torch.randn(1, 2, 128, 64, 32, device="cuda", dtype=BF), bb), "v must be [B, H, L, L, D]"
    assert not sm80.serves(vb.cpu(), bb.cpu()), "CPU tensors"
    assert not sm80.serves(torch.empty(1, 2, 1152, 1152, 32, device="meta", dtype=BF), torch.empty(1, 2, 1152, 1152, device="meta", dtype=BF)), "L above 1024"
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_ATTN_SM80", "0")
    assert not sm80.serves(vb, bb)
    torch.testing.assert_close(bias_only_attention(vb, bb), triton_bias_only_attention(vb, bb), atol=0, rtol=0)      # the switch: the Triton kernels


def test_the_door_keeps_triton_for_a_small_training_call(monkeypatch):
    """At L = 128 the CUDA backward loses to Triton (0.172 against 0.109 ms, 2026-10-04): the door takes CUDA for inference at every L and for training from L = 256; ``=all`` takes every call."""
    calls = []
    orig = sm80.bias_only_attention_sm80
    monkeypatch.setattr(sm80, "bias_only_attention_sm80", lambda *a, **k: calls.append(1) or orig(*a, **k))

    def run(length, grad):
        v, bias, _ = _case(1, 2, length, 32)
        vb, bb = v.to(BF).requires_grad_(grad), bias.to(BF).requires_grad_(grad)
        calls.clear()
        bias_only_attention(vb, bb)
        return len(calls)

    assert run(128, False) == 1, "inference at L = 128 takes the CUDA path"
    assert run(128, True) == 0, "a training call at L = 128 keeps Triton"
    assert run(256, True) == 1, "training from L = 256 takes the CUDA path"
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_ATTN_SM80", "all")
    assert run(128, True) == 1

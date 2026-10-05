"""The A100 whole-op attention core ``ops.augmented_attention_pair_bias`` (kernels/augmented_attention/whole_op.py -> integrations/augattn_sm80.py ``attention_ops``): the attention on
already-projected q / k / v [A, B, H, L, D] (the registry's ``projected_attention`` rows: head dim 24 / 32 / 48, any L, A and B, no mask, a shared key mask or one per sample), its output
and the gradients of q, k, v and the bias no worse against an fp64 PyTorch reference than the engine's Triton path (the switch off), ``torch.compile`` equal to eager, a captured CUDA
graph replaying to eager, and the calls it does not take keeping the Triton path."""

import pytest
import torch

from miniworld_engine import ops, settings
from miniworld_engine.integrations import augattn_sm80
from miniworld_engine.kernels.augmented_attention.reference import (
    augmented_attention_pair_bias_pytorch as reference,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="A100 (sm_80)"),
]
BF = torch.bfloat16


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    old = settings.configure(engine_backend="auto")
    monkeypatch.setenv("MINIWORLD_AUGATTN_SM80_OPS", "all")      # these tests exercise the sm_80 core on every shape; the default dispatch is tested below
    try:
        yield
    finally:
        settings.configure(**vars(old))


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _case(a, b, h, length, d, kind, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(a, b, h, length, d, device="cuda", generator=g) * 0.7 for _ in range(3))
    bias = torch.randn(b, h, length, length, device="cuda", generator=g) * 0.5
    mask = None
    if kind == "shared":
        mask = torch.rand(b, length, device="cuda", generator=g) > 0.2
        mask[:, :8] = True
        mask[:, length - length // 8:] = False
        mask = mask[None].expand(a, b, length)
    elif kind == "sample":
        mask = torch.rand(a, b, length, device="cuda", generator=g) > 0.2
        mask[..., :8] = True
    dy = torch.randn(a, b, h, length, d, device="cuda", generator=g)
    return q, k, v, bias, mask, dy


def _truth(q, k, v, bias, mask, dy):
    leaves = [t.detach().double().requires_grad_() for t in (q, k, v, bias)]
    tq, tk, tv = (t.transpose(2, 3) for t in leaves[:3])
    out = reference(tq, tk, tv, leaves[3].permute(0, 2, 3, 1), mask).transpose(2, 3)
    out.backward(dy.double())
    return [out.detach()] + [t.grad for t in leaves]


def _run(q, k, v, bias, mask, dy):
    leaves = [t.detach().to(BF).requires_grad_() for t in (q, k, v, bias)]
    out = ops.augmented_attention_pair_bias(*leaves, mask)
    out.backward(dy.to(BF))
    return [out.detach()] + [t.grad for t in leaves]


SHAPES = [(5, 1, 16, 384, 48, "none"), (3, 1, 8, 200, 48, "shared"), (2, 2, 16, 256, 24, "sample"), (1, 1, 8, 384, 48, "none"), (4, 1, 4, 128, 32, "shared"), (2, 3, 2, 130, 32, "sample"),
          (5, 1, 16, 640, 24, "none")]


@pytest.mark.parametrize(("a", "b", "h", "length", "d", "kind"), SHAPES)
def test_output_and_gradients_match_the_triton_path(a, b, h, length, d, kind, monkeypatch):
    args = _case(a, b, h, length, d, kind)
    q, k, v, bias, mask, _ = args
    assert augattn_sm80.serves_ops(q.to(BF), k.to(BF), v.to(BF), bias.to(BF), mask)
    truth = _truth(*args)
    calls = []
    orig = augattn_sm80.attention_ops
    monkeypatch.setattr(augattn_sm80, "attention_ops", lambda *a_, **k_: calls.append(1) or orig(*a_, **k_))
    got = _run(*args)
    assert calls, "the call did not take the sm_80 core"
    monkeypatch.setenv("MINIWORLD_AUGATTN_SM80", "0")
    n = len(calls)
    triton = _run(*args)
    assert len(calls) == n, "the switch did not keep the Triton path"
    for name, g_, t_, e_ in zip(("out", "dq", "dk", "dv", "dbias"), got, triton, truth, strict=True):
        assert torch.isfinite(g_).all(), name
        assert g_.dtype == t_.dtype, name
        assert g_.shape == t_.shape, name
        assert _rel(g_, e_) <= 1.5 * _rel(t_, e_) + 2e-3, f"{name}: sm_80 {_rel(g_, e_):.2e} vs Triton {_rel(t_, e_):.2e}"


def test_compile_and_graph_replay_match_eager():
    q, k, v, bias, mask, _ = _case(4, 1, 16, 256, 48, "sample")
    qb, kb, vb, bb = (t.to(BF) for t in (q, k, v, bias))

    def f(q_, k_, v_, b_):
        return ops.augmented_attention_pair_bias(q_, k_, v_, b_, mask)

    eager = f(qb, kb, vb, bb)
    compiled = torch.compile(f, fullgraph=True)(qb, kb, vb, bb)
    torch.testing.assert_close(compiled, eager, atol=0, rtol=0)
    sq, sk, sv, sb = (t.clone() for t in (qb, kb, vb, bb))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        f(sq, sk, sv, sb)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = f(sq, sk, sv, sb)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, atol=0, rtol=0)


def test_training_under_compile_matches_eager():
    q, k, v, bias, mask, dy = _case(3, 1, 16, 256, 48, "shared")

    def run(fn):
        leaves = [t.detach().to(BF).requires_grad_() for t in (q, k, v, bias)]
        out = fn(*leaves)
        out.backward(dy.to(BF))
        return [out.detach()] + [t.grad for t in leaves]

    eager = run(lambda *x: ops.augmented_attention_pair_bias(*x, mask))
    compiled = run(torch.compile(lambda *x: ops.augmented_attention_pair_bias(*x, mask), fullgraph=True))
    for e, c in zip(eager, compiled, strict=True):
        torch.testing.assert_close(c, e, atol=0, rtol=0)


def test_the_gate_serves_only_what_the_core_was_built_for(monkeypatch):
    q, k, v, bias, mask, _ = _case(2, 1, 16, 256, 48, "sample")
    qb, kb, vb, bb = (t.to(BF) for t in (q, k, v, bias))
    assert augattn_sm80.serves_ops(qb, kb, vb, bb, mask)
    assert augattn_sm80.serves_ops(qb, kb, vb, bb, None)
    assert not augattn_sm80.serves_ops(q, k, v, bias, mask), "fp32 inputs keep the Triton path"
    assert not augattn_sm80.serves_ops(qb, kb, vb, bb, mask.to(torch.uint8)), "a non-bool mask"
    assert not augattn_sm80.serves_ops(qb, kb, vb, bb, mask, "memory_efficient")
    assert not augattn_sm80.serves_ops(qb[..., :40], kb[..., :40], vb[..., :40], bb, mask), "head dim 40"
    assert not augattn_sm80.serves_ops(qb.cpu(), kb.cpu(), vb.cpu(), bb.cpu(), None), "CPU tensors"
    monkeypatch.setenv("MINIWORLD_AUGATTN_SM80", "0")
    assert not augattn_sm80.serves_ops(qb, kb, vb, bb, mask)


def test_the_default_dispatch_serves_the_calls_the_core_wins(monkeypatch):
    """The whole-op path stages q / k / v / bias / the output around the core: at the registry's sizes that loses to the Triton kernel in inference (13-71 % slower, measured 2026-10-04) and in small
    training calls, and wins in training from A B H L^2 >= 2e6 (16 x 768, A = 48: 1.3-1.95x) and at head dim 24. MINIWORLD_AUGATTN_SM80_OPS=all serves every call, 0 none."""
    monkeypatch.delenv("MINIWORLD_AUGATTN_SM80_OPS")

    def call(a, h, length, d, grad):
        q, k, v, bias, _, _ = _case(a, 1, h, length, d, "none")
        return [t.to(BF).requires_grad_(grad) for t in (q, k, v, bias)]

    assert not augattn_sm80.serves_ops(*call(5, 16, 384, 48, False), None), "inference keeps the Triton kernel"
    with torch.no_grad():
        assert not augattn_sm80.serves_ops(*call(48, 16, 384, 48, True), None), "no autograd, no core"
    assert augattn_sm80.serves_ops(*call(48, 16, 128, 48, True), None), "training above the size"
    assert not augattn_sm80.serves_ops(*call(1, 8, 256, 48, True), None), "a small training call keeps Triton"
    assert augattn_sm80.serves_ops(*call(1, 8, 768, 48, True), None)
    assert augattn_sm80.serves_ops(*call(1, 16, 256, 24, True), None), "head dim 24"
    monkeypatch.setenv("MINIWORLD_AUGATTN_SM80_OPS", "all")
    assert augattn_sm80.serves_ops(*call(5, 16, 384, 48, False), None)
    monkeypatch.setenv("MINIWORLD_AUGATTN_SM80_OPS", "0")
    assert not augattn_sm80.serves_ops(*call(48, 16, 384, 48, True), None)

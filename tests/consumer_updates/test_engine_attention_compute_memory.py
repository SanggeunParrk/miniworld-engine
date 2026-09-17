"""Compute-efficient attention uses bounded non-atomic scratch and preserves gradients."""

import pytest
import torch
import triton

from miniworld_engine.kernels.augmented_attention import interface
from miniworld_engine.kernels.augmented_attention.reference import (
    augmented_attention_pair_bias_pytorch,
)
from miniworld_engine.kernels.augmented_attention.triton import main


@pytest.mark.parametrize("training", [True, False])
def test_large_shape_keeps_compute_efficient(monkeypatch, training):
    monkeypatch.setattr(
        interface, "_pair_bias_compute_efficient", lambda *args: "compute"
    )
    monkeypatch.setattr(
        interface, "_pair_bias_memory_efficient", lambda *args: "atomic"
    )
    q = torch.empty(48, 1, 8192, 4, 32, device="meta", requires_grad=training)
    b = torch.empty(1, 8192, 8192, 4, device="meta")
    with torch.set_grad_enabled(training):
        assert interface.triton_augmented_attention_pair_bias(q, q, q, b) == "compute"
        assert (
            interface.triton_augmented_attention_pair_bias(
                q, q, q, b, compute_efficient=False
            )
            == "atomic"
        )


def pin(monkeypatch, bm=32, bn=64, stages=2):
    # Correctness tests cover known legal tilings; full production grids are
    # exercised by the independent cache builder, not re-tuned in every test.
    for fn, kwargs in [
        (main._attn_fwd, {"BLOCK_M1": 32, "BLOCK_M2": 64}),
        (main._attn_bwd_preprocess, {"BLOCK_M1": 32}),
        (main._attn_bwd, {"BLOCK_M1": bm, "BLOCK_M2": bn}),
        (main._dq_reduce, {"BLOCK_E": 256}),
    ]:
        monkeypatch.setattr(
            fn, "configs", [triton.Config(kwargs, num_warps=4, num_stages=stages)]
        )
        fn.cache.clear()


def inputs(dtype, d=32, length=259):
    torch.manual_seed(7)
    args = [
        torch.randn(3, 2, length, 4, d, device="cuda", dtype=dtype).requires_grad_()
        for _ in range(3)
    ]
    args.append(
        torch.randn(2, length, length, 4, device="cuda", dtype=dtype).requires_grad_()
    )
    mask = torch.rand(3, 2, length, device="cuda") > 0.25
    mask[0, 0] = False
    return args, mask


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dim", [16, 32, 48])
@pytest.mark.parametrize("tiles", [(32, 64), (64, 128), (128, 32)])
def test_chunked_matches_legacy_and_reference(monkeypatch, dtype, head_dim, tiles):
    pin(monkeypatch, *tiles)
    args, mask = inputs(dtype, head_dim)
    grad = torch.randn_like(args[0])
    monkeypatch.setattr(main, "_SPLIT_BIAS_BUDGET_BYTES", 1 << 60)
    legacy = interface.triton_augmented_attention_pair_bias(*args, mask)
    legacy_grads = torch.autograd.grad(legacy, args, grad)
    monkeypatch.setattr(main, "_SPLIT_BIAS_BUDGET_BYTES", 0)
    monkeypatch.setattr(main, "_CHUNK_WORKSPACE_BYTES", 1)
    monkeypatch.setattr(
        interface,
        "_pair_bias_memory_efficient",
        lambda *a: pytest.fail("atomic fallback"),
    )
    actual = interface.triton_augmented_attention_pair_bias(*args, mask)
    actual_grads = torch.autograd.grad(actual, args, grad)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref = augmented_attention_pair_bias_pytorch(*args, mask)
        ref_grads = torch.autograd.grad(ref, args, grad)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    for result, want in [(actual, ref), *zip(actual_grads, ref_grads)]:
        assert torch.isfinite(result).all()
        assert (result.float() - want.float()).norm() / want.float().norm() < 0.02
    for result, want in zip(actual_grads, legacy_grads):
        rel = (result.float() - want.float()).norm() / want.float().norm()
        assert rel < 2e-6, float(rel)
    assert torch.count_nonzero(actual[0, 0]) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", [128, 256, 384, 512, 640, 768])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("chunked", [False, True])
def test_head48_stage1_gradient(monkeypatch, length, dtype, chunked):
    # Production's 768 channels / 16 heads, including the tile that faulted
    # during full-grid tuning. Nonzero dQ catches silent descriptor corruption
    # even when an isolated launch happens not to trigger illegal-address errors.
    pin(monkeypatch, bm=32, bn=64, stages=1)
    monkeypatch.setattr(main, "_SPLIT_BIAS_BUDGET_BYTES", 0 if chunked else 1 << 60)
    monkeypatch.setattr(main, "_CHUNK_WORKSPACE_BYTES", 1)
    monkeypatch.setattr(
        interface,
        "_pair_bias_memory_efficient",
        lambda *a: pytest.fail("atomic fallback"),
    )
    torch.manual_seed(37)
    args = [
        torch.randn(2, 2, length, 16, 48, device="cuda", dtype=dtype).requires_grad_()
        for _ in range(3)
    ]
    args.append(
        torch.randn(2, length, length, 16, device="cuda", dtype=dtype).requires_grad_()
    )
    mask = torch.rand(2, 2, length, device="cuda") > 0.25
    mask[0, 0] = False
    grad = torch.randn_like(args[0])
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        actual = interface.triton_augmented_attention_pair_bias(*args, mask)
        actual_grads = torch.autograd.grad(actual, args, grad)
        ref = augmented_attention_pair_bias_pytorch(*args, mask)
        ref_grads = torch.autograd.grad(ref, args, grad)
        for result, want in [(actual, ref), *zip(actual_grads, ref_grads)]:
            assert torch.isfinite(result).all()
            rel = (result.float() - want.float()).norm() / want.float().norm()
            assert rel < 0.02, float(rel)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("length", [128, 131])
def test_compile_cuda_graph_and_repeatability(monkeypatch, dtype, length):
    pin(monkeypatch)
    monkeypatch.setattr(main, "_SPLIT_BIAS_BUDGET_BYTES", 0)
    monkeypatch.setattr(main, "_CHUNK_WORKSPACE_BYTES", 1)
    monkeypatch.setattr(
        interface,
        "_pair_bias_memory_efficient",
        lambda *a: pytest.fail("atomic fallback"),
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        args, mask = inputs(dtype, length=length)
        fn = torch.compile(
            interface.triton_augmented_attention_pair_bias,
            fullgraph=True,
            dynamic=False,
        )

        def step():
            for t in args:
                t.grad = None
            out = fn(*args, mask)
            out.sum().backward()
            return out, tuple(t.grad for t in args)

        out, grads = step()
        expected = (out.detach().clone(), tuple(g.detach().clone() for g in grads))
        del out, grads
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = step()
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for result, want in [
            (captured[0], expected[0]),
            *zip(captured[1], expected[1]),
        ]:
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result, want, atol=0, rtol=0)

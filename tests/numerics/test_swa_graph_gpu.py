"""FA2 graph capture and fullgraph training against independent legacy packing."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch


def legacy_window(q, k, v, cu_seqlens, seqused, max_seqlen, valid, n, s, scale, half_window):
    """Original exact-length FA2 packing; deliberately independent of the engine path."""
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    if not bool(valid.any()):
        return (q + k + v) * 0
    h, d = q.shape[2:]
    dtype = q.dtype
    clean = [torch.where(valid[..., None, None], t.to(torch.bfloat16), 0) for t in (q, k, v)]
    packed_q, indices, cu, maximum = unpad_input(clean[0], valid)[:4]
    packed_k, packed_v = [index_first_axis(t.reshape(n * s, h, d), indices) for t in clean[1:]]
    window = (-1, -1) if half_window < 0 else (half_window, half_window)
    out = flash_attn_varlen_func(packed_q, packed_k, packed_v, cu, cu, maximum, maximum,
                                softmax_scale=scale, window_size=window)
    return pad_input(out, indices, n, s).to(dtype)


@pytest.fixture
def swa():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("FA2 CUDA graph regression needs an sm80-family GPU")
    from miniworld_engine import settings
    from miniworld_engine.modules.swa_atom_attention import module

    if module._flash_backend() != "fa2":
        pytest.skip("FA2 unavailable")
    torch.manual_seed(0)
    previous = settings.configure(run_autotune=False, capture=False, autotune_on_miss_shards="",
                                  autotune_miss_cap=1)
    try:
        yield module
    finally:
        settings.configure(**asdict(previous))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mask_kind", ["front", "holes_empty", "all_empty"])
def test_fa2_no_grad_graph_replays_changed_masks(swa, dtype, mask_kind):
    n, s, heads, width = 3, 128, 4, 32
    q, k, v = [torch.randn(n, s, heads, width, device="cuda", dtype=dtype) for _ in range(3)]
    valid = torch.arange(s, device="cuda")[None, :] < torch.tensor([128, 93, 17], device="cuda")[:, None]
    if mask_kind == "holes_empty":
        valid[:, 1::3] = False
        valid[1] = False
    elif mask_kind == "all_empty":
        valid.zero_()
    counts = valid.sum(-1, dtype=torch.int32)
    cu = torch.arange(0, (n + 1) * s, s, device="cuda", dtype=torch.int32)

    def call():
        return swa._flash_window_core(q, k, v, cu, counts, s, valid, n, s, width**-.5, 16)

    def reference():
        if not bool(valid.any()):
            return torch.zeros_like(q)
        with torch.enable_grad():
            return legacy_window(q, k, v, cu, counts, s, valid, n, s, width**-.5, 16)

    with torch.no_grad():
        expected = reference()
        torch.testing.assert_close(call(), expected, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = call()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        q.copy_(torch.randn_like(q))
        valid.copy_(torch.rand_like(valid, dtype=torch.float32) > .35)
        valid[0] = False
        counts.copy_(valid.sum(-1, dtype=torch.int32))
        expected = reference()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert bool(torch.isfinite(result).all())


@pytest.mark.parametrize("target", ["attention", "dit"])
def test_swa_downstream_training_matches_legacy_packing(swa, target):
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.swa_dit import SWADiTBlock

    n, s, width = 2, 128, 128
    if target == "attention":
        model = swa.SWA3DRoPEAttention(width, 4, half_window=16)
    else:
        model = SWADiTBlock(width, width, 4, half_window=16,
                           implementation=ImplementationType.MINIWORLD)
    model = model.cuda().to(torch.bfloat16).train()
    if target == "dit":
        # Exercise both branches: identity initialization hides attention differences.
        modulation = model.adaln_modulation[1]
        assert isinstance(modulation, torch.nn.Linear)
        with torch.no_grad():
            modulation.weight.normal_(std=0.1)
    baseline = copy.deepcopy(model)
    valid = torch.rand(n, s, device="cuda") > .25
    valid[1] = False
    cos = torch.randn(1, s, 16, device="cuda")
    sin = torch.randn_like(cos)
    ap = swa.build_attention_params(cos, sin, valid, num_aug=n)
    x = torch.randn(n, s, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cond = torch.randn_like(x, requires_grad=True)
    bx, bc = x.detach().clone().requires_grad_(), cond.detach().clone().requires_grad_()
    grad = torch.randn_like(x)
    result = model(x, ap) if target == "attention" else model(x, cond, ap)
    result.backward(grad)
    with patch.object(swa, "flash_window_seqused", legacy_window):
        expected = baseline(bx, ap) if target == "attention" else baseline(bx, bc, ap)
        expected.backward(grad)
    pairs = {"output": (result, expected), "dx": (x.grad, bx.grad)}
    if target == "dit":
        pairs["dcond"] = cond.grad, bc.grad
    pairs.update({name: (value.grad, dict(baseline.named_parameters())[name].grad)
                  for name, value in model.named_parameters()})
    metrics = {}
    for name, (actual, ref) in pairs.items():
        assert actual is not None, name
        assert ref is not None, name
        assert bool(torch.isfinite(actual).all()), name
        assert bool(torch.isfinite(ref).all()), name
        delta = actual.float() - ref.float()
        metrics[name] = {"max_abs": delta.abs().max().item(),
                         "relative_frobenius": (delta.norm() / ref.float().norm().clamp_min(1e-12)).item()}
        # BF16 gradients may differ by rounding from FA2's nondeterministic backward reductions.
        torch.testing.assert_close(actual, ref, rtol=.02, atol=.02)
    print("SWA_TRAIN_LEGACY_COMPARISON " + json.dumps({"target": target, "metrics": metrics}), flush=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("only_q", [False, True])
def test_fa2_fullgraph_backward_and_changed_masks(swa, dtype, only_q):
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    graphs = {}

    def compiler(kind):
        def capture(gm, inputs):
            graphs.setdefault(kind, []).append(gm)
            return make_boxed_func(gm.forward)
        return capture

    run = torch.compile(swa.flash_window_seqused, fullgraph=True,
                        backend=aot_autograd(fw_compiler=compiler("forward"),
                                             bw_compiler=compiler("backward")))
    n, s, h, d = 3, 37, 4, 32
    tensors = [torch.randn(n, s, h, d, device="cuda", dtype=dtype).requires_grad_(i == 0 or not only_q)
               for i in range(3)]
    cu = torch.arange(0, (n + 1) * s, s, device="cuda", dtype=torch.int32)
    for mask_kind in ("holes", "front", "empty"):
        valid = torch.rand(n, s, device="cuda") > .3
        if mask_kind == "holes":
            valid[1] = False
        elif mask_kind == "front":
            valid = torch.arange(s, device="cuda")[None] < valid.sum(-1)[:, None]
        else:
            valid.zero_()
        counts = valid.sum(-1, dtype=torch.int32)
        refs = [t.detach().clone().requires_grad_(t.requires_grad) for t in tensors]
        args = (cu, counts, s, valid, n, s, d**-.5, 4)
        out = run(*tensors, *args)
        expected = legacy_window(refs[0], refs[1], refs[2], *args)
        grad = torch.randn_like(out)
        out.backward(grad)
        expected.backward(grad)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        for t, ref in zip(tensors, refs, strict=True):
            if t.requires_grad:
                assert t.grad is not None
                assert ref.grad is not None
                torch.testing.assert_close(t.grad, ref.grad, atol=.02, rtol=.02)
                assert torch.isfinite(t.grad).all()
                assert torch.count_nonzero(t.grad[~valid]) == 0
            else:
                assert t.grad is None
            t.grad = None
    assert len(graphs["forward"]) == len(graphs["backward"]) == 1
    # The mutating FA2 op may be nested inside auto_functionalized, so inspect arguments too.
    backward_graph = str(graphs["backward"][0].graph)
    assert "flash_attn._flash_attn_varlen_backward" in backward_graph, backward_graph
    assert "nonzero" not in backward_graph, backward_graph
    assert "_local_scalar_dense" not in backward_graph, backward_graph

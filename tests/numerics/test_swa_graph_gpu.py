"""FA2 no-grad graph capture and existing downstream training semantics on sm80+."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch


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
            return call()  # unchanged differentiable unpad path

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
    with patch.object(swa, "flash_window_seqused", swa._flash_window_core):
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

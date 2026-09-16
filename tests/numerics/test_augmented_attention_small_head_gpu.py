"""Small attention heads must pad dot lanes without changing outputs or gradients."""
import json

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("compute_efficient", [True, False])
@pytest.mark.parametrize("head_dim", [8, 24, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_padded_attention_matches_float_reference(head_dim, dtype, compute_efficient, monkeypatch):
    from miniworld_engine.kernels.augmented_attention import interface
    from miniworld_engine.kernels.augmented_attention.triton import (
        main,
        memory_efficient,
    )

    backend = main if compute_efficient else memory_efficient
    names = ["_attn_fwd", "_attn_bwd", "_attn_bwd_preprocess"]
    if compute_efficient:
        names.append("_dq_reduce")

    # This tests masked padding arithmetic, not the full autotune search. Keep
    # the same legal tile for both dimensions and precisions without shared writes.
    for name in names:
        tuner = getattr(backend, name)
        candidates = [config for config in tuner.configs
                      if config.num_warps == 4 and config.num_stages == 2
                      and all(config.kwargs.get(axis, 32) == 32 for axis in ("BLOCK_M1", "BLOCK_M2"))]
        assert candidates, name
        monkeypatch.setattr(tuner, "configs", candidates[:1])
        monkeypatch.setattr(tuner, "cache", {})

    torch.manual_seed(17)
    shape = (2, 1, 31, 2, head_dim)
    q, k, v = [torch.randn(shape, device="cuda", dtype=dtype).requires_grad_()
               for _ in range(3)]
    bias = torch.randn((1, 31, 31, 2), device="cuda", dtype=torch.float32).requires_grad_()
    mask = torch.rand((2, 1, 31), device="cuda") > 0.125
    mask[..., 0] = True
    reference_inputs = [x.detach().float().requires_grad_() for x in (q, k, v, bias)]
    rq, rk, rv, rb = reference_inputs
    logits = torch.einsum("ablhd,abmhd->abhlm", rq, rk) * head_dim**-0.5
    logits = logits + rb.permute(0, 3, 1, 2).unsqueeze(0)
    logits = logits.masked_fill(~mask[:, :, None, None, :], float("-inf"))
    reference = torch.einsum("abhlm,abmhd->ablhd", logits.softmax(-1), rv)
    actual = interface.triton_augmented_attention_pair_bias(
        q, k, v, bias, mask, compute_efficient=compute_efficient,
    )
    grad = torch.randn_like(actual)
    actual.backward(grad)
    reference.backward(grad.float())
    tolerance = {"atol": 0.04, "rtol": 0.04} if dtype == torch.bfloat16 else {"atol": 0.005, "rtol": 0.01}
    torch.testing.assert_close(actual.float(), reference, atol=tolerance["atol"], rtol=tolerance["rtol"])
    metrics = {"dtype": str(dtype), "head_dim": head_dim, "tolerance": tolerance}

    def errors(a, b):
        delta = a.float() - b.float()
        return {"max_abs": delta.abs().max().item(),
                "relative_frobenius": (delta.norm() / b.float().norm().clamp_min(1e-12)).item()}

    metrics["output"] = errors(actual, reference)
    for label, actual_input, reference_input in zip(
            ("dq", "dk", "dv", "dbias"), (q, k, v, bias), reference_inputs, strict=True):
        assert actual_input.grad is not None
        assert reference_input.grad is not None
        assert torch.isfinite(actual_input.grad).all()
        torch.testing.assert_close(actual_input.grad.float(), reference_input.grad,
                                   atol=tolerance["atol"], rtol=tolerance["rtol"])
        metrics[label] = errors(actual_input.grad, reference_input.grad)
    print("SMALL_HEAD_METRICS " + json.dumps(metrics, sort_keys=True))

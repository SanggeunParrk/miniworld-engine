"""Frozen-weight design gradients from the original ESM-family transition rows.

Requires the isolated ESMFold2 Transformers fork, pinned by upstream.
"""
import os

import pytest
import torch

from miniworld_engine.integrations import anthropic as A

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not os.environ.get("MINIWORLD_ANTHROPIC_ROOT"), reason="needs pinned upstream")]
pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2_common")


@pytest.mark.parametrize("row", ["esm_kd3", "esm_kd3:lean", "esm_t15_kd3"])
def test_frozen_transition_input_gradient(row):
    torch.manual_seed(513)
    provider = A.provider("transition")
    carried = provider.carried_module("esm_kd3")
    eps = float(carried.C._EPS)
    x = torch.randn(128 * 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = provider.pack(
        w_a=torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16) / 16,
        w_b=torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16) / 16,
        w_o=torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16) / 32,
        ln_w=torch.ones(256, device="cuda"), ln_b=torch.zeros(256, device="cuda"), eps=eps)
    got, selection = A.transition_autograd(x, weights, row=row, n_tokens=128)
    ref_x = x.detach().float().requires_grad_()
    norm = torch.nn.functional.layer_norm(ref_x, (256,), weights.ln_w, weights.ln_b, eps)
    a = torch.nn.functional.linear(norm, weights.w_a.float())
    b = torch.nn.functional.linear(norm, weights.w_b.float())
    want = ref_x + torch.nn.functional.linear(torch.nn.functional.silu(a) * b, weights.w_o.float())
    dy = torch.randn_like(x)
    actual_dx, = torch.autograd.grad(got, x, dy)
    expected_dx, = torch.autograd.grad(want, ref_x, dy.float())
    assert selection.row == row.split(":")[0]
    assert (got.float() - want).norm() / want.norm() < .02
    assert (actual_dx.float() - expected_dx).norm() / expected_dx.norm() < .03

"""The upstream input-gradient entry point, without a recomputation substitute."""
import os

import pytest
import torch

from miniworld_engine.integrations import anthropic as A

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not os.environ.get("MINIWORLD_ANTHROPIC_ROOT"), reason="needs upstream payload")]


@pytest.mark.parametrize("residual", [False, True])
def test_layer_norm_native_input_gradient(residual):
    torch.manual_seed(722)
    x = torch.randn(128 * 128, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, device="cuda", dtype=torch.float32) * .05 + 1
    dy = torch.randn_like(x)
    extra = torch.randn_like(x) if residual else None
    mean = x.float().mean(-1)
    rstd = (x.float().var(-1, unbiased=False) + 1e-5).rsqrt()
    dx, selected = A.layer_norm_backward_dx(dy, x, w, mean, rstd, residual_grad=extra)
    xx = x.float().requires_grad_()
    ref = torch.nn.functional.layer_norm(xx, (256,), w, eps=1e-5)
    want, = torch.autograd.grad(ref, xx, dy.float())
    if extra is not None:
        want = want + extra.float()
    assert selected.row == "ef2_ln_bwd_dx"
    assert (dx.float() - want).norm() / want.norm() < .01

"""Single-direction fused training preserves mask, dropout and all gradients."""
import pytest
import torch

from miniworld_engine.modules import TriangleMultiplication
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_unidirectional_training(outgoing, dtype, all_invalid, monkeypatch):
    torch.manual_seed(941)
    torch.backends.cuda.matmul.allow_tf32 = False
    actual = TriangleMultiplication(128, outgoing=outgoing, implementation=ImplementationType.TRITON,
                                    p_drop=.25).cuda().to(dtype).train()
    reference = TriangleMultiplication(128, outgoing=outgoing, implementation=ImplementationType.PYTORCH,
                                       p_drop=.25).cuda().float().train()
    with torch.no_grad():
        for name, param in actual.named_parameters():
            if "ln_" not in name:
                param.normal_(std=128**-.5)
        actual.ln_pair.bias.fill_(.2)
        actual.ln_out.bias.fill_(.3)
    reference.load_state_dict(actual.state_dict())
    length = 33  # Exercise row tails and both contraction orientations.
    scale = (torch.rand(1, 1, length, 128, device="cuda") > .25).to(dtype) / .75
    for model in (actual, reference):
        model.ln_pair.eps, model.ln_out.eps = .03, .07
        monkeypatch.setattr(model, "_make_drop_row_scale", lambda pair, p: scale.to(pair.dtype))
    x = torch.randn(1, length, length, 128, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    mask = torch.ones(1, length, device="cuda", dtype=torch.bool)
    mask[:, ::3] = False
    if all_invalid:
        mask.zero_()
    y, yr = actual(x, mask), reference(xr, mask)
    dy = torch.randn_like(y)
    y.backward(dy)
    yr.backward(dy.float())
    values = [("output", y, yr), ("input", x.grad, xr.grad)]
    refs = dict(reference.named_parameters())
    values += [(name, p.grad, refs[name].grad) for name, p in actual.named_parameters()]
    for name, a, b in values:
        assert a is not None, name
        assert b is not None, name
        assert torch.isfinite(a).all(), name
        assert torch.isfinite(b).all(), name
        error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)
        assert error < (.025 if dtype == torch.bfloat16 else .001), (name, error.item())

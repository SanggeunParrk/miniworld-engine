"""A non-bf16 input must run through Transition's PyTorch reference, whatever the backend.

`guard_dtype` sends every non-bf16 input to `_torch_forward`, while the projections are
bf16-pinned; the reference has to cast them to the activation dtype instead of failing.
"""
import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.modules import ImplementationType, Transition


@pytest.mark.parametrize("implementation", [ImplementationType.PYTORCH, ImplementationType.TRITON])
def test_fp32_input_runs_the_reference(implementation):
    torch.manual_seed(931)
    m = Transition(32, n=2, implementation=implementation)
    with torch.no_grad():
        m.squeeze.weight.normal_(std=0.1)
    x = torch.randn(3, 5, 32)

    y = m(x)

    h = F.layer_norm(x, (32,), m.ln_in.weight, m.ln_in.bias, m.ln_in.eps)
    a = h @ m.expand_a.weight.float().t()
    b = h @ m.expand_b.weight.float().t()
    expected = x + (F.silu(a) * b) @ m.squeeze.weight.float().t()
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, expected)

"""FP32 norm weights with a low-precision input also work without norm bias."""

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.primitives import LayerNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_mixed_norm_affine_fallback(dtype, bias):
    torch.manual_seed(640)
    norm = LayerNorm(16, bias=bias, implementation=ImplementationType.MINIWORLD).cuda().to(dtype)
    reference = torch.nn.LayerNorm(16, bias=bias).cuda().float()
    reference.load_state_dict(norm.state_dict())
    assert norm.weight.dtype == torch.float32
    x = torch.randn(2, 32, 16, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    dy = torch.randn_like(x)
    y, yr = norm(x), reference(xr)
    assert y.dtype == dtype
    torch.testing.assert_close(y.float(), yr, rtol=0.01, atol=0.02)
    y.backward(dy)
    yr.backward(dy.float())
    for actual, expected in [
        (x.grad, xr.grad),
        (norm.weight.grad, reference.weight.grad),
    ]:
        assert actual is not None
        assert expected is not None
        error = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-8)
        assert error < 0.01
    if bias:
        torch.testing.assert_close(
            norm.bias.grad, reference.bias.grad, rtol=0.01, atol=0.02
        )


# Engine CI selects GPU checks explicitly.
pytestmark = [pytest.mark.gpu, *([pytestmark] if "pytestmark" in globals() else [])]

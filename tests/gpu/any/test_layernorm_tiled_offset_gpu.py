"""Every feature tile must preserve variance when the row has a large offset."""
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [128, 137, 384])
@pytest.mark.parametrize("block_k", [32, 64, 512])
def test_tiled_layernorm_centered_variance(dtype, width, block_k):
    from miniworld_engine.kernels.layernorm.triton.main import layer_norm_fwd_fused
    torch.manual_seed(23)
    x = (torch.randn(37, width, device="cuda") + 1000).to(dtype)
    weight = torch.randn(width, device="cuda")
    bias = torch.randn_like(weight)
    y = torch.empty_like(x)
    mean = torch.empty(37, device="cuda")
    rstd = torch.empty_like(mean)
    layer_norm_fwd_fused.fn[(3,)](
        x, y, weight, bias, mean, rstd, rstd, width, 1, 37, width, 1e-5,
        BLOCK_M1=16, BLOCK_K=block_k, shape_key=37, HAS_ROWSCALE=False, num_warps=4)
    xd = x.double()
    expected_mean = xd.mean(-1)
    expected_rstd = torch.rsqrt(xd.var(-1, unbiased=False) + 1e-5)
    expected = ((xd - expected_mean[:, None]) * expected_rstd[:, None] * weight + bias).to(dtype)
    torch.testing.assert_close(mean.double(), expected_mean, atol=1.5e-4, rtol=0)
    torch.testing.assert_close(rstd.double(), expected_rstd, atol=1e-4, rtol=1e-4)
    # BF16 output rounding; FP32 tile means have at most a few offset-sized ULPs.
    torch.testing.assert_close(y, expected, atol=.016 if dtype == torch.bfloat16 else .001,
                               rtol=.008 if dtype == torch.bfloat16 else .001)

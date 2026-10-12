"""ESMFold2's projection bias must be normalized along with the outer product."""
import pytest
import torch

from miniworld_engine.modules import ImplementationType, OuterProductMean


@pytest.mark.parametrize("before", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_projection_order_with_nonzero_bias_and_mask(before, dtype):
    torch.manual_seed(5)
    module = OuterProductMean(4, 3, 2, normalize_before_proj=before,
                              implementation=ImplementationType.PYTORCH).to(dtype)
    with torch.no_grad():
        module.to_out.weight.fill_(0.25)
        module.to_out.bias.copy_(torch.tensor([1.0, -2.0, 3.0]))
    msa = torch.randn(1, 3, 2, 4, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[[True, True], [True, False], [False, False]]])
    normalized = module.ln_msa(msa)
    left = module.to_left(normalized) * mask[..., None]
    right = module.to_right(normalized) * mask[..., None]
    outer = torch.einsum("bmid,bmje->bijde", left, right).flatten(-2)
    count = torch.einsum("bmi,bmj->bij", mask.float(), mask.float()).clamp(min=1)[..., None]
    expected = (module.to_out((outer / count).to(dtype)) if before
                else (module.to_out(outer) / count).to(dtype))
    residual = torch.randn(1, 2, 2, 3, dtype=dtype)
    actual = module(msa, mask, residual=residual)
    torch.testing.assert_close(actual, residual + expected, atol=0, rtol=0)
    actual.float().sum().backward()
    assert msa.grad is not None
    assert torch.isfinite(msa.grad).all()


def test_swa_helpers_remain_importable_at_the_module_boundary():
    from miniworld_engine.modules.swa_atom_attention import (
        apply_rotary_emb_3d,
        build_3d_rope,
        build_attention_params,
        flash_window_seqused,
        sparse_neighbor_attention,
    )

    assert all(callable(fn) for fn in (
        apply_rotary_emb_3d, build_3d_rope, build_attention_params,
        flash_window_seqused, sparse_neighbor_attention,
    ))

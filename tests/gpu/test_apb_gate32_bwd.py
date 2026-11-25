"""``gate32_bwd`` (kernels/augmented_attention/cuda/apb/apb_rows.cu): the gate / dO / dd glue of AttentionPairBias's fp32 (TF32) training backward in
one kernel. Its dO and gate gradient are the statements' own fp32 operations (bitwise); dd sums each head's dhp products in order, where the
statements' ``sum(-1)`` is free to order them: equal to a few fp32 ulp."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="an sm_90 device"),
]


@pytest.mark.parametrize("heads,dhp", [(24, 32), (12, 32), (16, 32), (8, 48), (16, 48)])
@pytest.mark.parametrize("L", [128, 384])
def test_gate32_bwd_matches_the_statements(heads, dhp, L):
    from foldjet_engine.kernels.augmented_attention.cuda import apb

    W = heads * dhp
    g = torch.Generator(device="cuda").manual_seed(7)
    dog = torch.randn(L, W, device="cuda", generator=g)
    O = torch.randn(L, W, device="cuda", generator=g)
    qkvg = torch.randn(L, 4 * W, device="cuda", generator=g)
    qkvg[:, 3 * W :] = torch.sigmoid(qkvg[:, 3 * W :])
    s = qkvg[:, 3 * W :]

    # the statements it replaces
    dog_ref = dog.clone()
    dO_ref = dog_ref * s
    dog_ref.mul_(O)
    dd_ref = (dog_ref * s).view(L, heads, dhp).sum(-1).t().contiguous().view(1, heads, L)
    dcat_ref = torch.zeros(L, 4 * W, device="cuda")
    torch.mul(dog_ref, s * (1.0 - s), out=dcat_ref[:, 3 * W :])

    dO = torch.empty_like(dog)
    dd = dog.new_empty((1, heads, L))
    dcat = torch.zeros(L, 4 * W, device="cuda")
    apb.ext().gate32_bwd(dog, O, qkvg, dO, dd, dcat, dhp)
    assert torch.equal(dO, dO_ref)
    assert torch.equal(dcat[:, 3 * W :], dcat_ref[:, 3 * W :])
    assert not dcat[:, : 3 * W].any()  # the core's columns are untouched
    torch.testing.assert_close(dd, dd_ref, rtol=2e-6, atol=1e-6 * dd_ref.abs().max().item())

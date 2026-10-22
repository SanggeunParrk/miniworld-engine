"""RoseTTAFold3's OuterProductMean (biased left / right projections) and Chai-1's (8 groups, product LayerNorm) on the H100 composed path,
inference, against the fp64 torch statements: the engine's bf16 error held to the bf16 statements' own."""

import copy

import pytest
import torch

from miniworld_engine.kernels.outer_product_mean import composed
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.outer_product import OuterProductMean

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

VARIANTS = {
    "rf3_after": {"d_pair": 128, "d_hidden": 32, "projection_bias": True, "normalize_before_proj": False},
    "rf3_before": {"d_pair": 128, "d_hidden": 32, "projection_bias": True, "normalize_before_proj": True},
    "chai1": {"d_pair": 256, "d_hidden": 64, "groups": 8},
}


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize(("b", "s", "n"), [(1, 512, 128), (1, 1024, 256), (2, 256, 192)])
def test_opm_variant(name, b, s, n):
    options = dict(VARIANTS[name])
    torch.manual_seed(0)
    m = OuterProductMean(64, options.pop("d_pair"), options.pop("d_hidden"), implementation=ImplementationType.MINIWORLD, **options)
    m = m.cuda().eval()
    with torch.no_grad():
        for pname, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in pname and pname.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    g = torch.Generator(device="cuda").manual_seed(1)
    msa = torch.randn(b, s, n, 64, device="cuda", generator=g)
    mask = torch.rand(b, s, n, device="cuda", generator=g) > 0.2
    gold = copy.deepcopy(m).double()
    ref16 = copy.deepcopy(m).to(torch.bfloat16)
    eng = m.to(torch.bfloat16)
    with torch.no_grad():
        assert composed.serves(eng, msa.bfloat16(), mask)
        want = gold._variant_reference(msa.double(), mask, None, None)
        base = ref16._variant_reference(msa.bfloat16(), mask, None, None)
        got = eng(msa.bfloat16(), mask)

    def rel(x):
        return ((x.double() - want).norm() / want.norm()).item()

    assert torch.isfinite(got).all()
    assert rel(got) <= max(1.5 * rel(base), 1e-2), f"{name}: engine {rel(got):.2e} vs bf16 statements {rel(base):.2e}"

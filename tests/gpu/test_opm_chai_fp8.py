"""EXPERIMENTAL e4m3 grouped product of Chai-1's OuterProductMean (``kernels.outer_product_mean.chai.f8_products``, docs/fp8.md "Chai-1 grouped OPM"):
the transpose-quantize pass, the route's error against the bf16 route, the default (gate at "never") and the switch off = default bytes."""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.kernels.integrations import opm_fp8
from miniworld_engine.kernels.outer_product_mean import chai
from miniworld_engine.modules.msa import OuterProductMean

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _module(seed: int = 1):
    m = OuterProductMean(64, 256, 64, groups=8, product_norm_eps=0.1).cuda().eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif n.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    return m.to(torch.bfloat16).set_backend("cuda")


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90)")
def test_quant_transpose_pass() -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(8, 128, 64, device="cuda", generator=g).bfloat16()
    inv = torch.tensor([4.0], device="cuda")
    out = chai._f8_ext().chai_quant_t(a, inv)
    ref = (a.float() * 4.0).to(torch.float8_e4m3fn).transpose(1, 2)
    assert out.shape == (8, 64, 128) and torch.equal(out.view(torch.uint8), ref.contiguous().view(torch.uint8))


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90)")
def test_route_close_to_bf16_and_off_is_default() -> None:
    m = _module()
    g = torch.Generator(device="cpu").manual_seed(3)
    msa = torch.randn(1, 1024, 128, 64, generator=g).cuda().to(torch.bfloat16)
    old = (opm_fp8.POLICY, chai.F8_MIN_S)
    try:
        opm_fp8.POLICY = None
        with torch.no_grad():
            y0 = m(msa, None)
        opm_fp8.POLICY, chai.F8_MIN_S = "bound", 1 << 40  # the shipped default depth gate: never
        opm_fp8.GUARD_STATS["calls"] = 0
        with torch.no_grad():
            assert torch.equal(m(msa, None), y0)
        assert opm_fp8.GUARD_STATS["calls"] == 0
        chai.F8_MIN_S = 0
        with torch.no_grad():
            y8 = m(msa, None)
        assert opm_fp8.GUARD_STATS["calls"] == 1
    finally:
        opm_fp8.POLICY, chai.F8_MIN_S = old
    assert torch.isfinite(y8).all()
    rel = (y8.float() - y0.float()).pow(2).mean().sqrt() / y0.float().pow(2).mean().sqrt()
    assert 0 < rel < 0.2, rel

"""Bidirectional wide TriMul inference (D256/384/512) takes the fused wide K1/K3 path and
matches the Triton path within the bf16 band, with and without a sequence mask."""
import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_inference as W
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _module(d):
    torch.manual_seed(7)
    m = BidirectionalTriangleMultiplication(d, implementation=I.MINIWORLD, p_drop=0.0).cuda()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 1:
                p.copy_((1.0 if n.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
                p.data = p.data.to(torch.bfloat16)
    return m.eval()


@pytest.mark.parametrize("d", [256, 384, 512])
@pytest.mark.parametrize("masked", [False, True])
def test_wide_inference_matches_triton(d, masked, monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    n = 384
    m = _module(d)
    x = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, n, device="cuda") > 0.1) if masked else None
    calls = []
    real = W.wide_inference
    monkeypatch.setattr(W, "wide_inference", lambda *a: calls.append(1) or real(*a))
    with torch.no_grad():
        y = m(x, mask=mask)
        assert calls, "default dispatch did not take the wide K1/K3 inference path"
        try:
            settings.configure(engine_backend="triton")
            ref = m(x, mask=mask)
        finally:
            settings.configure(engine_backend="auto")
    delta, want = (y - x).float(), (ref - x).float()
    assert float((delta - want).norm() / want.norm()) < 2e-2


def test_wide_inference_compiles_to_one_graph():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    n, d = 384, 256
    m = _module(d)
    x = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        eager = m(x)
        compiled = torch.compile(m, fullgraph=True, dynamic=False)(x)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)

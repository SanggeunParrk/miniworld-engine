"""No-grad recycles retain dropout semantics while omitting D128 LN saves."""
import pytest
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import h100_training as H

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

@pytest.mark.parametrize("width", [64, 128])
def test_nograd_dropout_matches_saved_forward(width):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    torch.manual_seed(91)
    n, d = 384, width
    x = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(h, k, device="cuda", dtype=x.dtype) * k**-.5
               for h, k in [(2*d, d)]*4 + [(d, d), (d, 2*d)]]
    if d == 128:
        weights[:4] = [w.t().contiguous().t() for w in weights[:4]]
    norms = [torch.ones(c, device="cuda") if i % 2 == 0 else torch.zeros(c, device="cuda")
             for i, c in enumerate((d, d, 2*d, 2*d))]
    leaves = [x, *weights, *norms]
    mask = (torch.rand(n, n, device="cuda") > .1).to(x.dtype)
    scale = (torch.rand(n, d, device="cuda") > .25).to(x.dtype) / .75
    expected = H.forward(leaves, mask, scale)[0]
    if d != 128:
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_width import Training
        legacy = Training(*leaves, mask, scale, torch.empty_like(x)).forward()
        torch.testing.assert_close(expected, legacy, rtol=0, atol=0)
    with torch.no_grad():
        actual = H.bidirectional_trimul(*leaves, mask, scale)
        compiled = torch.compile(H.bidirectional_trimul, fullgraph=True,
                                 options={"triton.cudagraphs": False})(*leaves, mask, scale)
    for result in (actual, compiled):
        error = (result.float() - expected.float()).norm()
        assert error / expected.float().norm().clamp_min(1e-12) < 2e-6

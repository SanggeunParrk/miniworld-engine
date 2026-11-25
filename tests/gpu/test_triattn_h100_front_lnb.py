"""H100 TriangleAttention inference at d_pair 384 x 12 heads of 32 channels (fp16 core, unfused front): the pair bias is made inside the LayerNorm
kernel (``ln_bias_fwd``, ``h100_sm90_generic.FRONT_LNB``; the bias GEMM on mma.sync, the bits of the cuBLAS fp16 GEMM) and the module output is BIT-IDENTICAL
to the LayerNorm + cuBLAS bias GEMM + bias kernel route (starting / ending node, a key mask). L 2048 takes the row-blocked path (``ln_bias_blocks``)."""

import pytest
import torch

from foldjet_engine.kernels.triangle_attention.cuda import h100_sm90_generic as gen
from foldjet_engine.modules.triangle import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _module(starting: bool) -> TriangleAttention:
    torch.manual_seed(0)
    m = TriangleAttention(384, 12, d_hidden=12 * 32, starting=starting, master_weights=True, backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0, p.shape[1] ** -0.5)
            elif p is not m.ln_pair.weight:
                p.normal_(0, 0.3)
        m.ln_pair.weight.uniform_(0.5, 1.5)
    return m


@pytest.mark.parametrize("length", [384, 1024, 2048])
@pytest.mark.parametrize("starting", [True, False])
def test_ln_bias_front_bit_identical_to_gemm_bias(length: int, starting: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    m = _module(starting)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(1, length, length, 384, device="cuda", generator=g) * 3 + 1).bfloat16()
    mask = torch.rand(1, length, device="cuda", generator=g) > 0.1
    outs = {}
    with torch.no_grad():
        for flag in (False, True):
            monkeypatch.setattr(gen, "FRONT_LNB", flag)
            outs[flag] = m(x, mask)
    assert torch.equal(outs[False], outs[True])

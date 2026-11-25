"""H100 TriMul inference (bf16) at D64 / D128, any L: the K3 built with the k3.cu pipeline macros (``h100_b200style.K3_EXTRA``: K3_XSPF, K3_XEARLY,
K3_DESC, K3_EPI2) is BIT-IDENTICAL to the plain K3 (one direction, a key mask, L not a multiple of 64 included)."""

import pytest
import torch

from foldjet_engine.kernels.trimul_inproj.cuda import h100_b200style as BS
from foldjet_engine.modules.triangle import TriangleMultiplication

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("length", [200, 768, 1000])
@pytest.mark.parametrize("outgoing", [True, False])
def test_k3_pipe_bit_identical_to_plain_k3(d: int, length: int, outgoing: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    assert BS.K3_EXTRA.get(d), "the pipeline macros are the default of this width"
    torch.manual_seed(0)
    m = TriangleMultiplication(d, outgoing=outgoing, master_weights=True, backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, length, length, d, device="cuda", generator=g).bfloat16()
    mask = torch.rand(1, length, device="cuda", generator=g) > 0.1
    with torch.no_grad():
        piped = m(x, mask)
        monkeypatch.setattr(BS, "K3_EXTRA", {})
        plain = m(x, mask)
    assert torch.equal(piped, plain)

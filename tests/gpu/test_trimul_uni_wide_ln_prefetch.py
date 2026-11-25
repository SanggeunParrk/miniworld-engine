"""H100 TriMul single-direction D512 training: the output-LN backward with the next row tile's TMA loads in flight
(``output_ln_uni_pf.cu``) is BIT-IDENTICAL to the single-buffered source it replaces (``output_ln_d512_l384.cu``): output, input gradient and
every parameter gradient of the module (outgoing / incoming, L384 and L768, fp32 masters and bf16 parameters, a masked sample)."""

import pytest
import torch

from foldjet_engine.kernels.trimul_inproj.cuda import h100_uni_wide_training as UWT
from foldjet_engine.modules.triangle import TriangleMultiplication

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _run(m, x, mask, dy):
    for p in (x, *m.parameters()):
        p.grad = None
    y = m(x, mask)
    y.backward(dy)
    return [y.detach(), x.grad, *[p.grad for p in m.parameters()]]


@pytest.mark.parametrize("master", [True, False])
@pytest.mark.parametrize("length", [384, 768])
@pytest.mark.parametrize("outgoing", [True, False])
def test_prefetching_output_ln_is_bit_identical(outgoing: bool, length: int, master: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    torch.manual_seed(0)
    m = TriangleMultiplication(512, outgoing=outgoing, master_weights=master, backend="cuda").cuda().train()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 512**-0.5) if p.dim() > 1 else p.normal_(1.0, 0.1)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, length, length, 512, device="cuda", generator=g).bfloat16().requires_grad_()
    mask = torch.rand(1, length, device="cuda", generator=g) > 0.08
    dy = torch.randn(1, length, length, 512, device="cuda", generator=g).bfloat16()
    assert UWT._LN_PF
    torch.manual_seed(3)
    served = _run(m, x, mask, dy)
    monkeypatch.setattr(UWT, "_LN_PF", False)
    torch.manual_seed(3)
    single_buffered = _run(m, x, mask, dy)
    for a, b in zip(served, single_buffered, strict=True):
        assert torch.equal(a, b)

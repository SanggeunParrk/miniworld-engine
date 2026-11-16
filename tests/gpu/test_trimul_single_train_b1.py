"""H100 TriMul single-direction D128 training: the on-chip B1 kernel's vectorised (bf16x2) epilogue and interleaved LN rows (``B1_VEC``,
``B1_LNI`` of ``single_output/kernel.cu``) are BIT-IDENTICAL to the scalar one-row-at-a-time loops they replace: output, input gradient and every
parameter gradient of the module (outgoing / incoming, B 1 and 2, fp32 masters and bf16 parameters)."""

import pytest
import torch

from miniworld_engine.kernels.trimul_inproj.cuda import h100_single_output as SO
from miniworld_engine.modules.triangle import TriangleMultiplication

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _run(m, x, mask, dy):
    for p in (x, *m.parameters()):
        p.grad = None
    y = m(x, mask)
    y.backward(dy)
    return [y.detach(), x.grad, *[p.grad for p in m.parameters()]]


@pytest.mark.parametrize("master", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("length", [256, 384])
@pytest.mark.parametrize("outgoing", [True, False])
def test_b1_vector_epilogue_and_interleaved_ln_are_bit_identical(
    outgoing: bool, length: int, batch: int, master: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    torch.manual_seed(0)
    m = TriangleMultiplication(128, outgoing=outgoing, master_weights=master, backend="cuda").cuda().train()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 128**-0.5) if p.dim() > 1 else p.normal_(1.0, 0.1)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(batch, length, length, 128, device="cuda", generator=g).bfloat16().requires_grad_()
    mask = torch.rand(batch, length, device="cuda", generator=g) > 0.08
    dy = torch.randn(batch, length, length, 128, device="cuda", generator=g).bfloat16()
    torch.manual_seed(3)
    served = _run(m, x, mask, dy)
    original_build = SO.build
    monkeypatch.setattr(SO, "build", lambda debug=False, master=False: original_build(debug, master, ("-DB1_VEC=0", "-DB1_LNI=1")))
    torch.manual_seed(3)
    scalar = _run(m, x, mask, dy)
    for a, b in zip(served, scalar, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("master", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("outgoing", [True, False])
def test_b1_packed_gate_epilogue_is_bit_identical(outgoing: bool, batch: int, master: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """``B1_PK`` (default on): one cvt.rn.bf16x2 per pair of roundings and one bf16x2 multiply for dy * dropout scale, equal to the per-element path."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    length = 256
    torch.manual_seed(0)
    m = TriangleMultiplication(128, outgoing=outgoing, master_weights=master, backend="cuda").cuda().train()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 128**-0.5) if p.dim() > 1 else p.normal_(1.0, 0.1)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(batch, length, length, 128, device="cuda", generator=g).bfloat16().requires_grad_()
    mask = torch.rand(batch, length, device="cuda", generator=g) > 0.08
    dy = torch.randn(batch, length, length, 128, device="cuda", generator=g).bfloat16()
    torch.manual_seed(3)
    served = _run(m, x, mask, dy)
    original_build = SO.build
    monkeypatch.setattr(SO, "build", lambda debug=False, master=False: original_build(debug, master, ("-DB1_PK=0",)))
    torch.manual_seed(3)
    scalar = _run(m, x, mask, dy)
    for a, b in zip(served, scalar, strict=True):
        assert torch.equal(a, b)

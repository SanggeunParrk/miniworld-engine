"""The TRITON bidirectional trimul must run FP32 inference, not only BF16.

`_back_kernel` is registered BF16 only, so FP32 inference has to leave it for the split
LN+GEMM / gate pair -- the same one the training Function already uses for FP32.
"""
import pytest
import torch

from miniworld_engine.modules import BidirectionalTriangleMultiplication
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu


def build(implementation):
    torch.manual_seed(929)
    m = BidirectionalTriangleMultiplication(128, implementation=implementation).cuda().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "ln_" not in name:
                p.normal_(std=128**-.5)
        m.ln_pair.bias.fill_(.2)
        m.ln_out.bias.fill_(.3)
    return m


@pytest.mark.parametrize("masked", [False, True])
def test_fp32_inference_matches_pytorch_and_training_path(masked):
    triton_m = build(ImplementationType.TRITON)
    torch_m = build(ImplementationType.PYTORCH)
    torch_m.load_state_dict(triton_m.state_dict())
    x = torch.randn(1, 48, 48, 128, device="cuda")
    mask = None
    if masked:
        mask = torch.ones(1, 48, device="cuda", dtype=torch.bool)
        mask[:, ::5] = False

    with torch.no_grad():
        inference = triton_m(x, mask)
        reference = torch_m(x, mask)
    training = triton_m(x, mask).detach()     # grad on: the merged autograd Function

    assert inference.dtype == torch.float32
    assert torch.isfinite(inference).all()
    # cuBLAS / Triton dots run FP32 as TF32, so this is a TF32 band, not bitwise.
    torch.testing.assert_close(inference, reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(inference, training, rtol=1e-5, atol=1e-5)

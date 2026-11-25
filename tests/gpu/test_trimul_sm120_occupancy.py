"""RTX 5090 (sm_120a) TriMul inference: the output stages and the one-CTA-per-tile front of the D64 / D128 paths are sized for two (k3s, k3w BN 64) / three (k1t) CTAs an SM
in the 100 KB of an SM (the statistics exchange of k3s / k3w shares one area, k1t keeps its statistics in registers), and D 128 inference with in-kernel statistics runs k3w as
two BN 64 CTAs per token tile.  The MMAs, their k order and the epilogue arithmetic are those of the larger forms: the outputs are bitwise equal."""

import copy

import pytest
import torch

from foldjet_engine.measure import config as C
from foldjet_engine.modules import TriangleMultiplication

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="needs an RTX 5090 (sm_120)"
    ),
]


@pytest.mark.parametrize("length", [256, 384, 768])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("d_pair,d_hidden", [(64, 64), (64, 128), (128, 128)])
def test_two_per_sm_forms_bitwise(d_pair: int, d_hidden: int, outgoing: bool, length: int) -> None:
    from foldjet_engine.kernels.trimul_inproj.cuda import sm80_wide

    ext = sm80_wide._lib()
    module = TriangleMultiplication(d_pair=d_pair, d_hidden=d_hidden, outgoing=outgoing).cuda().float().eval()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    module.set_master_weights()
    inputs = module.make_inputs(
        length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)
    )
    kern = copy.deepcopy(module).set_backend("cuda")

    def run() -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            assert kern.unsupported(**inputs) is None
            return kern(**inputs).clone()

    try:
        new = run()
        ext.pad_set(1)  # the shared-memory sizes of the old forms: one CTA an SM
        ext.k3w64_set(0)  # D 128: the one-CTA BN 128 output stage
        old = run()
    finally:
        ext.pad_set(0)
        ext.k3w64_set(1)
    assert torch.equal(new, old)

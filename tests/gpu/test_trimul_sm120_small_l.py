"""RTX 5090 (sm_120a) TriMul D64 inference, hidden 128, L <= 256: the front is the two-group persistent ``k1w64_kernel`` (the one-CTA-per-tile
``k1t_kernel`` leaves the card empty at 128 / 512 token tiles) and at L 128 the output stage is the k3w ZPRE form; both give the output bits of the
forms they replace (the same MMAs in the same k order per element, the same statistics and epilogue arithmetic)."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import TriangleMultiplication

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="needs an RTX 5090 (sm_120)"
    ),
]


@pytest.mark.parametrize("length", [128, 256])
@pytest.mark.parametrize("outgoing", [True, False])
def test_small_l_front_and_output_stage_bitwise(outgoing: bool, length: int) -> None:
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    ext = sm80_wide._lib()
    module = TriangleMultiplication(d_pair=64, d_hidden=128, outgoing=outgoing).cuda().float().eval()
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
        default = run()  # the rule: k1w64 front (and, at L 128, k3w ZPRE)
        ext.k1t_set(1)  # force the one-CTA-per-tile k1t front
        old_front = run()
        ext.k1t_set(-1)
        ext.k3s_set(0)  # k3w ZPRE at every length
        zpre = run()
    finally:
        ext.k1t_set(-1)
        ext.k3s_set(1)
    assert torch.equal(default, old_front)
    assert torch.equal(default, zpre)

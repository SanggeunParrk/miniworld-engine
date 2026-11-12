"""FoldForge 5I28 (A100, OpenDDE n = 12): the structural-token refiner's pair is [S, S, 384] with S = 2944 (3.3e9 elements > 2**31). The
Triton TriMul's fused back half indexed its channel-major contraction in int32 (``channel * L * L``): an illegal address on A100, a
silently wrong result on H100. Inference now indexes in 64 bits above the bound; training refuses such a plane with a reason.

The reference of a plane that large does not fit next to the kernel's, so the check uses a symmetry instead: tiling the input R x R over
the token axes leaves every (P x P) block of TriMul's output equal to the output at P (the contraction sums R equal copies, a constant
factor the output LayerNorm removes). Each block of the big output is compared with the float32 reference at P.
"""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import TriangleMultiplication

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

P, R, D = 300, 8, 384  # L = 2400: 2.2e9 pair elements (> 2**31), ~27 GB peak


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def _module(outgoing: bool) -> TriangleMultiplication:
    m = TriangleMultiplication(D, outgoing=outgoing).cuda().float().eval()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.set_master_weights()
    return m


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
@pytest.mark.parametrize("outgoing", [True, False])
def test_triton_inference_past_2_31_elements(outgoing: bool) -> None:
    base = _module(outgoing)
    gen = torch.Generator(device="cuda").manual_seed(1)
    small = base.make_inputs(length=P, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    with torch.no_grad():
        gold = copy.deepcopy(base).float()(small["pair"].float(), small["mask"]).float()
        kern = copy.deepcopy(base).set_backend("triton")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            at_p = kern(**small).float()
        pair = small["pair"].repeat(1, R, R, 1).contiguous()
        mask = small["mask"].repeat(1, R)
        assert pair.numel() >= 2**31
        assert kern.unsupported(pair, mask) is None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            big = kern(pair, mask)
    scale = gold.abs().max()
    err_p = float((at_p - gold).abs().max() / scale)
    for i in range(R):
        for j in (0, R - 1):
            block = big[:, i * P : (i + 1) * P, j * P : (j + 1) * P].float()
            err = float((block - gold).abs().max() / scale)
            assert err <= max(1.5 * err_p, 8e-3), f"block ({i}, {j}): {err:.3e} (at P: {err_p:.3e})"


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
def test_triton_training_past_2_31_elements_is_refused() -> None:
    kern = _module(True).set_backend("triton").train()
    pair = torch.empty(1, 2400, 2400, D, device="cuda", dtype=torch.bfloat16)
    why = kern.unsupported(pair, None)
    assert why is not None and "2**31" in why
    with torch.no_grad():
        assert kern.eval().unsupported(pair, None) is None

"""A100 (sm_80) CUDA TriangleAttention past 2**31 pair elements: the token / element offsets of the front, the attention core and the back
are 64-bit (the front's and back's token index ``z L^2 + ...`` was 32-bit unsigned before; the element offsets already were 64-bit).

The float32 reference of a plane that large does not fit next to the kernel's, so the check uses a symmetry instead: tiling the pair R x R
over the token axes leaves every (P x P) block of the output equal to the output at P (each query sees its P keys R times: the softmax
weights divide by R and the R copies sum back). Each checked block of the big output, including the last one (the highest offsets), is
compared with the float32 reference at P.
"""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.a100, pytest.mark.slow]

#: (d_pair, n_head, P, R): L = P R, L^2 d_pair > 2**31 -- d_pair 128: L 4224 (2.28e9), d_pair 384 (the wide path): L 2560 (2.52e9)
CASES = [(128, 4, 384, 11), (384, 12, 256, 10)]


def _module(d_pair: int, n_head: int, starting: bool) -> TriangleAttention:
    m = TriangleAttention(d_pair, n_head, starting=starting).cuda().float().eval()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.set_master_weights()
    return m


@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize(("d_pair", "n_head", "P", "R"), CASES)
def test_cuda_inference_past_2_31_elements(d_pair: int, n_head: int, P: int, R: int, starting: bool) -> None:
    base = _module(d_pair, n_head, starting)
    gen = torch.Generator(device="cuda").manual_seed(1)
    small = base.make_inputs(length=P, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    small["mask"][:, P - 7 :] = False  # a masked tail: the tiled mask repeats it in every copy
    with torch.no_grad():
        gold = copy.deepcopy(base).float()(small["pair"].float(), small["mask"]).float()
        kern = copy.deepcopy(base).set_backend("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert kern.unsupported(**small) is None
            at_p = kern(**small).float()
        pair = small["pair"].repeat(1, R, R, 1).contiguous()
        mask = small["mask"].repeat(1, R)
        assert pair.numel() >= 2**31
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert kern.unsupported(pair, mask) is None
            big = kern(pair, mask)
        del pair
    scale = gold.abs().max()
    err_p = float((at_p - gold).abs().max() / scale)
    for i in (0, R // 2, R - 1):
        for j in (0, R - 1):
            block = big[:, i * P : (i + 1) * P, j * P : (j + 1) * P].float()
            err = float((block - gold).abs().max() / scale)
            assert err <= max(2.0 * err_p, 1e-2), f"block ({i}, {j}): {err:.3e} (at P: {err_p:.3e})"

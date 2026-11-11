"""R20 (FoldForge): the H100 atom-DiT block output must accept an in-place write while autograd records (a custom-Function output
that is a view refuses it), and the gradients must be those of the out-of-place form."""

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.atom_dit import AtomDiTProtenix

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) atom-DiT block")
@pytest.mark.parametrize("length", [1024, 1000])  # unpadded (a reshaped view before the fix) and padded rows
@pytest.mark.parametrize("key_window", ["pad", "slide"])
def test_block_output_takes_inplace_write(length: int, key_window: str) -> None:
    torch.manual_seed(0)
    m = AtomDiTProtenix(d_single=128, d_cond=128, d_pair=16, n_head=4, n=2, key_window=key_window).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m = typing.cast("AtomDiTProtenix", C.cast(m, "master").train(True))
    m.set_backend("cuda")
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=length, samples=8, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    grads = []
    for inplace in (False, True):
        x = inp["single"].detach().clone().requires_grad_(True)
        pair = inp["pair"].detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(single=x, cond=inp["cond"], pair=pair, mask=inp["mask"])
        if inplace:
            y.mul_(2.0)
        else:
            y = y * 2.0
        y.float().square().mean().backward()
        grads.append((x.grad, pair.grad))
    for a, b in zip(grads[0], grads[1], strict=True):
        assert torch.equal(a, b)

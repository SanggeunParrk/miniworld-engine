"""A100 (sm_80) TriMul D64 inference output stage: the 64-token single-stage ``k3s_kernel`` gives the same output bits as the k3w ZPRE form it
replaces (the same MMAs in the same k order per element, the same two-pass LayerNorm_out statistics), at hidden width 64 and 128, both directions,
the RoseTTAFold3 variant, padded inputs (the measurers' masks)."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import TriangleMultiplication, TriangleMultiplicationRF3

pytestmark = [pytest.mark.gpu, pytest.mark.a100]


def _module(cls, outgoing: bool, d_hidden: int | None):
    kw = {"d_pair": 64, "outgoing": outgoing}
    if d_hidden is not None:
        kw["d_hidden"] = d_hidden
    m = cls(**kw).cuda().float().eval()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.set_master_weights()
    return m


@pytest.mark.parametrize("length", [256, 640])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize(
    ("cls", "d_hidden"), [(TriangleMultiplication, None), (TriangleMultiplication, 128), (TriangleMultiplicationRF3, None)]
)
def test_k3s_bitwise_k3w(cls, d_hidden, outgoing: bool, length: int) -> None:
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    ext = sm80_wide._ext()
    base = _module(cls, outgoing, d_hidden)
    gen = torch.Generator(device="cuda").manual_seed(1)
    inputs = base.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    kern = copy.deepcopy(base).set_backend("cuda")
    outs = []
    try:
        for flag in (0, 1):
            ext.k3s_set(flag)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                assert kern.unsupported(**inputs) is None
                outs.append(kern(**inputs).clone())
    finally:
        ext.k3s_set(1)
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("length", [640, 1536])
@pytest.mark.parametrize("outgoing", [True, False])
def test_k1t_bitwise_k1w64(outgoing: bool, length: int) -> None:
    """The hidden-128 front in one CTA per token tile (``k1t_kernel``; 2 ring slots below L 1536, 3 from it) gives the output bits of the two-group
    persistent ``k1w64_kernel``."""
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    ext = sm80_wide._ext()
    base = _module(TriangleMultiplication, outgoing, 128)
    gen = torch.Generator(device="cuda").manual_seed(2)
    inputs = base.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    kern = copy.deepcopy(base).set_backend("cuda")
    outs = []
    try:
        for flag in (0, -1):
            ext.k1t_set(flag)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outs.append(kern(**inputs).clone())
    finally:
        ext.k1t_set(-1)
    assert torch.equal(outs[0], outs[1])

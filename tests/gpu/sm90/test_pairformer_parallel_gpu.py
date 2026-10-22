"""The parallel Pairformer block (Chai-1, RoseTTAFold3: every part reads the block input) and its CUDA residual sum on H100."""

import copy

import pytest
import torch

from miniworld_engine.architecture.blocks.pairformer import (
    PairformerBlock,
    _parallel_residual,
)
from miniworld_engine.modules import (
    AttentionPairBias,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_parallel_residual_is_one_rounding(dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(2, 96, 96, 128, device="cuda", generator=g).to(dtype)
    ys = [(x.float() + 0.1 * torch.randn(x.shape, device="cuda", generator=g)).to(dtype) for _ in range(5)]
    gold = x.double() + sum(y.double() - x.double() for y in ys)
    got = _parallel_residual(x, ys)
    assert got.dtype == dtype
    torch.testing.assert_close(got.double(), gold.to(dtype).double(), rtol=0, atol=1e-5 if dtype == torch.float32 else 2e-2)


def _block(impl):
    torch.manual_seed(0)
    block = PairformerBlock.from_components(
        triangle_outgoing=TriangleMultiplication(128, outgoing=True, implementation=impl, p_drop=0.0),
        triangle_incoming=TriangleMultiplication(128, outgoing=False, implementation=impl, p_drop=0.0),
        attention_starting=TriangleAttention(128, 4, starting=True, implementation=impl, p_drop=0.0),
        attention_ending=TriangleAttention(128, 4, starting=False, implementation=impl, p_drop=0.0),
        transition_pair=Transition(128, 4, implementation=impl),
        attention_single=AttentionPairBias(384, 128, 16, implementation=impl),
        transition_single=Transition(384, 4, implementation=impl),
        parallel=True,
    ).cuda().eval()
    with torch.no_grad():
        for name, p in block.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "ln" in name and p.ndim == 1 and name.endswith("weight") else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.1 * torch.randn_like(p))
    return block


def test_parallel_schedule_reads_the_block_input():
    block = _block(ImplementationType.PYTORCH).double()
    z = torch.randn(1, 64, 64, 128, device="cuda", dtype=torch.float64)
    s = torch.randn(1, 64, 384, device="cuda", dtype=torch.float64)
    mask = torch.ones(1, 64, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        pair, single = block(z, s, mask)
        parts = [block.tri_multi_outgoing(z, mask), block.tri_multi_incoming(z, mask), block.tri_atten_starting(z, mask),
                 block.tri_atten_ending(z, mask), block.transition_pair(z)]
        want_pair = z + sum(y - z for y in parts)
        want_single = s + (block.attention_pair_bias(s, z, mask) - s) + (block.transition_single(s) - s)
    torch.testing.assert_close(pair, want_pair, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(single, want_single, rtol=1e-10, atol=1e-10)


def test_parallel_block_on_the_engine_kernels():
    eng = _block(ImplementationType.MINIWORLD)
    gold_block = _block(ImplementationType.PYTORCH)
    gold_block.load_state_dict(eng.state_dict())
    ref16 = copy.deepcopy(gold_block).to(torch.bfloat16)
    gold_block = gold_block.double()
    eng = eng.to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(3)
    z = torch.randn(1, 256, 256, 128, device="cuda", generator=g)
    s = torch.randn(1, 256, 384, device="cuda", generator=g)
    mask = (torch.arange(256, device="cuda") < 230)[None]
    with torch.no_grad():
        gp, gs = gold_block(z.double(), s.double(), mask)
        bp, bs = ref16(z.bfloat16(), s.bfloat16(), mask)
        ep, es = eng(z.bfloat16(), s.bfloat16(), mask)
    rows = (slice(None), slice(0, 230), slice(0, 230))

    def rel(a, b):
        return ((a.double() - b).norm() / b.norm()).item()

    assert rel(ep[rows], gp[rows]) <= max(2 * rel(bp[rows], gp[rows]), 1e-2)
    assert rel(es[:, :230], gs[:, :230]) <= max(2 * rel(bs[:, :230], gs[:, :230]), 1e-2)

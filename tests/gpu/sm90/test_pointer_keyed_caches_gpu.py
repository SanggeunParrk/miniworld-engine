"""A new tensor that lands on a freed tensor's address (and version) is never served the old tensor's cached table.

The H100 inference paths cache a table made from the pair (its bias / softmax) per pair tensor, keyed on the pair's address and version
-- the same at every diffusion step of one sample. A caching allocator hands a freed block to the next allocation of that size, so the
next sample's pair lands on the same address with the same version, and a cache that does not hold the old pair serves the old table.
"""

import pytest
import torch

from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90a) required",
    ),
]

L, A = 256, 2


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def randomized(block):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in block.parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[1] ** -0.5)
            elif p.ndim == 1 and p.numel() > 1:
                p.add_(torch.randn_like(p) * 0.1)
    return block


def inputs(dtype):
    torch.manual_seed(1)
    single = torch.randn(A, 1, L, 768, device="cuda", dtype=dtype)
    cond = torch.randn(1, 1, L, 384, device="cuda", dtype=dtype).expand(A, 1, L, 384)
    mask = torch.rand(1, L, device="cuda") > 0.2
    return single, cond, mask


def two_pairs_at_one_address(dtype):
    first = torch.randn(1, L, L, 128, device="cuda", dtype=dtype)
    address = first.data_ptr()
    return first, address


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("make", ["bias_only", "token"])
def test_the_next_pair_is_not_served_the_previous_pairs_table(dtype, make):
    if make == "bias_only":
        block = BiasOnlyDiTBlock(n_head=16, d_head=48, implementation=ImplementationType.MINIWORLD)
        oracle = BiasOnlyDiTBlock(n_head=16, d_head=48, implementation=ImplementationType.PYTORCH)
    else:
        block = DiTBlock(768, 384, 128, 16, 2, implementation=ImplementationType.MINIWORLD)
        oracle = DiTBlock(768, 384, 128, 16, 2, implementation=ImplementationType.PYTORCH)
    block = randomized(block).cuda().to(dtype)
    oracle = oracle.cuda().float()
    oracle.load_state_dict({k: v.float() for k, v in block.state_dict().items()})
    single, cond, mask = inputs(dtype)

    first, address = two_pairs_at_one_address(dtype)
    with torch.no_grad():
        block(single, cond, first, mask)           # fills the cache
    del first
    torch.cuda.synchronize()
    second = torch.randn(1, L, L, 128, device="cuda", dtype=dtype) * 3
    if second.data_ptr() != address:
        pytest.skip("the allocator did not reuse the address")
    with torch.no_grad():
        got = block(single, cond, second, mask)
        want = oracle(single.float(), cond.float(), second.float(), mask)
    assert rel(got, want) < (2e-2 if dtype is torch.bfloat16 else 5e-3)

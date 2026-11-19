"""A100 (sm_80) OuterProductMean's masked outer product helpers: the dead-token liveness kernel (``ogemm_sm80.live_tokens``: one launch instead of
torch's reduction over the MSA rows) and the batched skip product of the Chai-1 groups (``ogemm_sm80.outer_batched``: one launch for the 8 groups,
bit for bit the eight per-group launches)."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.a100]


def _a100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


@pytest.mark.skipif(not _a100(), reason="an A100 (sm_80)")
@pytest.mark.parametrize("rows", [1, 7, 384, 1024, 2048, 3000])
@pytest.mark.parametrize("length", [4, 36, 100, 128, 384, 640])
def test_live_tokens_matches_the_row_max(rows: int, length: int) -> None:
    from miniworld_engine.kernels.outer_product_mean.cuda import ogemm_sm80

    gen = torch.Generator(device="cuda").manual_seed(rows * 1000 + length)
    mask = (torch.rand(rows, length, device="cuda", generator=gen) > 0.7).to(torch.uint8)
    mask[:, length - length // 8 :] = 0  # padding positions: dead in every row
    assert torch.equal(ogemm_sm80.live_tokens(mask), mask.amax(0).contiguous())


@pytest.mark.skipif(not _a100(), reason="an A100 (sm_80)")
@pytest.mark.parametrize("tokens,depth", [(384, 128), (512, 256), (768, 64)])
def test_batched_skip_product_is_the_per_group_launches(tokens: int, depth: int) -> None:
    from miniworld_engine.kernels.outer_product_mean.cuda import ogemm_sm80

    gen = torch.Generator(device="cuda").manual_seed(tokens)
    a = torch.randn(8, depth, 8 * tokens, device="cuda", dtype=torch.bfloat16, generator=gen)
    b = torch.randn(8, depth, 8 * tokens, device="cuda", dtype=torch.bfloat16, generator=gen)
    live = (torch.rand(tokens, device="cuda", generator=gen) > 0.25).to(torch.uint8)
    live[:8] = 1
    live[-tokens // 8 :] = 0
    dead = (live == 0).repeat_interleave(8)  # a dead token has a = b = 0 (the prologue masks them)
    a[:, :, dead] = 0
    b[:, :, dead] = 0
    batched = ogemm_sm80.outer_batched(a, b, live, 8)
    each = torch.empty_like(batched)
    for g in range(8):
        ogemm_sm80.outer(a[g], b[g], live, 8, each[g])
    assert torch.equal(batched, each)
    dense = torch.bmm(a.transpose(1, 2), b)
    assert (batched.float() - dense.float()).abs().max().item() <= 1e-2 * dense.float().abs().max().item()

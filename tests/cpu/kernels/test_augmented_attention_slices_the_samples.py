"""A declared shape that would run out of memory splits the augmentation axis instead (registry rule: never OOM, never a fallback).

``AugmentedAttentionPairBias.delta`` slices A when the fp32 logits [A, B, H, L, L] exceed ``LOGIT_BUDGET_BYTES``: the atom DiT at 8192 atoms x
48 samples asked for 48 GiB in one allocation on an 80 GB H100. The slice size depends on the shape only, and the samples are independent, so
the sliced call is the unsliced one."""
from __future__ import annotations

import torch

from miniworld_engine.modules.augmented_attention import AugmentedAttentionPairBias


def test_the_slice_size_is_a_function_of_the_shape():
    m = AugmentedAttentionPairBias(d_single=128, d_cond=128, d_pair=16, n_head=4)
    meta = torch.empty(48, 1, 8192, 128, device="meta")
    assert m.augment_chunk(meta) == 8                                   # 1 GiB of logits per sample: 8 per call, 6 calls
    assert m.augment_chunk(torch.empty(48, 1, 4096, 128, device="meta")) == 32
    assert m.augment_chunk(torch.empty(5, 1, 768, 768, device="meta")) == 5   # the token DiT is never sliced


def test_a_sliced_call_is_the_unsliced_one(monkeypatch):
    torch.manual_seed(0)
    m = AugmentedAttentionPairBias(d_single=32, d_cond=16, d_pair=8, n_head=4).double()
    single = torch.randn(6, 1, 12, 32, dtype=torch.float64, requires_grad=True)
    cond = torch.randn(6, 1, 12, 16, dtype=torch.float64)
    pair = torch.randn(1, 12, 12, 8, dtype=torch.float64)
    mask = torch.rand(6, 1, 12) > 0.2
    whole = m.delta(single, cond, pair, mask)
    (g_whole,) = torch.autograd.grad(whole.square().sum(), single)
    monkeypatch.setattr(AugmentedAttentionPairBias, "LOGIT_BUDGET_BYTES", 4 * 12 * 12 * 4 * 4)   # 4 samples per call: slices of 4 + 2
    assert m.augment_chunk(single) == 4
    sliced = m.delta(single, cond, pair, mask)
    (g_sliced,) = torch.autograd.grad(sliced.square().sum(), single)
    torch.testing.assert_close(sliced, whole, rtol=0, atol=1e-12)
    torch.testing.assert_close(g_sliced, g_whole, rtol=0, atol=1e-12)

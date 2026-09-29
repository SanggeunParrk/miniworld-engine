"""The public attention facade must not silently quantize its inputs."""
import sys
from types import SimpleNamespace

import pytest
import torch

from miniworld_engine.kernels.augmented_attention.whole_op import (
    augmented_attention_pair_bias,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_passes_caller_precision_to_the_backend(monkeypatch, dtype):
    query, key, value = [torch.randn(2, 1, 2, 7, 16, dtype=dtype) for _ in range(3)]
    bias = torch.randn(1, 2, 7, 7, dtype=torch.float32)
    seen = []

    def backend(q, k, v, b, mask):
        seen.extend((q, k, v, b))
        return v

    monkeypatch.setitem(
        sys.modules, "miniworld_engine.kernels.augmented_attention.triton.main",
        SimpleNamespace(triton_augmented_attention_pair_bias=backend),
    )
    result = augmented_attention_pair_bias(query, key, value, bias)
    for actual, original in zip(seen[:3], (query, key, value), strict=True):
        torch.testing.assert_close(actual.transpose(2, 3), original, atol=0, rtol=0)
    torch.testing.assert_close(seen[3].permute(0, 3, 1, 2), bias, atol=0, rtol=0)
    torch.testing.assert_close(result, value, atol=0, rtol=0)

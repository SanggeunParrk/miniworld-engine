"""H100 Transition D 768 (n 2) route: inference and training at every row count take the generic kernels, not the wide D768 build (its
training structure: a / b written, [Wa; Wb] packed per call -- 1.6 - 2.1x slower at inference, 1.2 - 1.85x slower per training step above
768 rows). Inference falls back to the wide build above 16384 rows. ``prefers_generic`` reads only dtype and shape, so this runs on CPU tensors."""

import pytest
import torch

from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a as G


@pytest.mark.parametrize("rows", [128, 768, 1000, 1024, 1536, 2048, 4096, 8192, 16384])
def test_d768_inference_prefers_generic(rows: int) -> None:
    assert G.prefers_generic(torch.empty(rows, 768, dtype=torch.bfloat16), training=False, n=2)


@pytest.mark.parametrize("rows", [128, 768, 1000, 1024, 1536, 2048, 4096, 8192, 16384, 32768])
def test_d768_training_prefers_generic(rows: int) -> None:
    assert G.prefers_generic(torch.empty(rows, 768, dtype=torch.bfloat16), training=True, n=2)


@pytest.mark.parametrize("d", [384, 512])
@pytest.mark.parametrize("rows", [16384, 36864, 65536, 147456, 589824])
def test_d384_d512_training_prefers_generic_from_16384(d: int, rows: int) -> None:
    assert G.prefers_generic(torch.empty(rows, d, dtype=torch.bfloat16), training=True, n=2)

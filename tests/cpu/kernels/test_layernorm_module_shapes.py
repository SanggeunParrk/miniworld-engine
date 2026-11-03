"""``modules.primitives.LayerNorm`` takes every ``normalized_shape`` ``nn.LayerNorm`` takes: an int, a list, a ``torch.Size`` and a plain
tuple (``nn.LayerNorm.normalized_shape`` is one; FoldForge's ``configure_model`` hands it over, and a tuple raised ``int()`` on it)."""
import pytest
import torch

from miniworld_engine.modules.primitives import LayerNorm


@pytest.mark.parametrize("shape", [128, [128], torch.Size([128]), (128,), torch.nn.LayerNorm(128).normalized_shape])
def test_every_normalized_shape_form(shape):
    assert LayerNorm(shape).normalized_shape == (128,)

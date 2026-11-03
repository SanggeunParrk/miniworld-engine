"""The engine LayerNorm on an unbatched [L, D] (and a single row) on the portable Triton path: its shape key reads the activation's
(B, L, D) shape, so the module hands it one batch of it (FoldForge's unbatched diffusion activations failed with "already flattened")."""
import dataclasses

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.primitives import LayerNorm

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]


@pytest.mark.parametrize("shape", [(384, 768), (768,)])
def test_an_unbatched_activation_normalizes_like_torch(shape):
    previous = settings.configure(force_portable_kernels=True)
    try:
        norm = LayerNorm(768, implementation=ImplementationType.MINIWORLD).cuda()
        x = torch.randn(*shape, device="cuda")
        want = torch.nn.functional.layer_norm(x, (768,), norm.weight, norm.bias, norm.eps)
        torch.testing.assert_close(norm(x), want, rtol=1e-4, atol=1e-4)
    finally:
        settings.configure(**dataclasses.asdict(previous))

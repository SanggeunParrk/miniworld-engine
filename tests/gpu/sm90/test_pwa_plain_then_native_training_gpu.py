"""MiniWorld's MSA attention (d_msa 64, d_pair 128, 8 heads x 32) trained at a plain-scope length and then at a native-scope one in the
SAME process. The two builds of the H100 PWA sources (plain and ``-DMW_PWA_CUDA_AUX``) define the same template launchers; with default
symbol visibility the second build ran its kernel through the first one's launcher and ``pwa_glue3`` failed with "invalid argument"
(present in 2.4.3; the bundle verify of the MSA-depth rows). Each call must train, and match the module's PyTorch statements.

Needs an H100 (sm_90a); skipped elsewhere.
"""

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0),
        reason="the native PWA training path is sm_90a",
    ),
]

DEPTH = 1024        # an MSA depth with a native training scope at N 384 / 768


def _grads(impl, n):
    torch.manual_seed(0)
    module = MSAPairWeightedAveraging(64, 128, 8, 32, implementation=ImplementationType(impl)).cuda().to(torch.bfloat16).train()
    gen = torch.Generator(device="cuda").manual_seed(1)
    with torch.no_grad():      # trained-like weights: the zero-initialised output projection would make every input gradient zero
        for p in module.parameters():
            if p.ndim >= 2:
                p.copy_((torch.randn(p.shape, device="cuda", generator=gen) * p.shape[-1] ** -0.5).to(p.dtype))
    msa = torch.randn(1, DEPTH, n, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    pair = torch.randn(1, n, n, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    module(msa, pair, mask).float().sum().backward()
    return msa.grad.float(), pair.grad.float()


def _cos(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def test_a_plain_length_then_native_lengths_all_train():
    for n in (256, 384, 768):          # 256 is outside the native scope, 384 / 768 inside it
        got, want = _grads("miniworld", n), _grads("pytorch", n)
        for g, w in zip(got, want, strict=True):
            assert torch.isfinite(g).all()
            assert _cos(g, w) > 0.999, n

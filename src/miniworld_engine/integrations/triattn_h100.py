"""Opt-in Hopper TriangleAttention following B200 front/core/tail fusion.

Set ``module._h100_b200style_cuda = True`` to test this candidate through the
module API. Default dispatch is unchanged until qualification is complete.
"""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.modules.dispatch import KernelBackend


def serves(module, pair, mask=None):
    if not getattr(module, "_h100_b200style_cuda", False):
        return False
    if (
        module._backend != KernelBackend.TRITON
        or settings.current().engine_backend == "triton"
    ):
        return False
    if not module.use_self_attention or module.use_qk_norm:
        return False
    if module.to_query.weight.shape[0] != module.n_head * 32:
        return False
    from miniworld_engine.kernels.triangle_attention.cuda.h100_triattn import supports

    weights = (
        module.to_query.weight,
        module.to_key.weight,
        module.to_value.weight,
        module.to_gate.weight,
        module.to_bias.weight,
        module.to_out.weight,
    )
    return supports(pair, weights, module.ln_pair.weight, module.ln_pair.bias, mask)


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None = None):
    from miniworld_engine.kernels.triangle_attention.cuda.h100_triattn import (
        forward as native,
    )

    return native(module, pair, mask)

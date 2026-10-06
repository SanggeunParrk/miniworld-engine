"""Hopper TriangleAttention following the B200 front/core/tail design.

The measured winning inference buckets use CUDA by default. Other shapes and
training remain explicit candidates. The grouped projection stages retain
cuBLAS intermediates; a fully fused Hopper frontend is still experimental.
"""

from __future__ import annotations

import os

import torch

from miniworld_engine import settings
from miniworld_engine.modules.dispatch import KernelBackend


def validated_inference(module, pair):
    """Select only the inference buckets with complete numerical/timing gates."""
    from miniworld_engine.modules.exceptions import ImplementationType

    # The common runner maps the MiniWorld label to the TRITON family enum.
    # Both family spellings dispatch into this CUDA implementation on Hopper.
    if (module.implementation not in {ImplementationType.MINIWORLD, ImplementationType.TRITON}
            or torch.is_grad_enabled()
            or module.training or not module.starting or not pair.is_cuda
            or torch.cuda.get_device_capability(pair.device) != (9, 0)):
        return False
    width = module.to_query.weight.shape[1]
    return (width in (64, 384, 512)
            or (width == 128 and pair.ndim == 4 and pair.shape[1] in (256, 384)))


def serves(module, pair, mask=None):
    from miniworld_engine.modules.exceptions import ImplementationType

    validated = validated_inference(module, pair)
    if not (validated or getattr(module, "_h100_b200style_cuda", False)
            or (module.implementation in {ImplementationType.MINIWORLD, ImplementationType.TRITON}
                and os.environ.get("MINIWORLD_TRIATTN_H100_CANDIDATE") == "1")):
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
    if validated and (
        pair.ndim != 4 or pair.shape[0] != 1
        or pair.shape[1] not in (128, 256, 384, 512, 640, 768)
        or not pair.is_contiguous()
        or any(not w.is_contiguous() for w in weights)
        or len({w.dtype for w in weights}) != 1
        or module.ln_pair.eps != 1e-5
        or any(t.dtype != torch.float32 or not t.is_contiguous()
               for t in (module.ln_pair.weight, module.ln_pair.bias))
        or (mask is not None and not mask.is_contiguous())
    ):
        return False
    return supports(pair, weights, module.ln_pair.weight, module.ln_pair.bias, mask)


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None = None):
    from miniworld_engine.kernels.triangle_attention.cuda.h100_triattn import (
        forward as native,
    )

    return native(module, pair, mask)

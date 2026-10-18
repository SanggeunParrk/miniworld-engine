"""Hopper TriangleAttention following the B200 front/core/tail design.

Required starting-node inference and training buckets use CUDA by default.
This selection prioritizes correctness and native coverage; it is not a claim
that every bucket wins on latency. Grouped projections retain cuBLAS
intermediates; a fully fused Hopper frontend is still experimental.
"""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.modules.dispatch import KernelBackend


def native_requested(module, pair):
    """Hopper engine calls require native execution or a contract error."""
    from miniworld_engine.modules.exceptions import ImplementationType

    # The common runner maps the MiniWorld label to the TRITON family enum.
    # Both family spellings dispatch into this CUDA implementation on Hopper.
    return (
        module.implementation
        in {ImplementationType.MINIWORLD, ImplementationType.TRITON}
        and pair.is_cuda
        and torch.cuda.get_device_capability(pair.device) == (9, 0)
    )


def serves(module, pair, mask=None):
    validated = native_requested(module, pair)
    if not (validated or getattr(module, "_h100_b200style_cuda", False)):
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
        not module.starting
        or pair.ndim != 4
        or pair.shape[0] != 1
        or pair.shape[1] not in (128, 256, 384, 512, 640, 768)
        or not pair.is_contiguous()
        or any(not w.is_contiguous() for w in weights)
        or len({w.dtype for w in weights}) != 1
        or module.ln_pair.eps != 1e-5
        or not 0.0 <= module.p_drop < 1.0
        or any(
            t.dtype != torch.float32 or not t.is_contiguous()
            for t in (module.ln_pair.weight, module.ln_pair.bias)
        )
        or (mask is not None and not mask.is_contiguous())
    ):
        return False
    return supports(pair, weights, module.ln_pair.weight, module.ln_pair.bias, mask)


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None = None):
    from miniworld_engine.integrations.token_dit_train import _native_caller_output
    from miniworld_engine.kernels.triangle_attention.cuda.h100_triattn import (
        forward as native,
    )

    return _native_caller_output(
        native(module, pair, mask), pair, *module.parameters(), force=True
    )

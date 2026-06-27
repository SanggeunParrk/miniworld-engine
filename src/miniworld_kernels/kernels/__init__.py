"""Flat kernel re-export surface.

Mirrors team-gm's ``team_gm.modules.kernels`` namespace so the vendored module
layers (``miniworld_kernels.modules.layers.*``) can call ``kernels.triton_*``
without knowing the per-op / per-backend folder layout. Each name resolves to
the canonical (``main`` = psk/benchmark) Triton entry point for that op.
"""

from __future__ import annotations

from .adaln.triton.main import triton_adaptive_layer_norm
from .augmented_attention.triton.main import triton_augmented_attention_pair_bias
from .bias_only_attention.triton.gate_out import fused_gate_out, sigmoid_gate_fused
from .bias_only_attention.triton.main import triton_bias_only_attention
from .layernorm.interface import layernorm_kernel
from .layernorm.triton.main import triton_layernorm
from .transition.triton.fused import triton_transition_fused
from .transition.triton.main import triton_transition
from .triangle_attention.triton.main import triton_triangle_attention_pair_bias
from .tm1.triton.main import triton_tm1
from .tm2.triton.main import triton_tm2


def cuda_transition(*args, **kwargs):
    """Lazy CUDA Transition entry (builds the .so on first call)."""
    from .transition.cuda import cuda_transition as cuda_transition_impl

    return cuda_transition_impl(*args, **kwargs)


def cute_transition_fused(*args, **kwargs):
    """Lazy cute (quack SM90 WGMMA) Transition fwd+bwd entry (imports cutlass on first call)."""
    from .transition.cute.fused import cute_transition_fused as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "cuda_transition",
    "cute_transition_fused",
    "fused_gate_out",
    "sigmoid_gate_fused",
    "layernorm_kernel",
    "triton_adaptive_layer_norm",
    "triton_augmented_attention_pair_bias",
    "triton_bias_only_attention",
    "triton_layernorm",
    "triton_tm1",
    "triton_tm2",
    "triton_transition",
    "triton_transition_fused",
    "triton_triangle_attention_pair_bias",
]

"""Flat kernel re-export surface.

Mirrors team-gm's ``team_gm.modules.kernels`` namespace so the vendored module
layers (``miniworld_kernels.modules.layers.*``) can call ``kernels.triton_*``
without knowing the per-op / per-backend folder layout. Each name resolves to
the canonical (``main`` = psk/benchmark) Triton entry point for that op.
"""

from __future__ import annotations

from .adaln.triton.inference import adaln_inference
from .adaln.triton.main import triton_adaptive_layer_norm
from .adaln.triton.training import adaln_train
from .augmented_attention.triton.main import triton_augmented_attention_pair_bias
from .bias_only_attention.triton.gate_out import fused_gate_out, sigmoid_gate_fused
from .conditioned_transition.triton.interface import (
    cond_transition_inference_dispatch,
)
from .conditioned_transition.triton.training import cond_transition_train
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


def cuda_transition_b2b(*args, **kwargs):
    """Lazy hand-CUDA fused b2b Transition forward (builds the .so on first call).

    Fixed AF3 shapes only (d_hidden=128, n=4 -> K=128, ND=512, D=128). Beats the Triton
    b2b forward ~1.29x at this config. Inference-only (no backward saved)."""
    from .transition.cuda import cuda_transition_b2b as _impl

    return _impl(*args, **kwargs)


def cute_transition_fused(*args, **kwargs):
    """Lazy cute (quack SM90 WGMMA) Transition fwd+bwd entry (imports cutlass on first call)."""
    from .transition.cute.fused import cute_transition_fused as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "adaln_inference",
    "adaln_train",
    "cond_transition_inference_dispatch",
    "cond_transition_train",
    "cuda_transition",
    "cuda_transition_b2b",
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

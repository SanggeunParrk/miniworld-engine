"""Flat re-export surface for the **primitive** fusion kernels.

This is an INTERNAL surface: the per-GEMM / LN / gate / attention fusion units out
of which the public whole-ops (:mod:`miniworld_kernels.ops`) are built. It is also
used by the internal benchmark/reference harness (``miniworld_kernels.modules``).
Model code should consume ``ops.*`` (whole model-layer ops), never reach in here.

Each name resolves lazily to the canonical Triton entry point for that op, without
knowing the per-op / per-backend folder layout. Import stays side-effect-free (no
triton/cutlass loaded until a name is first accessed); the name set is pinned by
``tests/test_public_api.py`` for internal stability.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "adaln_inference": (".adaln.triton.inference", "adaln_inference"),
    "adaln_train": (".adaln.triton.training", "adaln_train"),
    "cond_transition_inference_dispatch": (
        ".conditioned_transition.triton.interface",
        "cond_transition_inference_dispatch",
    ),
    "cond_transition_train": (
        ".conditioned_transition.triton.training",
        "cond_transition_train",
    ),
    "fused_gate_out": (".bias_only_attention.triton.gate_out", "fused_gate_out"),
    "layernorm_kernel": (".layernorm.interface", "layernorm_kernel"),
    "sigmoid_gate_fused": (
        ".bias_only_attention.triton.gate_out",
        "sigmoid_gate_fused",
    ),
    "triton_adaptive_layer_norm": (
        ".adaln.triton.main",
        "triton_adaptive_layer_norm",
    ),
    "triton_augmented_attention_pair_bias": (
        ".augmented_attention",
        "triton_augmented_attention_pair_bias",
    ),
    "triton_bias_only_attention": (
        ".bias_only_attention.triton.main",
        "triton_bias_only_attention",
    ),
    "triton_layernorm": (".layernorm.triton.main", "triton_layernorm"),
    "triton_tm1": (".tm1.triton.main", "triton_tm1"),
    "triton_tm2": (".tm2.triton.main", "triton_tm2"),
    "triton_transition": (".transition.triton.main", "triton_transition"),
    "triton_transition_fused": (
        ".transition.triton.fused",
        "triton_transition_fused",
    ),
    "triton_triangle_attention_pair_bias": (
        ".triangle_attention.triton.main",
        "triton_triangle_attention_pair_bias",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


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

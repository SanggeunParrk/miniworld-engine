"""Model-level ops (parts cut from the full model) that connect fused kernels.

Each op is its own folder (``modules/<op>/``) holding the connecting nn.Module +
reference + its benchmark results. No triton/cute/cuda backends live here — those
belong to the fusion units under ``miniworld_kernels.kernels``.

NOTE: this namespace is an INTERNAL reference / benchmark harness that
composes the kernels. It is NOT the consumed public contract (that is
``miniworld_kernels.kernels``) and may change without a semver bump. Re-exports
resolve lazily so importing an independent submodule does not require every
optional comparison backend.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "AdaptiveLayerNorm": (".adaptive_layernorm", "AdaptiveLayerNorm"),
    "AugmentedAttentionPairBias": (
        ".augmented_attention",
        "AugmentedAttentionPairBias",
    ),
    "BidirectionalTriangleAttention": (
        ".triangle_attention",
        "BidirectionalTriangleAttention",
    ),
    "BidirectionalTriangleMultiplication": (
        ".triangle_multiplication",
        "BidirectionalTriangleMultiplication",
    ),
    "ConditionedTransition": (".conditioned_transition", "ConditionedTransition"),
    "Dropout": (".primitives", "Dropout"),
    "ImplementationType": (".exceptions", "ImplementationType"),
    "InvalidImplementationError": (".exceptions", "InvalidImplementationError"),
    "KernelBackend": (".dispatch", "KernelBackend"),
    "LayerNorm": (".primitives", "LayerNorm"),
    "Linear": (".primitives", "Linear"),
    "Pairformer": (".pairformer", "Pairformer"),
    "PairformerBlock": (".pairformer", "PairformerBlock"),
    "PairformerConfig": (".pairformer", "PairformerConfig"),
    "Transition": (".transition", "Transition"),
    "TriangleAttention": (".triangle_attention", "TriangleAttention"),
    "TriangleMultiplication": (
        ".triangle_multiplication",
        "TriangleMultiplication",
    ),
    "TrianglePairAttention": (".triangle_attention", "TrianglePairAttention"),
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


__all__ = [
    "AdaptiveLayerNorm",
    "AugmentedAttentionPairBias",
    "BidirectionalTriangleAttention",
    "BidirectionalTriangleMultiplication",
    "ConditionedTransition",
    "Dropout",
    "ImplementationType",
    "InvalidImplementationError",
    "KernelBackend",
    "LayerNorm",
    "Linear",
    "Pairformer",
    "PairformerBlock",
    "PairformerConfig",
    "Transition",
    "TriangleAttention",
    "TriangleMultiplication",
    "TrianglePairAttention",
]

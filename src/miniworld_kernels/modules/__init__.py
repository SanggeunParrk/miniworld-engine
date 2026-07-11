"""Model-level ops (parts cut from the full model) that connect fused kernels.

Each op is its own folder (``modules/<op>/``) holding the connecting nn.Module +
reference + its benchmark results. No triton/cute/cuda backends live here — those
belong to the fusion units under ``miniworld_kernels.kernels``.

NOTE: this namespace is an INTERNAL reference / benchmark harness that
composes the kernels. It is NOT the consumed public contract (that is
``miniworld_kernels.kernels``); it imports the full backend + baseline
stack and may change without a semver bump.
"""

from .adaptive_layernorm import AdaptiveLayerNorm
from .augmented_attention import AugmentedAttentionPairBias
from .conditioned_transition import ConditionedTransition
from .dispatch import KernelBackend
from .exceptions import ImplementationType, InvalidImplementationError
from .pairformer import Pairformer, PairformerBlock, PairformerConfig
from .primitives import Dropout, LayerNorm, Linear
from .transition import Transition
from .triangle_attention import (
    BidirectionalTriangleAttention,
    TriangleAttention,
    TrianglePairAttention,
)
from .triangle_multiplication import (
    BidirectionalTriangleMultiplication,
    TriangleMultiplication,
)

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

"""Model-level ops (parts cut from the full model) that connect fused kernels.

Each op is its own folder (``modules/<op>/``) holding the connecting nn.Module +
reference + its benchmark results. No triton/cute/cuda backends live here — those
belong to the fusion units under ``miniworld_kernels.kernels``.
"""

from .adaptive_layernorm import AdaptiveLayerNorm
from .augmented_attention import AugmentedAttentionPairBias
from .conditioned_transition import ConditionedTransition
from .exceptions import ImplementationType, InvalidImplementationError
from .primitives import Dropout, LayerNorm, Linear
from .transition import Transition
from .triangle_attention import TriangleAttention, TrianglePairAttention
from .triangle_multiplication import TriangleMultiplication

__all__ = [
    "AdaptiveLayerNorm",
    "AugmentedAttentionPairBias",
    "ConditionedTransition",
    "Dropout",
    "ImplementationType",
    "InvalidImplementationError",
    "LayerNorm",
    "Linear",
    "Transition",
    "TriangleAttention",
    "TriangleMultiplication",
    "TrianglePairAttention",
]

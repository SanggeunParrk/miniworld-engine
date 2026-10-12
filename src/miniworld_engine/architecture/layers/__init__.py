"""Historical checkpoint adapters retained for existing ESMFold2 consumers.

Canonical AF-family blocks import miniworld_engine.modules directly. These
older import paths preserve the checkpoint layouts and delta/residual contracts
of the pre-FoldForge workflows; do not substitute them into canonical blocks.
"""

from miniworld_engine.architecture.layers.adaln import AdaptiveLayerNorm
from miniworld_engine.architecture.layers.attention_pair_bias import (
    AttentionPairBias,
)
from miniworld_engine.architecture.layers.augmented_attention import (
    AugmentedAttentionPairBias,
)
from miniworld_engine.architecture.layers.conditioned_transition import (
    ConditionedTransition,
)
from miniworld_engine.architecture.layers.embeddings import (
    RelativePositionEmbedding,
    fourier_embedding,
)
from miniworld_engine.architecture.layers.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)
from miniworld_engine.architecture.layers.outer_product import (
    OuterProduct,
    OuterProductMean,
)
from miniworld_engine.architecture.layers.swa_atom_attention import (
    SWA3DRoPEAttention,
    build_3d_rope,
)
from miniworld_engine.architecture.layers.transition import Transition
from miniworld_engine.architecture.layers.triangle_updates import (
    BidirectionalTriangleMultiplication,
    TriangleAttention,
    TriangleMultiplication,
)

__all__ = [
    "AdaptiveLayerNorm",
    "AttentionPairBias",
    "AugmentedAttentionPairBias",
    "BidirectionalTriangleMultiplication",
    "ConditionedTransition",
    "MSAPairWeightedAveraging",
    "OuterProduct",
    "OuterProductMean",
    "RelativePositionEmbedding",
    "SWA3DRoPEAttention",
    "Transition",
    "TriangleAttention",
    "TriangleMultiplication",
    "build_3d_rope",
    "fourier_embedding",
]

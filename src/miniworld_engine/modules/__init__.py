"""Model-level ops. Every module is a :class:`~miniworld_engine.modules.primitives.BackendModule`: its ``"pytorch"`` backend is the
reference (the single source of truth) every kernel backend is checked against.

One file per op family; a folder only where a family needs several files:

    primitives.py  the backend mechanism and the layers the ops compose (Linear, Dropout, MPLinear, MasterWeights, master_linear, ...)
    functional.py  pure functions of the model math (gates, rotary, the sampler's Kabsch / churn / Euler glue, smooth LDDT)
    norm.py        LayerNorm, RMSNorm, AdaptiveLayerNorm, IdentityAdaptiveLayerNorm, AdaptiveRMSNorm
    transition.py  Transition, ConditionedTransition, UpGatedConditionedTransition
    msa.py         OuterProduct, OuterProductMean, MSAPairWeightedAveraging
    triangle.py    TriangleAttention (+ AF3 / RF3 / Chai1 / Chai1Confidence), TriangleMultiplication (+ RF3), the bidirectional blocks
    pairformer.py  PairformerBlock, Pairformer, PairformerConfig
    mpnn/          ProteinMPNN (features, layers, model, loss)
    attention.py   AttentionPairBias, AugmentedAttentionPairBias (+ FlatQKNorm / Ungated variants), BiasOnlyAttention
    token_dit.py   TokenDiT (selector) + one block class per model: TokenDiTProtenix (also AF3) / Boltz2 / RoseTTAFold3 / Chai1 / MiniWorld /
                   MiniWorldBiasOnly
    atom_dit.py    AtomDiT (selector) + one block class per model: AtomDiTOpenFold3 (also AF3) / Boltz2 / RoseTTAFold3 / Chai1 / Protenix /
                   ESMFold2 / MiniWorld, SWA3DRoPEAttention, SWAAtomTransformer
    embedder.py    InputFeatureEmbedder
    smooth_lddt.py SmoothLddtLoss (the diffusion loss, AF3 Algorithm 27)
    sampler_ops.py AugmentNoise, AlignEuler, PoolAtoms (the diffusion sampler's glue and the atom encoder's pooling)
    pair_embed.py  PairEmbedAdaLN, PairEmbedGeom, PairEmbedCA (La-Proteina's pair embeddings: the denoiser's pair builder, the autoencoder's encoder / decoder pair)
    token_pair_init.py  TokenPairInit (the input feature embedder's token-pair initialisation: left + right + relative-position and bond Linears)
"""

from miniworld_engine.modules.atom_dit import (
    AtomDiT,
    AtomDiTBoltz2,
    AtomDiTChai1,
    AtomDiTESMFold2,
    AtomDiTMiniWorld,
    AtomDiTOpenFold3,
    AtomDiTProtenix,
    AtomDiTRoseTTAFold3,
    SWA3DRoPEAttention,
    SWAAtomTransformer,
)
from miniworld_engine.modules.attention import (
    AttentionPairBias,
    AugmentedAttentionPairBias,
    BiasOnlyAttention,
    FlatQKNormAugmentedAttention,
    UngatedAugmentedAttention,
)
from miniworld_engine.modules.embedder import InputFeatureEmbedder
from miniworld_engine.modules.msa import MSAPairWeightedAveraging, OuterProduct, OuterProductMean
from miniworld_engine.modules.norm import AdaptiveLayerNorm, AdaptiveRMSNorm, IdentityAdaptiveLayerNorm, LayerNorm, RMSNorm
from miniworld_engine.modules.pair_embed import PairEmbedAdaLN, PairEmbedCA, PairEmbedGeom
from miniworld_engine.modules.pairformer import Pairformer, PairformerBlock, PairformerConfig
from miniworld_engine.modules.primitives import BackendModule, Dropout, Linear, MasterWeights, MPLinear
from miniworld_engine.modules.token_dit import (
    TokenDiT,
    TokenDiTBoltz2,
    TokenDiTChai1,
    TokenDiTMiniWorld,
    TokenDiTMiniWorldBiasOnly,
    TokenDiTProtenix,
    TokenDiTRoseTTAFold3,
)
from miniworld_engine.modules.sampler_ops import AlignEuler, AugmentNoise, PoolAtoms
from miniworld_engine.modules.smooth_lddt import SmoothLddtLoss
from miniworld_engine.modules.token_pair_init import TokenPairInit
from miniworld_engine.modules.transition import ConditionedTransition, Transition, UpGatedConditionedTransition
from miniworld_engine.modules.triangle import (
    BidirectionalTriangleAttention,
    BidirectionalTriangleMultiplication,
    TriangleAttention,
    TriangleAttentionAF3,
    TriangleAttentionChai1,
    TriangleAttentionChai1Confidence,
    TriangleAttentionRF3,
    TriangleMultiplication,
    TriangleMultiplicationRF3,
)

__all__ = [
    "AlignEuler",
    "AtomDiT",
    "AtomDiTBoltz2",
    "AtomDiTChai1",
    "AtomDiTESMFold2",
    "AtomDiTMiniWorld",
    "AtomDiTOpenFold3",
    "AtomDiTProtenix",
    "AtomDiTRoseTTAFold3",
    "AugmentNoise",
    "AttentionPairBias",
    "AugmentedAttentionPairBias",
    "BiasOnlyAttention",
    "FlatQKNormAugmentedAttention",
    "UngatedAugmentedAttention",
    "InputFeatureEmbedder",
    "SWA3DRoPEAttention",
    "SWAAtomTransformer",
    "TokenDiT",
    "TokenDiTBoltz2",
    "TokenDiTChai1",
    "TokenDiTMiniWorld",
    "TokenDiTMiniWorldBiasOnly",
    "TokenDiTProtenix",
    "TokenDiTRoseTTAFold3",
    "AdaptiveLayerNorm",
    "AdaptiveRMSNorm",
    "IdentityAdaptiveLayerNorm",
    "BackendModule",
    "BidirectionalTriangleAttention",
    "BidirectionalTriangleMultiplication",
    "ConditionedTransition",
    "Dropout",
    "LayerNorm",
    "Linear",
    "MPLinear",
    "MSAPairWeightedAveraging",
    "MasterWeights",
    "OuterProduct",
    "OuterProductMean",
    "PairEmbedAdaLN",
    "PairEmbedCA",
    "PairEmbedGeom",
    "Pairformer",
    "PairformerBlock",
    "PairformerConfig",
    "PoolAtoms",
    "RMSNorm",
    "SmoothLddtLoss",
    "TokenPairInit",
    "Transition",
    "UpGatedConditionedTransition",
    "TriangleAttention",
    "TriangleAttentionAF3",
    "TriangleAttentionChai1",
    "TriangleAttentionChai1Confidence",
    "TriangleAttentionRF3",
    "TriangleMultiplication",
    "TriangleMultiplicationRF3",
]

"""Model-level ops (parts cut from the full model) that connect fused kernels.

Every op is its own folder (``modules/<op>/``) holding the connecting nn.Module + reference + its
benchmark results, and every folder holds a ``module.py``. Three ops used to be flat files instead
(``attention_pair_bias.py``, ``msa_pair_weighted_averaging.py``, ``swa_atom_attention.py``), which
made the rule above a description of eight of eleven ops.

The flat modules that remain are NOT ops -- they are the shared infrastructure the ops are built
from, and they are exactly these four:

    dispatch.py    backend selection (public ImplementationType -> internal KernelBackend)
    exceptions.py  the public implementation enum and its error
    primitives.py  layer classes the ops compose (Linear, LayerNorm, Dropout, MPLinear)
    functional.py  free functions the ops compose (sigmoid_gate, swish_gate)

``functional.py`` is named for torch's own split between layer classes and the free functions
beside them. It was ``ops.py``, one level below :mod:`miniworld_engine.ops` -- which is the public
WHOLE-OP contract, the opposite kind of thing.

``tests/layout/test_module_layout.py`` holds both halves of this rule.

NOTE: this namespace is an INTERNAL reference / benchmark harness that
composes the kernels. It is NOT the consumed public contract (that is
``miniworld_engine.kernels``); it imports the full backend + baseline
stack and may change without a semver bump.
"""

from miniworld_engine.modules.adaptive_layernorm import AdaptiveLayerNorm
from miniworld_engine.modules.attention_pair_bias import AttentionPairBias
from miniworld_engine.modules.augmented_attention import AugmentedAttentionPairBias
from miniworld_engine.modules.conditioned_transition import ConditionedTransition
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)
from miniworld_engine.modules.outer_product import OuterProduct, OuterProductMean
from miniworld_engine.modules.pairformer import (
    Pairformer,
    PairformerBlock,
    PairformerConfig,
)
from miniworld_engine.modules.primitives import Dropout, LayerNorm, Linear, MPLinear
from miniworld_engine.modules.swa_atom_attention import SWA3DRoPEAttention
from miniworld_engine.modules.transition import Transition
from miniworld_engine.modules.triangle_attention import (
    BidirectionalTriangleAttention,
    TriangleAttention,
    TrianglePairAttention,
)
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
    TriangleMultiplication,
)

__all__ = [
    "AdaptiveLayerNorm",
    "AttentionPairBias",
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
    "MPLinear",
    "MSAPairWeightedAveraging",
    "OuterProduct",
    "OuterProductMean",
    "Pairformer",
    "PairformerBlock",
    "PairformerConfig",
    "SWA3DRoPEAttention",
    "Transition",
    "TriangleAttention",
    "TriangleMultiplication",
    "TrianglePairAttention",
]

"""PyTorch module blocks for Team GM.

team-gm holds the REPRESENTATIVE AF3 blocks (Pairformer, MSA module, template,
diffusion transformer, SWA atom transformer). They compose the building-block ops
from ``miniworld_engine.modules``. Model-specific / reduced variants (e.g. the pair-
only MiniPairformer, MiniMSAModule) belong to the terminal product repos, not here.
"""

from miniworld_engine.architecture.blocks.diffusion_transformer import (
    DiffusionTransformer,
    DiffusionTransformerBlock,
)
from miniworld_engine.architecture.blocks.msa_module import (
    MSAModule,
    MSAModuleBlock,
    MSAPairformer,
    MSAPairformerBlock,
)
from miniworld_engine.architecture.blocks.pairformer import (
    Pairformer,
    PairformerBlock,
)
from miniworld_engine.architecture.blocks.swa_atom_transformer import (
    SWAAtomBlock,
    SWAAtomTransformer,
)
from miniworld_engine.architecture.blocks.template import (
    TemplateEmbedder,
    TemplatePairformer,
    TemplatePairformerBlock,
)

__all__ = [
    "DiffusionTransformer",
    "DiffusionTransformerBlock",
    "MSAModule",
    "MSAModuleBlock",
    "MSAPairformer",
    "MSAPairformerBlock",
    "Pairformer",
    "PairformerBlock",
    "SWAAtomBlock",
    "SWAAtomTransformer",
    "TemplateEmbedder",
    "TemplatePairformer",
    "TemplatePairformerBlock",
]

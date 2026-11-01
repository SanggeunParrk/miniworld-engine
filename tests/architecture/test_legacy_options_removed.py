"""The legacy magnitude-preserving / rotation options of the architecture layers and blocks were removed: they are still accepted as keyword
arguments (so configs that spell them out as false keep loading) and must be false; switched on they raise instead of building a different model."""

import pytest

from miniworld_engine.architecture.blocks.rope_swa_af3_transformer import RoPESWAAF3Block
from miniworld_engine.architecture.blocks.swa_atom_transformer import SWAAtomBlock, SwiGLUFFN
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.architecture.layers.adaln import AdaptiveLayerNorm
from miniworld_engine.architecture.layers.augmented_attention import AugmentedAttentionPairBias
from miniworld_engine.architecture.layers.conditioned_transition import ConditionedTransition

PYTORCH = ImplementationType.PYTORCH

BUILD = {
    "AdaptiveLayerNorm": lambda **kw: AdaptiveLayerNorm(16, 8, **kw),
    "ConditionedTransition": lambda **kw: ConditionedTransition(16, 8, implementation=PYTORCH, **kw),
    "AugmentedAttentionPairBias": lambda **kw: AugmentedAttentionPairBias(16, 8, 4, 2, implementation=PYTORCH, **kw),
    "RoPESWAAF3Block": lambda **kw: RoPESWAAF3Block(16, 8, 2, implementation=PYTORCH, **kw),
    "SWAAtomBlock": lambda **kw: SWAAtomBlock(16, 8, 2, implementation=PYTORCH, **kw),
    "SwiGLUFFN": lambda **kw: SwiGLUFFN(48, implementation=PYTORCH, **kw),
}
OPTIONS = {
    "AdaptiveLayerNorm": ("use_rotation", "mp_full"),
    "ConditionedTransition": ("magnitude_preserving", "use_rotation", "mp_full"),
    "AugmentedAttentionPairBias": ("magnitude_preserving", "use_rotation", "mp_full"),
    "RoPESWAAF3Block": ("magnitude_preserving", "use_rotation", "mp_full", "mp_residual"),
    "SWAAtomBlock": ("magnitude_preserving", "mp_full", "mp_residual"),
    "SwiGLUFFN": ("magnitude_preserving",),
}


@pytest.mark.parametrize("name", sorted(BUILD))
def test_the_options_spelled_false_still_build(name):
    BUILD[name](**dict.fromkeys(OPTIONS[name], False))


@pytest.mark.parametrize(("name", "option"), [(n, o) for n in sorted(BUILD) for o in OPTIONS[n]])
def test_a_removed_option_raises_instead_of_building_another_model(name, option):
    with pytest.raises(NotImplementedError, match=option):
        BUILD[name](**{option: True})

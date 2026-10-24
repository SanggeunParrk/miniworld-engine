"""The architecture-level ``ImplementationType`` names the Anthropic release too, and maps to the engine's option."""
from miniworld_engine.architecture._engine_impl import to_engine_impl
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.modules.exceptions import ImplementationType as EngineImplementation


def test_the_anthropic_member_maps_to_the_engines():
    assert ImplementationType("anthropic") is ImplementationType.ANTHROPIC
    assert to_engine_impl(ImplementationType.ANTHROPIC) is EngineImplementation.ANTHROPIC
    assert to_engine_impl("anthropic") is EngineImplementation.ANTHROPIC


def test_the_existing_values_are_unchanged():
    assert to_engine_impl(ImplementationType.MINIWORLD_ENGINE) is EngineImplementation.MINIWORLD
    assert to_engine_impl(ImplementationType.PYTORCH) is EngineImplementation.PYTORCH
    assert to_engine_impl(ImplementationType.CUEQUIVARIANCE) is EngineImplementation.CUEQUIVARIANCE
    assert ImplementationType("miniworld_kernels") is ImplementationType.MINIWORLD_ENGINE
    assert ImplementationType.MINIWORLD_KERNELS is ImplementationType.MINIWORLD_ENGINE

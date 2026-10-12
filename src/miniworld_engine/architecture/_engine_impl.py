"""Translate the historical team-gm backend names at the engine boundary."""

import torch

from miniworld_engine import settings
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.modules.exceptions import (
    ImplementationType as EngineImplementation,
)

_B200_CAPABILITY = (10, 0)


def torch_pointwise() -> bool:
    """Whether pointwise and normalization modules run as PyTorch ops here.

    B200 policy: the pieces around the GEMMs and attention are PyTorch ops that
    ``torch.compile`` fuses (inductor), not the engine's hand-written Triton
    kernels. ``miniworld_engine.settings.b200_engine_triton`` keeps the engine
    Triton kernels (the A/B switch). An engine that predates the switch keeps
    its Triton kernels, so this is a no-op until the engine pin has it.

    Call it when a module is BUILT: it asks the device, which a traced forward
    must not.
    """
    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == _B200_CAPABILITY
    ):
        return False
    return not getattr(settings.current(), "b200_engine_triton", True)


def to_engine_impl_pointwise(
    implementation: ImplementationType,
) -> EngineImplementation:
    """Like :func:`to_engine_impl`, but PyTorch under :func:`torch_pointwise`."""
    if torch_pointwise():
        return EngineImplementation.PYTORCH
    return to_engine_impl(implementation)


def to_engine_impl(implementation: ImplementationType) -> EngineImplementation:
    """Preserve team-gm configuration values while selecting engine backends."""
    implementation = ImplementationType(implementation)
    return EngineImplementation(
        "miniworld"
        if implementation == ImplementationType.MINIWORLD_ENGINE
        else implementation.value,
    )

# vendored from team-gm psk/benchmark : src/team_gm/modules/exceptions.py
from enum import Enum


class ImplementationType(str, Enum):
    """Implementation types for certain modules.

    ``miniworld`` is the repo's own kernel family as a single *source* backend
    (parallel to ``cuequivariance`` / ``te`` naming a vendor, not a technology):
    selecting it auto-routes to the best internal impl (triton-persistent / cute /
    …) per shape. ``triton`` / ``cute`` remain selectable for explicit per-impl
    benching. Plots display ``miniworld`` as "ours" (see ``viz.style``)."""

    PYTORCH = "pytorch"
    TRITON = "triton"
    CUDA = "cuda"
    CUTE = "cute"
    CUEQUIVARIANCE = "cuequivariance"
    MINIWORLD = "miniworld"


class InvalidImplementationError(ValueError):
    """Raised when an invalid implementation type is specified."""

    def __init__(self, implementation: str) -> None:
        msg = f"Invalid implementation: '{implementation}'."
        super().__init__(msg)

# vendored from team-gm psk/benchmark : src/team_gm/modules/exceptions.py
from enum import Enum


class ImplementationType(str, Enum):
    """Implementation types for certain modules."""

    PYTORCH = "pytorch"
    TRITON = "triton"
    CUDA = "cuda"
    CUTE = "cute"
    CUEQUIVARIANCE = "cuequivariance"


class InvalidImplementationError(ValueError):
    """Raised when an invalid implementation type is specified."""

    def __init__(self, implementation: str) -> None:
        msg = f"Invalid implementation: '{implementation}'."
        super().__init__(msg)

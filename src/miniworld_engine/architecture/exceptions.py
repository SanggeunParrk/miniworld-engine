from enum import Enum


class ImplementationType(str, Enum):
    """Implementation types for certain modules."""

    PYTORCH = "pytorch"
    MINIWORLD_ENGINE = "miniworld_engine"
    CUEQUIVARIANCE = "cuequivariance"
    # Source compatibility for old consumers; iteration uses the canonical member.
    MINIWORLD_KERNELS = MINIWORLD_ENGINE

    @classmethod
    def _missing_(cls, value):
        if value == "miniworld_kernels":
            return cls.MINIWORLD_ENGINE
        return None


class InvalidImplementationError(ValueError):
    """Raised when an invalid implementation type is specified."""

    def __init__(self, implementation: str) -> None:
        valid = ", ".join(f"'{impl.value}'" for impl in ImplementationType)
        msg = f"Invalid implementation: '{implementation}'. Valid options: {valid}."
        super().__init__(msg)

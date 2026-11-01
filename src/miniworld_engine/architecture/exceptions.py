from enum import Enum


class ImplementationType(str, Enum):
    """Implementation types for certain modules."""

    PYTORCH = "pytorch"
    MINIWORLD_ENGINE = "miniworld_engine"
    CUEQUIVARIANCE = "cuequivariance"
    ANTHROPIC = "anthropic"
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


def legacy_options_removed(where: str, **options: bool) -> None:
    """Raise when a removed legacy option (magnitude-preserving ``magnitude_preserving`` / ``mp_full`` / ``mp_residual`` or rotation
    modulation ``use_rotation``) is switched on. The options are accepted, and must be false, so configs that spell them out as false keep
    loading; the code behind them is in git history (see the commit that added this function)."""
    on = sorted(name for name, value in options.items() if value)
    if on:
        msg = f"{where}: the legacy option(s) {', '.join(on)} were removed (no engine kernel or current model uses them); restore them from git history if needed."
        raise NotImplementedError(msg)

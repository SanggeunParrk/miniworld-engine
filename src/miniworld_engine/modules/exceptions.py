# vendored from team-gm psk/benchmark : src/team_gm/modules/exceptions.py
from enum import Enum


class ImplementationType(str, Enum):
    """Public backend option for the model-level modules.

    Two of these are the intended user-facing choices:

      * ``pytorch``   — the readable reference (and the autograd oracle).
      * ``miniworld`` — "ours (auto)": the repo's own kernel family as a single
        *source* backend (parallel to ``cuequivariance`` naming a vendor, not a
        technology). Selecting it auto-routes to the best correct internal kernel
        for the running GPU / shape / mode via ``modules.dispatch`` — this is what
        production should select. Plots display it as "ours" (see ``viz.style``).
      * ``cuequivariance`` — the vendor baseline (comparison only).
      * ``anthropic`` — kernels from Anthropic's uplifting-biomolecular-modeling release:
        the TriMul units of a payload named by ``TRIMUL_NATIVE_BUILD_DIR`` (see
        ``integrations.anthropic_trimul``), and the MSA-module paths built on the
        ``opt_core`` runtime named by ``OPT_CORE_DIR`` (see ``integrations.anthropic_msa``
        — its OuterProductMean epilogue is this repo's own CUDA kernel). Inference only,
        and it refuses rather than silently reroute; ``miniworld`` uses the same payloads
        where it can and falls back where it cannot.

    Triton is NOT a public backend. Behind ``miniworld`` the engine uses its own CUDA path where one
    serves the call; on a card without one it routes to the Triton kernel family only if the engine
    holds a record for that exact card, else it warns and runs the PyTorch reference
    (:mod:`miniworld_engine.autotune.records`, ``docs/dispatch.md``). The strings ``"triton"`` and ``"cuda"``
    are still read, with a ``DeprecationWarning``, as ``miniworld`` (removed in the next minor release);
    forcing the Triton family for A/B work is ``settings.force_triton_kernels``."""

    PYTORCH = "pytorch"
    CUEQUIVARIANCE = "cuequivariance"
    MINIWORLD = "miniworld"
    ANTHROPIC = "anthropic"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and value.strip().lower() in ("triton", "cuda"):
            import warnings

            old = value.strip().lower()
            warnings.warn(
                f"implementation={old!r} is deprecated and is read as 'miniworld' (it is removed in the next minor release). "
                "Triton is not a public backend: on H100 / B200 'miniworld' is the CUDA path (no Triton there), and elsewhere it "
                "routes to Triton only on a card with a record. Forcing the Triton kernel family is only possible through the A/B "
                "switch settings.force_triton_kernels=True.",
                DeprecationWarning,
                stacklevel=3,
            )
            return cls.MINIWORLD
        return None


class InvalidImplementationError(ValueError):
    """Raised when an invalid implementation type is specified."""

    def __init__(self, implementation: str) -> None:
        msg = f"Invalid implementation: '{implementation}'."
        super().__init__(msg)

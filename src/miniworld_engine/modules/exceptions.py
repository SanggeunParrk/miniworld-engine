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
      * ``anthropic`` — the TriMul kernels of Anthropic's uplifting-biomolecular-modeling
        release, run from a payload named by ``TRIMUL_NATIVE_BUILD_DIR`` (see
        ``integrations.anthropic_trimul``). Inference only, and it refuses rather than
        silently reroute; ``miniworld`` uses the same payload where it can and falls back
        where it cannot.

    ``triton`` / ``cute`` / ``cuda`` are the concrete *technology* backends that
    ``miniworld`` resolves to. They are INTERNAL — the module layer resolves to a
    :class:`~miniworld_engine.modules.dispatch.KernelBackend` and never branches on
    ``ImplementationType`` in ``forward``. They remain accepted here only so the
    benchmark harness can pin one explicitly for per-impl A/B measurement; new
    application code should pass ``miniworld`` (or ``pytorch``) instead."""

    PYTORCH = "pytorch"
    TRITON = "triton"
    CUDA = "cuda"
    CUTE = "cute"
    CUEQUIVARIANCE = "cuequivariance"
    MINIWORLD = "miniworld"
    ANTHROPIC = "anthropic"


class InvalidImplementationError(ValueError):
    """Raised when an invalid implementation type is specified."""

    def __init__(self, implementation: str) -> None:
        msg = f"Invalid implementation: '{implementation}'."
        super().__init__(msg)

"""Explicit run configuration, replacing the engine's environment-variable switches.

Environment variables made the engine's behaviour invisible: which kernel a run actually used, and
whether the full autotune grid or a cached top-K was benched, depended on shell state that no
caller declared and no log recorded. That is how a build spent hours benching the PyTorch reference
while reporting it as ours, and how a capture silently skipped every kernel on the losing side of a
dispatch decision.

So the knobs live here instead, as one explicit object:

    from miniworld_engine import settings
    settings.configure(run_autotune=True, capture=True, autotune_kernels={"transition"})

``configure`` replaces the active settings wholesale and returns the previous value, so a caller can
restore it; ``current()`` reads them. Defaults match the engine's shipped behaviour, so an untouched
import behaves exactly as before.

Some kernels read these at IMPORT time (a module-level ``configs`` list is chosen once), so a
process that needs non-default settings must call ``configure`` BEFORE importing kernel modules —
which is what the CLI does.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

#: Kernels whose triton autotuner unlocks its full config grid. Formerly TRITON_AUTOTUNE, which
#: named ONE kernel group per run ("transition", "tri_attention", "tri_multi", "adaln", ...); a set
#: expresses the same thing without forcing a run per group.
AutotuneKernel = Literal[
    "transition", "tri_attention", "tri_multi", "adaln", "layernorm", "layer_norm_linear",
    "augmented_attention",
]

#: How a device-calibrated dispatch switch is resolved. "auto" benchmarks once per GPU and caches
#: the winner; "off" uses the static threshold; "force" re-runs the calibration ignoring the cache.
DispatchMode = Literal["auto", "off", "force"]


@dataclasses.dataclass(frozen=True)
class Settings:
    """One run's configuration. Frozen: change it through :func:`configure`, not by mutation."""

    #: Bench the full config grid instead of the cached top-K. Formerly MINIWORLD_RUN_AUTOTUNE.
    run_autotune: bool = False
    #: Record every benched (config -> ms) for an autotune-cache build. Formerly
    #: MINIWORLD_AUTOTUNE_CAPTURE.
    capture: bool = False
    #: Kernels with their full grid unlocked. Formerly TRITON_AUTOTUNE.
    autotune_kernels: frozenset[str] = frozenset()
    #: Worker processes used to PRE-compile an autotune round before it is timed. None = one per
    #: usable core (capped). 1 disables it. Only a build ever sets this; see autotune.capture.
    compile_jobs: int | None = None
    #: bias_only gate epilogue calibration. Formerly MINIWORLD_BIASONLY_AUTOTUNE.
    biasonly_dispatch: DispatchMode = "auto"
    #: layernorm backend calibration. Formerly MINIWORLD_LN_AUTOTUNE.
    layernorm_dispatch: DispatchMode = "auto"

    # ---- capture-time dispatch pins -------------------------------------------------------- #
    # A build must capture BOTH sides of every device-calibrated switch. The card picks one side
    # for the shapes being swept, so the other side's kernels never fire and never get captured —
    # yet they still run in production at other shapes, then with no cached configs at all. Pinning
    # lets a build sweep each side explicitly. None = let the engine decide, as at run time.
    #: bias_only gate epilogue: "fused" (bias_only_gate_out_fwd) or "split" (bias_only_sigmul_*).
    pin_gate_backend: Literal["fused", "split"] | None = None
    #: Inference LN+proj concat fusion (layernorm_linear).
    pin_infer_concat: bool | None = None
    #: layernorm partial-reduction variant.
    pin_ln_partial_reduction: bool | None = None

    # ---- transition backend selection ------------------------------------------------------ #
    #: Force the split path (ln_in + non-fused triton_transition) over the fused kernels. An
    #: escape hatch and A/B lever; the fused large-d path is the default. Formerly
    #: MINIWORLD_TRANSITION_FORCE_SPLIT.
    transition_force_split: bool = False
    #: Route d=128/n=4 inference through the hand-CUDA fused b2b kernel (~1.29x the Triton b2b).
    #: Formerly MINIWORLD_TRANSITION_CUDA_B2B.
    transition_cuda_b2b: bool = True
    #: Large-d training backend: None = torch fallback, "triton" = cute+triton hybrid, "cute" =
    #: all-cute. Formerly MINIWORLD_TRANSITION_LARGE_D_TRAINING.
    transition_large_d_training: Literal["triton", "cute"] | None = None
    #: Backward backend when the cute path is engaged. Formerly MINIWORLD_TRANSITION_CUTE_BACKWARD.
    transition_cute_backward: Literal["triton", "cute"] = "triton"

    #: Route the sm90 large-d (K in {256,512}) gate-backward through the hand-CUDA WGMMA kernel
    #: (beats the Triton recompute). Formerly MINIWORLD_TRANSITION_GATEBWD_WGMMA.
    transition_gatebwd_wgmma: bool = True
    #: Fuse the layernorm stats pass into the transition kernel. Formerly
    #: MINIWORLD_TRANSITION_FUSE_STATS.
    transition_fuse_stats: bool = False
    #: Use the hand-CUDA layernorm backward inside transition. Formerly
    #: MINIWORLD_TRANSITION_LNBWD_CUDA.
    transition_lnbwd_cuda: bool = True
    #: Version-B backward that saves xn instead of recomputing it. Formerly
    #: TRANSITION_SAVEDXN_SPLIT_BWD.
    transition_savedxn_split_bwd: bool = False
    #: Fold dA/dB into the layernorm backward. Formerly TRANSITION_DAB_LNBWD.
    transition_dab_lnbwd: bool = False
    #: Privatised dgamma/dbeta accumulators in the transition LN backward (1.31x at L=1024; the
    #: alternative is a single-accumulator atomic path). Formerly
    #: MINIWORLD_TRANSITION_LNBWD_PRIVATIZE.
    transition_lnbwd_privatize: bool = True

    # ---- layernorm backend selection ------------------------------------------------------- #
    #: Force one layernorm backward path, bypassing cache + heuristic. Formerly MINIWORLD_LN_BWD.
    layernorm_bwd_path: Literal["persistent", "partial", "atomic", "cuda"] | None = None
    #: Use the hand-CUDA layernorm backward. Formerly MINIWORLD_LN_IN_CUDA.
    layernorm_cuda_bwd: bool = False
    #: Force one layernorm_linear out-backward path. Formerly MINIWORLD_LNOUT_BWD.
    layernorm_out_bwd_path: str | None = None

    # ---- triangle-multiplication backend selection ----------------------------------------- #
    #: Override the trimul implementation choice. Formerly MINIWORLD_TRIMUL_IMPL.
    trimul_impl: str | None = None
    #: Override the trimul output layout. Formerly MINIWORLD_TRIMUL_OUT_LAYOUT.
    trimul_out_layout: str | None = None
    #: Engage the cute in-projection dispatch at all. Formerly TRIMUL_DISPATCH.
    trimul_cute_dispatch: bool = True
    #: Log each cute in-projection dispatch decision. Formerly TRIMUL_DISPATCH_LOG.
    trimul_dispatch_log: bool = False
    #: Fused training front for the sm100 in-projection. Formerly MINIWORLD_TRAIN_FRONT_FUSED.
    trimul_train_front_fused: bool = True

    # ---- sm100 (B200) bring-up knobs -------------------------------------------------------- #
    # Debug/bring-up levers for the cute sm100 kernels, moved off MW_* / LNL_* environment
    # variables. These were migrated mechanically: sm_100 has no execution path on the Ampere
    # cards available here, so only import and lint are verified — the kernels' use of these
    # values is not.
    #: Sigmoid formulation in the sm100 epilogues. Formerly MW_SIG.
    sm100_sig_mode: str = "rsqrt"
    #: Epilogue depth. None keeps each kernel's own default (b2b forward 3, gate-backward 1), which
    #: differed per site under the single MW_EPI_DEPTH variable.
    sm100_epi_depth: int | None = None
    #: Split (non-fused) gate backward. Formerly MW_SPLIT.
    sm100_split: bool = False
    #: Skip the gradient computation in the gate backward. Formerly MW_NOGRAD.
    sm100_no_grad: bool = False
    #: Single-output variant of the gate backward. Formerly MW_ONEOUT.
    sm100_one_out: bool = False
    #: Projection only, no gate. Formerly MW_PROJONLY.
    sm100_proj_only: bool = False
    #: Gate in the epilogue. Formerly MW_NOGATE_EPI (inverted).
    sm100_gate_epi: bool = True
    #: Drop the exp in the gate. Formerly MW_NOEXP.
    sm100_no_exp: bool = False
    #: Print kernel setup during bring-up. Formerly MW_SETUP_DBG.
    sm100_setup_debug: bool = False
    #: layernorm_linear cute debug level. Formerly LNL_DEBUG.
    lnl_debug: int = 0
    #: Warp-specialised stats bring-up debug level. Formerly LNL_WS_DEBUG.
    lnl_ws_debug: int = 0
    #: Warp-specialised stats production path. Formerly LNL_WS.
    lnl_ws: int = 0

    # ---- diagnostics ----------------------------------------------------------------------- #
    #: jaxtyped+beartype decoration on annotated functions. Off by default: it is a per-call cost.
    #: Formerly SHOULD_TYPECHECK.
    typecheck: bool = False
    #: Assert that SWA's atom blocks really are front-packed, at runtime. Formerly
    #: SWA_CHECK_FRONT_PACKED.
    swa_check_front_packed: bool = False

    def autotunes(self, kernel: str) -> bool:
        """Is ``kernel``'s full config grid unlocked for this run?"""
        return kernel in self.autotune_kernels


_ACTIVE = Settings()


def current() -> Settings:
    """The active settings."""
    return _ACTIVE


def configure(**kwargs) -> Settings:  # noqa: ANN003
    """Replace the active settings; returns the previous value so a caller can restore it."""
    global _ACTIVE
    previous = _ACTIVE
    fields = {f.name for f in dataclasses.fields(Settings)}
    unknown = set(kwargs) - fields
    if unknown:
        raise TypeError(f"unknown setting(s): {', '.join(sorted(unknown))}")
    if "autotune_kernels" in kwargs and kwargs["autotune_kernels"] is not None:
        kwargs["autotune_kernels"] = frozenset(kwargs["autotune_kernels"])
    _ACTIVE = dataclasses.replace(previous, **kwargs)
    return previous


def reset() -> None:
    """Restore shipped defaults."""
    global _ACTIVE
    _ACTIVE = Settings()

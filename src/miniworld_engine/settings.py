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
import os
from typing import Literal, get_args

#: Kernels whose triton autotuner unlocks its full config grid. Formerly TRITON_AUTOTUNE, which
#: named ONE kernel group per run; a set expresses the same thing without forcing a run per group.
#:
#: Exactly the groups something asks about. It listed seven, and four of them --  "tri_multi",
#: "layernorm", "layer_norm_linear", "augmented_attention" -- had no call site at all, so naming
#: one unlocked nothing and said nothing. A vocabulary entry is a promise that a name does
#: something.
#:
#: "tri_attention" covers two call sites: triangle_attention/triton/atomic.py and
#: gated_projection/triton/main.py. The gate projection is the projection triangle attention
#: gates with, and it inherited the key when it was vendored -- stated here because the coupling
#: is invisible from either file.
AutotuneKernel = Literal["transition", "tri_attention", "adaln"]

#: How a device-calibrated dispatch switch is resolved. "auto" benchmarks once per GPU and caches
#: the winner; "off" uses the static threshold; "force" re-runs the calibration ignoring the cache.
DispatchMode = Literal["auto", "off", "force"]


@dataclasses.dataclass(frozen=True)
class Settings:
    """One run's configuration. Frozen: change it through :func:`configure`, not by mutation."""

    #: Bench the full config grid instead of the cached top-K. Formerly MINIWORLD_RUN_AUTOTUNE.
    #: Kernel groups the autotune build unlocks; read by `autotunes()` below.
    autotune_kernels: frozenset[AutotuneKernel] = frozenset()
    run_autotune: bool = False
    #: Record every benched (config -> ms) for an autotune-cache build. Formerly
    #: MINIWORLD_AUTOTUNE_CAPTURE.
    capture: bool = False
    #: During a build, leave a key the cache ALREADY holds alone instead of re-benching its grid.
    #:
    #: `build all` had to choose between two work lists, and neither is complete on its own. The
    #: per-op sweep covers what registry.csv DECLARES -- every kernel, every shape bucket -- but
    #: drives each kernel through its own driver, so it never produces the constexpr combinations a
    #: module's real dispatch does (`SAVE_PREACT=1`, `ADD_RESIDUAL=0`, `H2=512,K=256`). Measured on
    #: an A6000: a cache built that way answers `missing_pairs 0` to the declared question and
    #: misses 363 lookups the module matrix actually makes, across 42 of 91 ops. Driving modules
    #: reaches those keys, and reaches only the 48 of 91 kernels some module happens to dispatch.
    #:
    #: So the default now runs both, and this is what makes the second pass affordable: with it
    #: set, a build behaves like a run for keys that are already tuned (cached top-K, ~3 configs)
    #: and like a build for keys that are not (full grid). Only the gaps get searched. Without it
    #: the module pass re-benches every key the op sweep already did -- the 244 GPU-h that made
    #: the two work lists an either/or in the first place.
    fill_gaps: bool = False
    #: Worker processes used to PRE-compile an autotune round before it is timed. None = one per
    #: usable core (capped). 1 disables it. Only a build ever sets this; see autotune.capture.
    compile_jobs: int | None = None
    #: Compile a PROBE slice of a round first, fit the two models in `autotune/viability.py` and
    #: `autotune/compile_budget.py` from it, and skip the configs the probes prove cannot pay off
    #: -- they need more shared memory than the card has, or their compile does not finish inside
    #: the budget. Both are fitted per kernel and validated per kernel; a kernel neither can
    #: describe compiles its whole grid, which is what a build did before either existed.
    #:
    #: Only a build ever sets this. Measured on an A6000, four ops, 7 (op, bucket) pairs, each arm
    #: from a cold cache:
    #:
    #:     arm wall      1166 s -> 948 s      19% faster
    #:     compile        2229 -> 1208 s      46% less
    #:     one op         2,592 configs -> 1,116 ruled out, compile 763 s -> 247 s
    #:     the op with nothing to rule out    0 ruled out, and the kernel reported undescribable
    #:
    #: And the control the comparison needed, because the instrument is coarse here: running the
    #: SAME settings twice disagreed about 3 of the 7 buckets, worst penalty 12.5%. With the
    #: predictor on it disagreed about 6 of 7 -- worst penalty also 12.5%, which is 8,192 ns
    #: against 9,216 ns, one step of the event timer's own 1.024 us. The disagreement is the same
    #: KIND and the same SIZE as the instrument's disagreement with itself; there are more of them
    #: because ruling a config out changes which of several TIED configs is drawn.
    #:
    #: Still default-off: seven buckets on one card is a thin sample for a default, and a full
    #: build compares 1,244. Turn it on for a build and compare that build's cache against the
    #: shipped one -- that is the evidence a default needs.
    predict_unusable: bool = False
    #: Megabytes zeroed before each timed iteration, to evict the previous one from L2. 0 leaves
    #: triton's own buffer, which is 256 MB on every card.
    #:
    #: 256 MB is sized for the largest L2 in the fleet. On a 6 MB A6000 it is 40x more than the
    #: eviction needs, and it dominates: measured on an idle card, zeroing it takes 390 us against
    #: 10 us for the kernel being timed -- 97% of a bench iteration.
    #:
    #: MUST move together with `bench_rep_ms`. `do_bench` picks its repeat count to fill a time
    #: budget, so a cheaper iteration alone just buys more iterations: 16 MB at triton's 100 ms
    #: budget ran 4,243 launches in 145 ms against 348 in 104 ms -- slower. 16 MB at 10 ms ran 452
    #: launches in 15 ms: seven times cheaper than the default WITH thirty percent more samples.
    bench_clear_mb: int = 0
    #: Milliseconds of measurement per config. 0 leaves triton's 100 (with 25 ms of warmup); a
    #: value here also scales the warmup by the same 1:4. See `bench_clear_mb` -- these two are
    #: one decision.
    #:
    #: Changing either changes the number a config REPORTS -- 5.1 us against 8.2 us for the same
    #: kernel -- and an A/B has now shown that it changes what the build CHOOSES, for the worse.
    #:
    #: 16 MB at 10 ms, four units, against the same units unchanged: five of seven buckets picked a
    #: different config, and scored on the unchanged arm's own measurements those picks were 8%,
    #: 13%, 21%, 25% and 40% slower. The winner was never removed -- it was benched every time and
    #: placed 8th, 11th, 17th and 44th.
    #:
    #: The mechanism is the one that makes the clear expensive in the first place. Zeroing 256 MB
    #: takes 390 us, which is long enough for the CPU to stay ahead of the card, so the events
    #: bracket the kernel. Zeroing 16 MB takes ~24 us, the launch loop becomes the bottleneck, and
    #: the events bracket the LAUNCH GAP instead -- every fast kernel reads about 8.19 us, which is
    #: a multiple of the timer's own 1.024 us step, and configs stop being distinguishable.
    #:
    #: So this pair is not the way to make benching cheaper. Two things that might be: cut
    #: `bench_rep_ms` while LEAVING the clear alone, which shortens the measurement without
    #: changing what it measures; or `use_cuda_graph`, which removes the launch loop from the
    #: measurement entirely. Neither is measured yet.
    bench_rep_ms: int = 0
    #: Path to a per-card lock file a unit holds while it MEASURES, so two units sharing a card
    #: never measure at once (their readings would both drift). Empty = this unit has the card to
    #: itself. Set by the build when --units-per-gpu > 1; see autotune.capture.
    bench_lock: str = ""
    #: On a cache MISS, how many configs the autotuner may sweep before it gives up on tuning and
    #: uses a heuristic subset instead. 0 disables the fallback and restores the full-grid sweep.
    #:
    #: A miss is the normal state on any GPU nobody has built a cache for, and the full grid is
    #: 205,266 configs -- so without this, the first forward on a new card runs a tuning sweep
    #: inside itself. triton-dejavu solves this with TRITON_DEJAVU_FORCE_FALLBACK and a
    #: user-supplied heuristic; Liger-Kernel skips autotuning entirely and computes BLOCK_SIZE and
    #: num_warps from the row width. This is the same idea: a miss should cost a small, bounded
    #: search, not an unbounded one. A BUILD (run_autotune=True) always gets the full grid.
    autotune_miss_cap: int = 24
    #: How kernel entry points are exposed to ``torch.compile``: "custom_op" (opaque graph node,
    #: keeps surrounding fusion) or "disable" (graph break). Same kernel, same numbers -- the
    #: gradients are bit-identical either way. Read at IMPORT time by kernels._compile.
    #:
    #: "disable" was the default while it was the only mode that could load: 47 of the 58 entry
    #: points wrapped an ``autograd.Function`` method, which ``custom_op`` cannot register, so
    #: selecting the other mode raised on import. Now that every launch has a fake, measured on
    #: an A6000 (Pairformer x4, L=384) "custom_op" is the better default:
    #:
    #:   training, torch.compile, no CUDA graph   164.4 ms -> 156.0 ms   (the main-config regime)
    #:   training, mode="reduce-overhead"         166.0 ms -> 155.1 ms   (was SLOWER than eager)
    #:   training, compile + captured graph       CRASHED  -> 153.8 ms
    #:
    #: "disable" remains for A/B and as the escape hatch if a fake is ever wrong: it needs none.
    compile_wrap: Literal["disable", "custom_op"] = "custom_op"
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
    #: "partial" was removed with the path itself (see kernels/layernorm/notes/
    #: removed-partial-path.md). It is not accepted here rather than silently ignored:
    #: `_resolve_bwd_path` only honours an override that is in `_VALID_BWD_PATHS`, so a
    #: still-declared "partial" would have fallen through to calibration with no warning.
    layernorm_bwd_path: Literal["persistent", "atomic", "cuda"] | None = None
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
    #: Fused training front for the sm100 in-projection. Formerly MINIWORLD_TRAIN_FRONT_FUSED.
    trimul_train_front_fused: bool = True

    # ---- sm100 (B200) bring-up knobs -------------------------------------------------------- #
    # Debug/bring-up levers for the cute sm100 kernels, moved off MW_* / LNL_* environment
    # variables. These were migrated mechanically: sm_100 has no execution path on the Ampere
    # cards available here, so only import and lint are verified — the kernels' use of these
    # values is not.
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


def _compile_wrap_from_env() -> Literal["disable", "custom_op"]:
    """``MINIWORLD_COMPILE_WRAP``, the one setting that has to come from the environment.

    ``kernels._compile`` reads ``compile_wrap`` when the decorator RUNS, i.e. at kernel-module
    import, so a parent process cannot set it for a child by calling :func:`configure` -- by the
    time the child's ``main()`` runs, every op has already been registered (or not). That is the
    same reason ``MINIWORLD_CONFIG_DIR`` exists, and the CLI's ``--compile-wrap`` sets this.

    An unrecognised value raises rather than silently falling back: the whole point of moving off
    environment variables was that a typo used to change behaviour invisibly.
    """
    raw = os.environ.get("MINIWORLD_COMPILE_WRAP", "").strip()
    if not raw:
        return "custom_op"
    if raw == "custom_op":
        return "custom_op"
    if raw == "disable":
        return "disable"
    msg = (f"MINIWORLD_COMPILE_WRAP={raw!r} is not a compile_wrap mode; "
           f"expected 'disable' or 'custom_op'")
    raise ValueError(msg)


_ACTIVE = Settings(compile_wrap=_compile_wrap_from_env())


def current() -> Settings:
    """The active settings."""
    return _ACTIVE


def configure(**kwargs) -> Settings:
    """Replace the active settings; returns the previous value so a caller can restore it."""
    global _ACTIVE
    previous = _ACTIVE
    fields = {f.name for f in dataclasses.fields(Settings)}
    unknown = set(kwargs) - fields
    if unknown:
        raise TypeError(f"unknown setting(s): {', '.join(sorted(unknown))}")
    if "autotune_kernels" in kwargs and kwargs["autotune_kernels"] is not None:
        names = frozenset(kwargs["autotune_kernels"])
        known = frozenset(get_args(AutotuneKernel))
        unknown = sorted(names - known)
        if unknown:
            raise ValueError(
                f"autotune_kernels={unknown} is not a group anything reads, so it would unlock "
                f"nothing. Known groups: {sorted(known)}.")
        kwargs["autotune_kernels"] = names
    _ACTIVE = dataclasses.replace(previous, **kwargs)
    return previous


def reset() -> None:
    """Restore shipped defaults."""
    global _ACTIVE
    _ACTIVE = Settings()

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
import warnings
from pathlib import Path
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
    #: module's real dispatch does (`SAVE_PREACT=1`, `SAVE_GATE=0`, `H2=512,K=256`). Measured on
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
    #: Where a MISS should tune the full grid and keep the answer, instead of taking the bounded
    #: `autotune_miss_cap` subset and forgetting it.
    #:
    #: The cap exists because the cache read and the build share one call site, and returning the
    #: full grid there is 205,266 configs -- correct for a build, ruinous for a forward. But the
    #: bounded search is also not WRITTEN anywhere, so the next process meets the same shape and
    #: pays for the same 24 configs again, forever, and the cache never learns a shape the build
    #: did not predict.
    #:
    #: Set this to a directory and a miss becomes a build: the full grid is searched and this
    #: process's measurements are written to its own shard there at exit, for `dev merge` to fold
    #: in. A shard per process, never the in-repo tree, so concurrent runs cannot race -- the same
    #: rule `dump_shard` already states.
    #:
    #: OFF by default and it must stay off by default: the first launch at an unseen shape then
    #: blocks for the length of a build unit, which this repo has measured at 15 s to 9 minutes.
    #: That is a deliberate offline cost, not something to hand an inference request. `build all`
    #: predicting the shape is the fix; this is what catches the one it did not.
    autotune_on_miss_shards: str = ""
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

    #: Process-wide application policy; configure before constructing models.
    engine_backend: Literal["auto", "triton"] = "auto"
    #: Connected CUDA bidirectional training widths (subset of 64/128/256/384/512). D64 uses its
    #: own fused path (h100_d64_training), about 2.0x Triton under CUDA-graph replay at L384/768.
    trimul_h100_training_widths: tuple[int, ...] = (64, 128, 256, 384, 512)

    # ---- transition backend selection ------------------------------------------------------ #
    #: Route the sm_90 d in {128,256}/n=4 case of ``triton_transition_fused`` (the raw-op Triton
    #: kernel behind benches / ConditionedTransition) through the hand-CUDA b2b forward.
    #: Formerly MINIWORLD_TRANSITION_CUDA_B2B.
    transition_cuda_b2b: bool = True
    #: Route d=128/n=4 bf16 Transition on sm_90 through the fused hand-CUDA forward and
    #: backward (two launches instead of five; ~1.9x the Triton residual path at L=384, both
    #: training and inference). Falls back on any other shape, dtype or architecture.
    #: MINIWORLD_TRANSITION_FUSED_SM90A=0 also turns it off.
    transition_fused_sm90a: bool = True
    #: Route the pair-bias attention core, when it is asked to compute in bf16 (``compute_dtype=torch.bfloat16``), on
    #: sm_90 with 16 heads x 48, B == 1 and L a multiple of 128, through the hand-CUDA forward and backward
    #: (kernels/augmented_attention/cuda: fwd+bwd 2.5x the bf16 Triton core at L=768, A=48, peak memory -1.9 GB).
    #: Falls back on anything else. MINIWORLD_AUGATTN_BF16_SM90=0 also turns it off.
    augmented_attention_bf16_sm90: bool = True
    #: The same routing on sm_100 (B200) through the tcgen05 forward and backward
    #: (kernels/augmented_attention/cuda/sm100: fwd+bwd 3.0x the bf16 Triton core at L=768, A=48), when additionally
    #: A is even and there is no key mask. Falls back on anything else. MINIWORLD_AUGATTN_BF16_SM100=0 also turns it off.
    augmented_attention_bf16_sm100: bool = True
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

    # ---- SWA atom DiT fused block (kernels/swa_dit) ----------------------------------------- #
    # Moved from team-gm (commit 14f2c73), whose environment switches these replace one for one, with the same
    # defaults. `engine_backend="triton"` turns every *_cuda stage off as well.
    #: Let `modules.swa_dit.SWADiTBlock` (implementation other than pytorch) run the fused block when
    #: `kernels.swa_dit.interface.refusal` accepts the call (bf16 or fp32, d_atom 128 / 4 heads, window 128, SwiGLU hidden 256);
    #: off keeps the per-op path. Formerly team-gm's opt-in MINIWORLD_SWA_FUSED on SWAAtomTransformer.
    swa_dit_fused: bool = True
    #: B200 only: keep the engine's hand-written Triton norm kernels (`LayerNorm` / `RMSNorm` with implementation MINIWORLD) instead of
    #: PyTorch ops that `torch.compile` fuses (the default on B200: no hand-written Triton on that card). Other cards are unaffected.
    b200_engine_triton: bool = False
    #: Compute the 3D RoPE angles of `modules.swa_atom_attention.build_3d_rope` the way engines before 2026-10-06 did: with
    #: einsums, which under autocast run as bf16 matmuls. The angles are then wrong (UID rotations off by up to a whole turn at
    #: ids in the thousands), and a model TRAINED that way has adapted to them: the v2.0.0 phase 2a checkpoint (trained with engine
    #: 135b7966 + 4) has a diffusion loss of 0.13 with the correct fp32 angles and 0.057 with these. Turn this on to resume or
    #: evaluate such checkpoints; leave it off for anything trained from scratch.
    rope_angles_bf16_compat: bool = False
    #: Serve the qkvg forward (RMSNorm + adaLN + Q/K/V/gate projections + q/k-norm + RoPE) with the hand-CUDA sm_90a
    #: wgmma kernel on sm_90; off (or any other card, or a failed build) takes the Triton kernel. Formerly SWA_QKVG_FWD.
    swa_dit_qkvg_fwd_cuda: bool = True
    #: Same for the out-projection + gated residual + SwiGLU FFN forward. Formerly SWA_FFN_FWD.
    swa_dit_ffn_fwd_cuda: bool = True
    #: Same for the FFN backward (used with `swa_dit_ffn_dw="mat"` only). Formerly SWA_FFN_BWD.
    swa_dit_ffn_bwd_cuda: bool = True
    #: FFN weight gradients: "mat" materialises [da | db] and h and takes dW from cuBLAS; "fused" recomputes them in
    #: the `_swa_ffn_dw_kernel` Triton kernel (correct, slower in Triton: 255 registers). Formerly SWA_FFN_DW.
    swa_dit_ffn_dw: Literal["mat", "fused"] = "mat"
    #: dtype of dq1, the gradient handed from the FFN half of the backward to the attention half, on the Triton FFN
    #: backward (the hand-CUDA one writes bf16). Formerly SWA_DQ1.
    swa_dit_dq1: Literal["bf16", "fp32"] = "bf16"

    # ---- diagnostics ----------------------------------------------------------------------- #
    #: jaxtyped+beartype decoration on annotated functions. Off by default: it is a per-call cost.
    #: Formerly SHOULD_TYPECHECK.
    typecheck: bool = False
    #: Assert that SWA's atom blocks really are front-packed, at runtime. Formerly
    #: SWA_CHECK_FRONT_PACKED.
    swa_check_front_packed: bool = False
    #: FlashAttention-4 SWA attention (`modules.swa_atom_attention`): the forward saves its output and log-sum-exp and the backward
    #: reads them (`flash_window_fa4`), with the padding-row cleaning left to the compiled graph around the op. Off keeps
    #: `flash_window_seqused`, whose backward re-runs the flash forward and cleans q/k/v/dq/dk/dv with eager elementwise
    #: kernels (about 11 launches per block and a recomputed forward). Same numbers either way.
    swa_flash_saves_lse: bool = True

    # ---- kernel switches (formerly MINIWORLD_* environment variables; name = the variable without its prefix, lower case)
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT`` (default on).
    bias_only_dit: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_HOIST`` (default on).
    bias_only_dit_hoist: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_TRAIN`` (default on).
    bias_only_dit_train: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_TRAIN_PACK1`` (default on).
    bias_only_dit_train_pack1: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_BWD_DG`` (default on).
    bias_only_dit_bwd_dg: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED`` (default on).
    bias_only_dit_bwd_fused: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_BWD_PBFIN`` (default on).
    bias_only_dit_bwd_pbfin: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_BWD_MID`` (default off).
    bias_only_dit_bwd_mid: bool = False
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_BWD_PVDPB`` (default on).
    bias_only_dit_bwd_pvdpb: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_BWD_PVPDL`` (default on).
    bias_only_dit_bwd_pvpdl: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_PAIR_EXACT`` (default off).
    bias_only_dit_pair_exact: bool = False
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_CORE`` (default on).
    bias_only_dit_core: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_INF3`` (default on).
    bias_only_dit_inf3: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA`` (default on).
    bias_only_dit_inf3_2cta: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_INF3_BF16`` (default on).
    bias_only_dit_inf3_bf16: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_INF3_P_EARLY`` (default on).
    bias_only_dit_inf3_p_early: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_PV_BATCH`` (default on).
    bias_only_dit_pv_batch: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_TF32`` (default on).
    bias_only_dit_tf32: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_COND`` (default off).
    bias_only_dit_cond: bool = False
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_RESLN`` (default off).
    bias_only_dit_resln: bool = False
    #: Formerly ``MINIWORLD_LOCAL_DIT_HOIST`` (default on).
    local_dit_hoist: bool = True
    #: Formerly ``MINIWORLD_LOCAL_DIT_TF32_FUSED`` (default on).
    local_dit_tf32_fused: bool = True
    #: Formerly ``MINIWORLD_LOCAL_DIT_TF32`` (default on).
    local_dit_tf32: bool = True
    #: Formerly ``MINIWORLD_LOCAL_DIT_SM100`` (default on).
    local_dit_sm100: bool = True
    #: Formerly ``MINIWORLD_OPM_TRAIN_SAVE_O`` (default on).
    opm_train_save_o: bool = True
    #: Formerly ``MINIWORLD_TOKEN_DIT_ROWS_CUDA`` (default on).
    token_dit_rows_cuda: bool = True
    #: Formerly ``MINIWORLD_TOKEN_DIT_TRAIN`` (default on).
    token_dit_train: bool = True
    #: Formerly ``MINIWORLD_AUGATTN_BF16_SM90`` (default on).
    augattn_bf16_sm90: bool = True
    #: Formerly ``MINIWORLD_AUGATTN_BF16_SM100`` (default on).
    augattn_bf16_sm100: bool = True
    #: Formerly ``MINIWORLD_TF32_PDL`` (default on).
    tf32_pdl: bool = True
    #: Formerly ``MINIWORLD_SWA_DIT_SM100`` (default on).
    swa_dit_sm100: bool = True
    #: Formerly ``MINIWORLD_SWA_DIT_TF32`` (default on).
    swa_dit_tf32: bool = True
    #: Formerly ``MINIWORLD_SWA_FUSED`` (default off).
    swa_fused: bool = False
    #: Formerly ``MINIWORLD_SWA_GLOBAL_BF16`` (default off).
    swa_global_bf16: bool = False
    #: Formerly ``MINIWORLD_TRANSITION_FUSED_SM100A`` (default on).
    transition_fused_sm100a: bool = True
    #: Formerly ``MINIWORLD_TRANSITION_FUSED_SM80`` (default on).
    transition_fused_sm80: bool = True
    #: Formerly ``MINIWORLD_TRANSITION_WIDE_SM90A`` (default on).
    transition_wide_sm90a: bool = True
    #: Formerly ``MINIWORLD_TRANSITION_WIDE_SAVE_H`` (default off).
    transition_wide_save_h: bool = False
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_TF32_GLU`` (default off).
    bias_only_dit_tf32_glu: bool = False
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_TF32_GLU_BWD`` (default on).
    bias_only_dit_tf32_glu_bwd: bool = True
    #: Formerly ``MINIWORLD_STATIC_WEIGHTS`` (default off).
    static_weights: bool = False
    #: Formerly ``MINIWORLD_STATIC_CONDITIONING`` (default off).
    static_conditioning: bool = False
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_PV_VB``: force this value; None lets the heuristic choose.
    bias_only_dit_pv_vb: int | None = None
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_SG``: force this value; None lets the heuristic choose.
    bias_only_dit_sg: int | None = None
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_INF3_CL``: force this value; None lets the heuristic choose.
    bias_only_dit_inf3_cl: int | None = None
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_INF3_TAIL_CL``: force this value; None lets the heuristic choose.
    bias_only_dit_inf3_tail_cl: int | None = None
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_HOIST_WSPLIT``: force this value; None lets the heuristic choose.
    bias_only_dit_hoist_wsplit: int | None = None
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_WGRAD_SPLIT``: force this value; None lets the heuristic choose.
    bias_only_dit_wgrad_split: int | None = None
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR``: True / False forces it; None lets the heuristic choose.
    bias_only_dit_dpb_pair: bool | None = None
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_TF32_DPB_PAIR``: True / False forces it; None lets the heuristic choose.
    bias_only_dit_tf32_dpb_pair: bool | None = None
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_BWD_TAIL``.
    bias_only_dit_bwd_tail: str = "split3"
    #: Formerly ``MINIWORLD_TOKEN_DIT_GEMM_SWIGLU``.
    token_dit_gemm_swiglu: str = "auto"
    #: Formerly ``MINIWORLD_TF32_FUSED_UNITS``.
    tf32_fused_units: str = "auto"
    #: Formerly ``MINIWORLD_TRANSITION_SM100_REPL``; 0 keeps the built-in value.
    transition_sm100_repl: int = 0
    #: Formerly ``MINIWORLD_TRANSITION_WIDE_SMALL_TILES``; 0 keeps the built-in value.
    transition_wide_small_tiles: int = 0
    #: Formerly ``MINIWORLD_TRANSITION_WIDE_SMALL_FWD_TILES``; 0 keeps the built-in value.
    transition_wide_small_fwd_tiles: int = 0
    #: Formerly ``MINIWORLD_TRANSITION_WIDE_REPL``; 0 keeps the built-in value.
    transition_wide_repl: int = 0
    #: Formerly ``MINIWORLD_PLAN_CACHE_DIR``; empty means unset.
    plan_cache_dir: str = ""
    #: Formerly ``MINIWORLD_TRANSITION_SM100_NVCC``; empty means unset.
    transition_sm100_nvcc: str = ""
    #: Formerly ``MINIWORLD_ENGINE_JIT_ROOT``: where JIT-built kernels are cached; empty means ``$XDG_CACHE_HOME/miniworld_engine_jit`` (``~/.cache`` when unset). Use :func:`engine_jit_root`.
    engine_jit_root: str = ""
    #: Formerly ``MINIWORLD_OPM_TRAIN``: False disables the fused OPM training path; None = automatic.
    opm_train: bool | None = None
    #: Formerly ``MINIWORLD_PWA_TRAIN``: False disables the fused PWA training path; None = automatic.
    pwa_train: bool | None = None
    #: Formerly ``MINIWORLD_PWA_INFER``: like ``pwa_train`` for inference; None follows ``pwa_train``.
    pwa_infer: bool | None = None
    #: Formerly MiniWorld's ``MINIWORLD_FUSED_PAIR_INIT``: the fused token pair-init kernel.
    token_pair_init_fused: bool = True
    # --- A100 (sm_80) kernel switches, formerly MINIWORLD_*_SM80 environment variables ---
    #: The A100 sm_80 cleanup shortcuts of the attention / OPM / PWA / TriAttn glue (off: the earlier
    #: packed-weight and saved-intermediate path). Formerly MINIWORLD_A100_CLEANUP.
    a100_cleanup: bool = True
    #: The A100 hand-CUDA augmented-attention integration (off: the module's Triton path serves).
    #: Formerly MINIWORLD_AUGATTN_SM80.
    augattn_sm80: bool = True
    #: The fused projection + pair-bias composition of the A100 augmented attention (off: the module's
    #: own projections and PyTorch gates around the core; bf16 only). Formerly
    #: MINIWORLD_AUGATTN_SM80_FUSED.
    augattn_sm80_fused: bool = True
    #: The streaming pair-bias kernel for widths without a dedicated one (off: ATen LayerNorm +
    #: cuBLAS). Formerly MINIWORLD_AUGATTN_SM80_PBGEN.
    augattn_sm80_pbgen: bool = True
    #: Rows a thread of the generic pair-bias kernel takes (1, 2 or 4), for A/B runs. Read at IMPORT of
    #: kernels.augmented_attention.cuda.sm80. Formerly MINIWORLD_AUGATTN_SM80_PBGEN_ROWS.
    augattn_sm80_pbgen_rows: int = 2
    #: Extra -D / nvcc flags for the sm_80 augmented-attention extension (A/B experiments; their own
    #: build). Formerly MINIWORLD_AUGATTN_SM80_FLAGS.
    augattn_sm80_flags: str = ""
    #: Which calls the A100 whole-op augmented-attention core serves: "auto" (training calls the core
    #: wins), "all", or "0" (none). Formerly MINIWORLD_AUGATTN_SM80_OPS.
    augattn_sm80_ops: str = "auto"
    #: Opt-in: the shared-mask uniform core of the atom augmented-attention forward (bf16, d 128, head
    #: dim 32). Formerly MINIWORLD_AUGATTN_SM80_SHARED_CORE.
    augattn_sm80_shared_core: bool = False
    #: Opt-in: the shared-mask score core of the atom augmented-attention forward (bf16, d 128, head
    #: dim 32). Formerly MINIWORLD_AUGATTN_SM80_SHARED_SCORE.
    augattn_sm80_shared_score: bool = False
    #: Force the shared-mask uniform-softmax kernel of the atom augmented attention on or off; None
    #: picks it for bf16, d 128, head dim 32. Formerly MINIWORLD_AUGATTN_SM80_SHARED_UNIFORM.
    augattn_sm80_shared_uniform: bool | None = None
    #: Force the native resident qkvg projection of the atom augmented attention on or off; None picks
    #: it per shape. Formerly MINIWORLD_AUGATTN_SM80_ATOM_QKVG.
    augattn_sm80_atom_qkvg: bool | None = None
    #: Force the tiled qkvg projection of the atom augmented attention on or off; None picks it per
    #: shape (bf16 inference). Formerly MINIWORLD_AUGATTN_SM80_ATOM_QKVG_TILED.
    augattn_sm80_atom_qkvg_tiled: bool | None = None
    #: Force the fused output projection of the atom augmented attention on or off; None picks it per
    #: shape. Formerly MINIWORLD_AUGATTN_SM80_ATOM_OUTPUT_FUSED.
    augattn_sm80_atom_output_fused: bool | None = None
    #: Force the tiled output projection of the small A1 atom inference rows on or off; None means on.
    #: Formerly MINIWORLD_AUGATTN_SM80_ATOM_OUTPUT_TILED.
    augattn_sm80_atom_output_tiled: bool | None = None
    #: dQ accumulation order variant of the plain backward core of the atom augmented attention (0: the
    #: shipped one). Formerly MINIWORLD_DENSE_CORE_DQ_ORDER.
    dense_core_dq_order: int = 0
    #: dK/dV accumulation order variant of the plain backward core of the atom augmented attention (0:
    #: the shipped one). Formerly MINIWORLD_DENSE_CORE_DKV_ORDER.
    dense_core_dkv_order: int = 0
    #: dQ accumulation order variant of the mixed-precision backward core of the atom augmented
    #: attention (0: the shipped one). Formerly MINIWORLD_DENSE_E274_DQ_ORDER.
    dense_e274_dq_order: int = 0
    #: dK/dV accumulation order variant of the mixed-precision backward core of the atom augmented
    #: attention (0: the shipped one). Formerly MINIWORLD_DENSE_E274_DKV_ORDER.
    dense_e274_dkv_order: int = 0
    #: The A100 hand-CUDA AttentionPairBias integration (off: the module's Triton path serves).
    #: Formerly MINIWORLD_APB_SM80.
    apb_sm80: bool = True
    #: The 16-bit training path of the A100 AttentionPairBias. Formerly MINIWORLD_APB_SM80_TRAIN16.
    apb_sm80_train16: bool = True
    #: Typed (bf16) gradient outputs of the A100 AttentionPairBias backward. Formerly
    #: MINIWORLD_APB_SM80_TYPED_GRAD.
    apb_sm80_typed_grad: bool = True
    #: Save the natural-layout activations for the A100 AttentionPairBias backward. Formerly
    #: MINIWORLD_APB_SM80_SAVE_NATURAL.
    apb_sm80_save_natural: bool = True
    #: The A100 bias-only attention door: "1" serves it (small-L training excepted), "all" serves every
    #: call `serves` accepts, "0" turns it off. Formerly MINIWORLD_BIAS_ONLY_ATTN_SM80.
    bias_only_attn_sm80: str = "1"
    #: The A100 single-block weight pack and fused pair LayerNorm / projection of the bias-only DiT
    #: runner (off: the earlier preparation). Formerly MINIWORLD_BIAS_ONLY_DIT_PREP.
    bias_only_dit_prep: bool = True
    #: Depth of the cp.async ring of the sm_80 bias-only DiT core, for A/B runs. Read at IMPORT of
    #: kernels.bias_only_dit.cuda.sm80. Formerly MINIWORLD_BIAS_ONLY_DIT_SM80_STAGES.
    bias_only_dit_sm80_stages: int = 2
    #: Forced samples per CTA (1..4) of the sm_80 bias-only DiT core; None picks the fewest CTAs that
    #: fill the card. Formerly MINIWORLD_BIAS_ONLY_DIT_SM80_SG.
    bias_only_dit_sm80_sg: int | None = None
    #: Schedule of the sm_80 bias-gradient core (0..3); None is 0. Read at IMPORT of
    #: kernels.bias_only_dit.cuda.sm80. Formerly MINIWORLD_BIAS_ONLY_DIT_SM80_DPB.
    bias_only_dit_sm80_dpb: int | None = None
    #: The A100 hand-CUDA OuterProductMean integration (off: the module's Triton path serves). Formerly
    #: MINIWORLD_OPM_SM80.
    opm_sm80: bool = True
    #: Keep O for the OPM backward (off: recompute it; saves 1.2 GB at L768). Formerly
    #: MINIWORLD_OPM_SM80_SAVE_O.
    opm_sm80_save_o: bool = True
    #: Opt-in: the C++ host-overhead pipeline of the short OPM training path (S 1024, d_msa 64).
    #: Formerly MINIWORLD_OPM_SM80_PIPELINE.
    opm_sm80_pipeline: bool = False
    #: The native pipeline of the L128 OPM training path (off: the previous short training
    #: implementation). Formerly MINIWORLD_OPM_SM80_NATIVE128_PIPELINE.
    opm_sm80_native128_pipeline: bool = True
    #: Opt-in: typed (bf16) gradient outputs of the OPM backward. Formerly
    #: MINIWORLD_OPM_SM80_TYPED_GRAD.
    opm_sm80_typed_grad: bool = False
    #: Force the native mask-count kernel of the OPM on or off; None picks it per shape (inference
    #: only). Formerly MINIWORLD_OPM_SM80_MASK_COUNT_NATIVE.
    opm_sm80_mask_count_native: bool | None = None
    #: Force the mask epilogue of the OPM on or off; None picks it per shape (inference only). Formerly
    #: MINIWORLD_OPM_SM80_MASK_EPILOGUE.
    opm_sm80_mask_epilogue: bool | None = None
    #: Schedule variant (tile buffers) of the OPM forward kernels, for experiments (0: the shipped
    #: one). Formerly MINIWORLD_OPM_SM80_PL.
    opm_sm80_pl: int = 0
    #: Schedule variant (ring depth / CTAs per SM) of the OPM epilogue, for experiments (0: the shipped
    #: one). Formerly MINIWORLD_OPM_SM80_EPI.
    opm_sm80_epi: int = 0
    #: Schedule variant (ring depth / CTAs per SM) of the OPM dG kernel, for experiments (0: the
    #: shipped one). Formerly MINIWORLD_OPM_SM80_DG.
    opm_sm80_dg: int = 0
    #: Schedule variant (buffers / CTAs per SM) of the OPM pair backward, for experiments (0: the
    #: shipped one). Formerly MINIWORLD_OPM_SM80_PB.
    opm_sm80_pb: int = 0
    #: The A100 hand-CUDA MSAPairWeightedAveraging integration (off: the module's previous path
    #: serves). Formerly MINIWORLD_PWA_SM80.
    pwa_sm80: bool = True
    #: Native broadcast contractions for the qualified L128 PWA training rows. Formerly
    #: MINIWORLD_PWA_SM80_BROADCAST_NATIVE.
    pwa_sm80_broadcast_native: bool = True
    #: Short-row broadcast contraction variant of the PWA: "1" pipeline, "2" / "3" capture variants,
    #: "0" none. Formerly MINIWORLD_PWA_SM80_BROADCAST_PIPELINE.
    pwa_sm80_broadcast_pipeline: str = "0"
    #: Force the fused contract + gate kernel of the PWA on or off; None means off for training and on
    #: for inference. Formerly MINIWORLD_PWA_SM80_CONTRACT_GATE_FUSED.
    pwa_sm80_contract_gate_fused: bool | None = None
    #: Opt-in: the early-exit native dw kernel of the short PWA rows. Formerly
    #: MINIWORLD_PWA_SM80_DW_SHORT.
    pwa_sm80_dw_short: bool = False
    #: Force the native dw kernel of the PWA on or off; None picks it for depth 384 / 640. Formerly
    #: MINIWORLD_PWA_SM80_DW_NATIVE.
    pwa_sm80_dw_native: bool | None = None
    #: Force typed (bf16) gradient outputs of the PWA backward on or off; None picks it for depth 128 /
    #: 256 / 384. Formerly MINIWORLD_PWA_SM80_TYPED_GRAD.
    pwa_sm80_typed_grad: bool | None = None
    #: Force the tuned glue kernel of the PWA backward on or off; None picks it for depth 256 .. 768.
    #: Formerly MINIWORLD_PWA_SM80_GLUE_TUNED.
    pwa_sm80_glue_tuned: bool | None = None
    #: Force the fused gradient-reduce of the PWA backward on or off; None picks it for the (1024, 128,
    #: 64) / L128 row. Formerly MINIWORLD_PWA_SM80_GRAD_REDUCE.
    pwa_sm80_grad_reduce: bool | None = None
    #: The partial-sum pair gradient of the PWA backward. Formerly MINIWORLD_PWA_SM80_PAIR_PARTIAL.
    pwa_sm80_pair_partial: bool = True
    #: Forced number of chunks (1, 2, 4, 8) the PWA splits the MSA rows in; None picks it from the
    #: depth. Formerly MINIWORLD_PWA_SM80_SPLIT.
    pwa_sm80_split: int | None = None
    #: Schedule variant (CTAs per SM) of the PWA pair backward, for experiments (0: the shipped one).
    #: Formerly MINIWORLD_PWA_SM80_PB.
    pwa_sm80_pb: int = 0
    #: The A100 hand-CUDA TriangleAttention integration (off: the Triton path serves). Formerly
    #: MINIWORLD_TRIATTN_SM80.
    triattn_sm80: bool = True
    #: The compact (masked-row gather) inference schedule of the sm_80 triangle attention. Formerly
    #: MINIWORLD_TRIATTN_SM80_COMPACT.
    triattn_sm80_compact: bool = True
    #: The internal-tile inference schedule (128 queries, 48 keys at L divisible by 384); off keeps the
    #: preceding schedule. Formerly MINIWORLD_TRIATTN_SM80_INTERNAL.
    triattn_sm80_internal: bool = True
    #: The tuned inference schedules of the sm_80 triangle attention (off: the original 32-key
    #: schedule). Formerly MINIWORLD_TRIATTN_SM80_TUNED.
    triattn_sm80_tuned: bool = True
    #: The fixed-shift softmax of the sm_80 triangle attention inference at L >= 512 (off: the online
    #: schedule). Formerly MINIWORLD_TRIATTN_SM80_SOFTMAX.
    triattn_sm80_softmax: bool = True
    #: 32-wide padding of the d 256 projection GEMMs at L >= 512 (off: 8). Formerly
    #: MINIWORLD_TRIATTN_SM80_GEMM_PAD.
    triattn_sm80_gemm_pad: bool = True
    #: The native dropout RNG of the sm_80 triangle attention training path. Formerly
    #: MINIWORLD_TRIATTN_SM80_NATIVE_RNG.
    triattn_sm80_native_rng: bool = True
    #: Capture-aware launches of the sm_80 triangle attention training path. Formerly
    #: MINIWORLD_TRIATTN_SM80_TRAIN_CAPTURE_AWARE.
    triattn_sm80_train_capture_aware: bool = True
    #: Opt-in: direct inner launches of the sm_80 triangle attention training path. Formerly
    #: MINIWORLD_TRIATTN_SM80_TRAIN_INNER.
    triattn_sm80_train_inner: bool = False
    #: Opt-in: the row engine instead of the fused one at d_pair 64 / 128 with 2 / 4 heads (A/B).
    #: Formerly MINIWORLD_TRIATTN_SM80_ROWS.
    triattn_sm80_rows: bool = False
    #: Opt-in: d_pair 128 (hidden 128, 4 heads) takes the wide path too (tests, A/B). Formerly
    #: MINIWORLD_TRIATTN_SM80_WIDE.
    triattn_sm80_wide: bool = False
    #: Force the 8-row native dQ kernel of the sm_80 triangle attention backward on or off; None picks
    #: it for L 640 / 768. Formerly MINIWORLD_TRIATTN_SM80_DQ_ROWS.
    triattn_sm80_dq_rows: bool | None = None
    #: Force the native bias-gradient GEMM of the sm_80 triangle attention backward on or off; None
    #: picks it for L 768. Formerly MINIWORLD_TRIATTN_SM80_NATIVE_DB.
    triattn_sm80_native_db: bool | None = None
    #: Force the native Wo-gradient GEMM of the sm_80 triangle attention backward on or off; None picks
    #: it for L 768. Formerly MINIWORLD_TRIATTN_SM80_NATIVE_WO.
    triattn_sm80_native_wo: bool | None = None
    #: Force the native fused output projection of the sm_80 triangle attention inference on or off;
    #: None picks it for L 128. Formerly MINIWORLD_TRIATTN_SM80_NATIVE_OUT.
    triattn_sm80_native_out: bool | None = None
    #: Gamma-gradient mode of the L384 sm_80 triangle attention backward: "2" exact finalize, "1" exact
    #: gamma only, otherwise the plain path. Formerly MINIWORLD_TRIATTN_SM80_EXACT_GAMMA.
    triattn_sm80_exact_gamma: str = "2"
    #: Extra -D / nvcc flags for the sm_80 triangle-attention extension (A/B experiments; their own
    #: build). Formerly MINIWORLD_TRIATTN_SM80_FLAGS.
    triattn_sm80_flags: str = ""
    #: The A100 hand-CUDA TriangleMultiplication in-projection (off: the Triton path serves; shared by
    #: the D128 and wide paths). Formerly MINIWORLD_TRIMUL_SM80.
    trimul_sm80: bool = True
    #: The wide (D256 / D512) set of the A100 TriangleMultiplication (off: that set alone). Formerly
    #: MINIWORLD_TRIMUL_SM80_WIDE.
    trimul_sm80_wide: bool = True
    #: The D512 native training path (off: the previous D512 training fallback). Formerly
    #: MINIWORLD_TRIMUL_SM80_WIDE512.
    trimul_sm80_wide512: bool = True
    #: The row-mask k1 kernel of the wide TriangleMultiplication inference. Formerly
    #: MINIWORLD_TRIMUL_SM80_ROW_MASK.
    trimul_sm80_row_mask: bool = True
    #: The tuned k3 kernel of the wide TriangleMultiplication inference. Formerly
    #: MINIWORLD_TRIMUL_SM80_TUNED.
    trimul_sm80_tuned: bool = True
    #: Extra -D flags for the sm_80 TriangleMultiplication extension (A/B experiments; their own
    #: build). Formerly MINIWORLD_TRIMUL_SM80_FLAGS.
    trimul_sm80_flags: str = ""
    #: Extra -D flags for the wide sm_80 TriangleMultiplication extensions (A/B experiments; their own
    #: build). Formerly MINIWORLD_TRIMUL_SM80_WIDE_FLAGS.
    trimul_sm80_wide_flags: str = ""
    #: The fused atom-width (d = dc = 128) AdaLN kernels of the sm_80 path. Formerly ``MINIWORLD_ADALN_ATOM``.
    adaln_atom: bool = True
    #: Forced launch config of the atom AdaLN forward (-1: picked by row count). Formerly ``MINIWORLD_ADALN_ATOM_CFG``.
    adaln_atom_cfg: int = -1
    #: The fp32 (TF32) fused atom AdaLN forward. Formerly ``MINIWORLD_ADALN_ATOM_TF32``.
    adaln_atom_tf32: bool = True
    #: Side-stream branches of the sm_80 AdaLN backward (off: everything on the current stream). Formerly ``MINIWORLD_ADALN_BRANCH``.
    adaln_branch: bool = True
    #: Fewest rows of fp32 atom-width AdaLN inference served by the sm_80 path (0: all). Formerly ``MINIWORLD_ADALN_FP32_ATOM_MIN_ROWS``.
    adaln_fp32_atom_min_rows: int = 8192
    #: The A100 hand-CUDA AdaLN path (off: the Triton kernels). Formerly ``MINIWORLD_ADALN_SM80``.
    adaln_sm80: bool = True
    #: Fused conditioning weight-gradient of the sm_80 atom AdaLN backward (opt-in). Formerly ``MINIWORLD_ADALN_SM80_ATOM_COND_WGRAD_FUSED``.
    adaln_sm80_atom_cond_wgrad_fused: bool = False
    #: Extra nvcc / -D flags for the sm_80 AdaLN extension (A/B experiments; own build). Formerly ``MINIWORLD_ADALN_SM80_FLAGS``.
    adaln_sm80_flags: str = ""
    #: Native fp32 parameter-gradient cast in the sm_80 AdaLN backward (opt-in). Formerly ``MINIWORLD_ADALN_SM80_NATIVE_PARAM_CAST``.
    adaln_sm80_native_param_cast: bool = False
    #: Packed parameter cast in the sm_80 AdaLN forward (opt-in). Formerly ``MINIWORLD_ADALN_SM80_PACK_PARAM_CAST``.
    adaln_sm80_pack_param_cast: bool = False
    #: Save the bf16 parameter cast for the sm_80 AdaLN backward (opt-in). Formerly ``MINIWORLD_ADALN_SM80_SAVE_PARAM_CAST``.
    adaln_sm80_save_param_cast: bool = False
    #: Blocks per SM of the sm_80 AdaLN column-sum kernels. Read at import of kernels.adaln.cuda.sm80. Formerly ``MINIWORLD_ADALN_WARP_BLOCKS``.
    adaln_warp_blocks: int = 4
    #: The fused atom-width ConditionedTransition kernels of the sm_80 path. Formerly ``MINIWORLD_CONDTRANS_ATOM``.
    condtrans_atom: bool = True
    #: The fp32 (TF32) fused atom ConditionedTransition forward. Formerly ``MINIWORLD_CONDTRANS_ATOM_TF32``.
    condtrans_atom_tf32: bool = True
    #: The A100 hand-CUDA ConditionedTransition path (off: the Triton tail). Formerly ``MINIWORLD_CONDTRANS_SM80``.
    condtrans_sm80: bool = True
    #: Fused conditioning weight-gradient of the sm_80 atom ConditionedTransition backward (opt-in). Formerly ``MINIWORLD_CONDTRANS_SM80_ATOM_COND_WGRAD_FUSED``.
    condtrans_sm80_atom_cond_wgrad_fused: bool = False
    #: Force (True/False) the narrow atom ConditionedTransition kernel; None: by row count. Formerly ``MINIWORLD_CONDTRANS_SM80_ATOM_NARROW``.
    condtrans_sm80_atom_narrow: bool | None = None
    #: Force (True/False) the small-rows atom ConditionedTransition kernel; None: by row count. Formerly ``MINIWORLD_CONDTRANS_SM80_ATOM_SMALL``.
    condtrans_sm80_atom_small: bool | None = None
    #: Force (True/False) the split-hidden atom ConditionedTransition kernel; None: by row count. Formerly ``MINIWORLD_CONDTRANS_SM80_ATOM_SPLIT``.
    condtrans_sm80_atom_split: bool | None = None
    #: Launch config of the split atom kernel; None: 1 at 1024 rows, else 5. Formerly ``MINIWORLD_CONDTRANS_SM80_ATOM_SPLIT_CFG``.
    condtrans_sm80_atom_split_cfg: int | None = None
    #: Split weight-gradient of the sm_80 atom ConditionedTransition backward (opt-in). Formerly ``MINIWORLD_CONDTRANS_SM80_ATOM_WGRAD_SPLIT``.
    condtrans_sm80_atom_wgrad_split: bool = False
    #: Wide weight-gradient variant of the split atom backward (opt-in). Formerly ``MINIWORLD_CONDTRANS_SM80_ATOM_WGRAD_WIDE``.
    condtrans_sm80_atom_wgrad_wide: bool = False
    #: Extra nvcc / -D flags for the sm_80 ConditionedTransition extension (A/B experiments; own build). Formerly ``MINIWORLD_CONDTRANS_SM80_FLAGS``.
    condtrans_sm80_flags: str = ""
    #: Native fp32 parameter-gradient cast in the sm_80 ConditionedTransition backward (opt-in). Formerly ``MINIWORLD_CONDTRANS_SM80_NATIVE_PARAM_CAST``.
    condtrans_sm80_native_param_cast: bool = False
    #: Packed parameter cast in the sm_80 ConditionedTransition forward (opt-in). Formerly ``MINIWORLD_CONDTRANS_SM80_PACK_PARAM_CAST``.
    condtrans_sm80_pack_param_cast: bool = False
    #: Save the bf16 parameter cast for the sm_80 ConditionedTransition backward (opt-in). Formerly ``MINIWORLD_CONDTRANS_SM80_SAVE_PARAM_CAST``.
    condtrans_sm80_save_param_cast: bool = False
    #: The A100 row-scale dropout CUDA kernel (opt-in). Formerly ``MINIWORLD_DROPOUT_SCALE_SM80``.
    dropout_scale_sm80: bool = False
    #: The A100 hand-CUDA gated-projection kernels (off: Triton). Formerly ``MINIWORLD_GATED_SM80``.
    gated_sm80: bool = True
    #: The A100 hand-CUDA LayerNorm + Linear kernels (off: Triton). Formerly ``MINIWORLD_LNLINEAR_SM80``.
    lnlinear_sm80: bool = True
    #: The A100 hand-CUDA LayerNorm / RMSNorm / RoPE rows: "0" off, "force" at every size, anything else on (small training steps stay on Triton). Formerly ``MINIWORLD_NORMS_SM80``.
    norms_sm80: str = "1"
    #: Extra nvcc flags for the sm_80 norm / RMSNorm / RoPE extensions (A/B experiments; own build). Formerly ``MINIWORLD_NORMS_SM80_FLAGS``.
    norms_sm80_flags: str = ""
    #: The token-DiT conditioned-transition prep kernels. Formerly ``MINIWORLD_TOKEN_DIT_PREP``.
    token_dit_prep: bool = True
    #: Extra nvcc flags for the sm_80 fused Transition backward extension (own build). Formerly ``MINIWORLD_TRANSITION_BWD_SM80_FLAGS``.
    transition_bwd_sm80_flags: str = ""
    #: Replication count of the sm_80 fused Transition backward (0: the kernel default). Formerly ``MINIWORLD_TRANSITION_BWD_SM80_NREP``.
    transition_bwd_sm80_nrep: int = 0
    #: The sm_80 fused Transition backward (off: the unfused path). Formerly ``MINIWORLD_TRANSITION_FUSED_BWD_SM80``.
    transition_fused_bwd_sm80: bool = True
    #: The sm_80 fused Transition forward (off: the unfused path). Formerly ``MINIWORLD_TRANSITION_FUSED_FWD_SM80``.
    transition_fused_fwd_sm80: bool = True
    #: Extra nvcc flags for the sm_80 fused Transition forward extension (own build). Formerly ``MINIWORLD_TRANSITION_FWD_SM80_FLAGS``.
    transition_fwd_sm80_flags: str = ""
    #: Cast the wide sm_80 Transition weight gradients with the grad_cast kernel at D256 (opt-in). Formerly ``MINIWORLD_TRANSITION_WIDE_GRAD_CAST``.
    transition_wide_grad_cast: bool = False
    #: FP32 weight-gradient accumulation of the wide sm_80 Transition backward at D256. Formerly ``MINIWORLD_TRANSITION_WIDE_GRAD_FP32``.
    transition_wide_grad_fp32: bool = True
    #: The wide (D up to 768) sm_80 Transition kernels (off: Triton). Formerly ``MINIWORLD_TRANSITION_WIDE_SM80``.
    transition_wide_sm80: bool = True
    #: Extra nvcc flags for the wide sm_80 Transition extension (own build). Formerly ``MINIWORLD_TRANSITION_WIDE_SM80_FLAGS``.
    transition_wide_sm80_flags: str = ""
    #: Serve the qualified A100 atom DiT block training path; off keeps the module composition. Formerly MINIWORLD_ATOM_DIT_SM80.
    atom_dit_sm80: bool = True
    #: Config index of the private three-output gradient-join experiment (atom_dit_sm80_gradient_join). Formerly MINIWORLD_ATOM_DIT_SM80_JOIN_THREE_CFG.
    atom_dit_sm80_join_three_cfg: int = 0
    #: Config index of the private two-output gradient-join experiment (atom_dit_sm80_gradient_join). Formerly MINIWORLD_ATOM_DIT_SM80_JOIN_TWO_CFG.
    atom_dit_sm80_join_two_cfg: int = 0
    #: Use the A100 native 32x128 local DiT block path; off retains the previous module composition. Formerly MINIWORLD_LOCAL_DIT_SM80.
    local_dit_sm80: bool = True
    #: Extra nvcc flags (-DNAME=value) for the A100 MPNN edge-tail kernel build, for A/B experiments; part of the build's name. Formerly MINIWORLD_MPNN_EDGE_FLAGS.
    mpnn_edge_flags: str = ""
    #: Print the ptxas resource report of the A100 MPNN edge-tail build. Formerly MINIWORLD_MPNN_EDGE_PTXAS.
    mpnn_edge_ptxas: bool = False
    #: Run the A100 (sm_80) CUDA MPNN edge kernels; off keeps the Triton ones. Formerly MINIWORLD_MPNN_EDGE_SM80.
    mpnn_edge_sm80: bool = True
    #: Run the A100 (sm_80) CUDA MPNN message-side kernels; off keeps the Triton ones. Formerly MINIWORLD_MPNN_MSG_SM80.
    mpnn_msg_sm80: bool = True
    #: Backward form of the A100 MPNN message kernel: "fused" (default) or "split" (the older two-step form, for A/B runs). Formerly MINIWORLD_MPNN_MSG_SM80_BWD.
    mpnn_msg_sm80_bwd: str = "fused"
    #: Extra nvcc flags for the A100 MPNN message kernel build, for A/B experiments. Formerly MINIWORLD_MPNN_MSG_SM80_FLAGS.
    mpnn_msg_sm80_flags: str = ""
    #: Extra nvcc flags for the A100 MPNN node-message kernel build, for A/B experiments. Formerly MINIWORLD_MPNN_NODE_SM80_FLAGS.
    mpnn_node_sm80_flags: str = ""
    #: Use the A100 CUDA sigmoid gate; off takes the Triton sigmoid_gate_fused. Formerly MINIWORLD_SIGMOID_GATE_SM80.
    sigmoid_gate_sm80: bool = True
    #: Run the A100 (sm_80) CUDA SWA DiT / window-attention kernels; off keeps flash / the Triton stages. Formerly MINIWORLD_SWA_DIT_SM80.
    swa_dit_sm80: bool = True
    #: Use the tuned sm_80 SWA DiT FFN forward. Formerly MINIWORLD_SWA_DIT_SM80_FFN_TUNED.
    swa_dit_sm80_ffn_tuned: bool = True
    #: Extra nvcc flags for the A100 SWA DiT kernel build, for A/B experiments. Formerly MINIWORLD_SWA_DIT_SM80_FLAGS.
    swa_dit_sm80_flags: str = ""
    #: Use the tuned sm_80 SWA DiT inference path. Formerly MINIWORLD_SWA_DIT_SM80_INFER_TUNED.
    swa_dit_sm80_infer_tuned: bool = True
    #: Native sm_80 down-projection weight gradient in the SWA DiT backward. Formerly MINIWORLD_SWA_DIT_SM80_NATIVE_DWD.
    swa_dit_sm80_native_dwd: bool = True
    #: Native sm_80 gradient cast in the SWA DiT backward. None = on at S = 4096 and wherever the grouped5 weight gradient is on. Formerly MINIWORLD_SWA_DIT_SM80_NATIVE_GRAD_CAST.
    swa_dit_sm80_native_grad_cast: bool | None = None
    #: Grouped five-weight native weight gradient in the SWA DiT backward. None = on at S = 8192, off otherwise. Formerly MINIWORLD_SWA_DIT_SM80_NATIVE_GROUPED5_DW.
    swa_dit_sm80_native_grouped5_dw: bool | None = None
    #: Grouped eight-weight native weight gradient with allocation (opt-in). Formerly MINIWORLD_SWA_DIT_SM80_NATIVE_GROUPED8_ALLOC.
    swa_dit_sm80_native_grouped8_alloc: bool = False
    #: Grouped eight-weight native weight gradient (opt-in). Formerly MINIWORLD_SWA_DIT_SM80_NATIVE_GROUPED8_DW.
    swa_dit_sm80_native_grouped8_dw: bool = False
    #: Grouped native weight gradient in the SWA DiT backward. None = on at S = 4096, off otherwise. Formerly MINIWORLD_SWA_DIT_SM80_NATIVE_GROUPED_DW.
    swa_dit_sm80_native_grouped_dw: bool | None = None
    #: Native qkvg weight gradient in the SWA DiT backward (opt-in). Formerly MINIWORLD_SWA_DIT_SM80_NATIVE_QKVG_DW.
    swa_dit_sm80_native_qkvg_dw: bool = False
    #: Use the aliasing variant of the native A100 token DiT training weight pack (opt-in; needs token_dit_sm80_native_train_pack). Formerly MINIWORLD_TOKEN_DIT_SM80_ALIAS_TRAIN_PACK.
    token_dit_sm80_alias_train_pack: bool = False
    #: Direct BF16 q/k/v gradients in the A100 token DiT training attention backward. Formerly MINIWORLD_TOKEN_DIT_SM80_DIRECT_GRAD.
    token_dit_sm80_direct_grad: bool = True
    #: Native CUDA weight pack for the A100 token DiT training path (opt-in). Formerly MINIWORLD_TOKEN_DIT_SM80_NATIVE_TRAIN_PACK.
    token_dit_sm80_native_train_pack: bool = False
    #: Typed-gradient variant of the A100 token DiT training path (opt-in). Formerly MINIWORLD_TOKEN_DIT_TYPED_GRAD.
    token_dit_typed_grad: bool = False

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
    msg = (
        f"MINIWORLD_COMPILE_WRAP={raw!r} is not a compile_wrap mode; "
        f"expected 'disable' or 'custom_op'"
    )
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
                f"nothing. Known groups: {sorted(known)}."
            )
        kwargs["autotune_kernels"] = names
    _ACTIVE = dataclasses.replace(previous, **kwargs)
    return previous


def reset() -> None:
    """Restore shipped defaults."""
    global _ACTIVE
    _ACTIVE = Settings()


def engine_jit_root() -> Path:
    """Directory the JIT-built kernels are cached in."""
    root = current().engine_jit_root
    if root:
        return Path(root)
    xdg = os.environ.get(
        "XDG_CACHE_HOME"
    )  # the standard cache location, like the autotune plan cache
    return (
        Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    ) / "miniworld_engine_jit"


#: Environment variables that used to switch these settings and are not read any more. The value is the setting that replaced
#: the variable ("" when the switch was removed). Setting one now changes nothing, so it is reported instead of ignored silently.
_REMOVED_ENVIRONMENT = {
    "MINIWORLD_AUGATTN_BF16_SM100": "augattn_bf16_sm100",
    "MINIWORLD_AUGATTN_BF16_SM90": "augattn_bf16_sm90",
    "MINIWORLD_BIAS_ONLY_DIT": "bias_only_dit",
    "MINIWORLD_BIAS_ONLY_DIT_BWD_DG": "bias_only_dit_bwd_dg",
    "MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED": "bias_only_dit_bwd_fused",
    "MINIWORLD_BIAS_ONLY_DIT_BWD_MID": "bias_only_dit_bwd_mid",
    "MINIWORLD_BIAS_ONLY_DIT_BWD_PBFIN": "bias_only_dit_bwd_pbfin",
    "MINIWORLD_BIAS_ONLY_DIT_BWD_PVDPB": "bias_only_dit_bwd_pvdpb",
    "MINIWORLD_BIAS_ONLY_DIT_BWD_PVPDL": "bias_only_dit_bwd_pvpdl",
    "MINIWORLD_BIAS_ONLY_DIT_BWD_TAIL": "bias_only_dit_bwd_tail",
    "MINIWORLD_BIAS_ONLY_DIT_COND": "bias_only_dit_cond",
    "MINIWORLD_BIAS_ONLY_DIT_CORE": "bias_only_dit_core",
    "MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR": "bias_only_dit_dpb_pair",
    "MINIWORLD_BIAS_ONLY_DIT_HOIST": "bias_only_dit_hoist",
    "MINIWORLD_BIAS_ONLY_DIT_HOIST_WSPLIT": "bias_only_dit_hoist_wsplit",
    "MINIWORLD_BIAS_ONLY_DIT_INF3": "bias_only_dit_inf3",
    "MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA": "bias_only_dit_inf3_2cta",
    "MINIWORLD_BIAS_ONLY_DIT_INF3_BF16": "bias_only_dit_inf3_bf16",
    "MINIWORLD_BIAS_ONLY_DIT_INF3_CL": "bias_only_dit_inf3_cl",
    "MINIWORLD_BIAS_ONLY_DIT_INF3_P_EARLY": "bias_only_dit_inf3_p_early",
    "MINIWORLD_BIAS_ONLY_DIT_INF3_TAIL_CL": "bias_only_dit_inf3_tail_cl",
    "MINIWORLD_BIAS_ONLY_DIT_PAIR_EXACT": "bias_only_dit_pair_exact",
    "MINIWORLD_BIAS_ONLY_DIT_PV_BATCH": "bias_only_dit_pv_batch",
    "MINIWORLD_BIAS_ONLY_DIT_PV_VB": "bias_only_dit_pv_vb",
    "MINIWORLD_BIAS_ONLY_DIT_RESLN": "bias_only_dit_resln",
    "MINIWORLD_BIAS_ONLY_DIT_SG": "bias_only_dit_sg",
    "MINIWORLD_BIAS_ONLY_DIT_TF32": "bias_only_dit_tf32",
    "MINIWORLD_BIAS_ONLY_DIT_TF32_DPB_PAIR": "bias_only_dit_tf32_dpb_pair",
    "MINIWORLD_BIAS_ONLY_DIT_TF32_GLU": "bias_only_dit_tf32_glu",
    "MINIWORLD_BIAS_ONLY_DIT_TF32_GLU_BWD": "bias_only_dit_tf32_glu_bwd",
    "MINIWORLD_BIAS_ONLY_DIT_TRAIN": "bias_only_dit_train",
    "MINIWORLD_BIAS_ONLY_DIT_TRAIN_PACK1": "bias_only_dit_train_pack1",
    "MINIWORLD_BIAS_ONLY_DIT_WGRAD_SPLIT": "bias_only_dit_wgrad_split",
    "MINIWORLD_ENGINE_JIT_ROOT": "engine_jit_root",
    "MINIWORLD_FUSED_PAIR_INIT": "token_pair_init_fused",
    "MINIWORLD_LOCAL_DIT_HOIST": "local_dit_hoist",
    "MINIWORLD_LOCAL_DIT_SM100": "local_dit_sm100",
    "MINIWORLD_LOCAL_DIT_TF32": "local_dit_tf32",
    "MINIWORLD_LOCAL_DIT_TF32_FUSED": "local_dit_tf32_fused",
    "MINIWORLD_OPM_TRAIN": "opm_train",
    "MINIWORLD_OPM_TRAIN_SAVE_O": "opm_train_save_o",
    "MINIWORLD_PLAN_CACHE_DIR": "plan_cache_dir",
    "MINIWORLD_PWA_INFER": "pwa_infer",
    "MINIWORLD_PWA_TRAIN": "pwa_train",
    "MINIWORLD_ROPE_ANGLES_BF16": "rope_angles_bf16_compat",
    "MINIWORLD_STATIC_CONDITIONING": "static_conditioning",
    "MINIWORLD_STATIC_WEIGHTS": "static_weights",
    "MINIWORLD_SWA_DIT_SM100": "swa_dit_sm100",
    "MINIWORLD_SWA_DIT_TF32": "swa_dit_tf32",
    "MINIWORLD_SWA_FUSED": "swa_fused",
    "MINIWORLD_SWA_GLOBAL_BF16": "swa_global_bf16",
    "MINIWORLD_TF32_FUSED_UNITS": "tf32_fused_units",
    "MINIWORLD_TF32_PDL": "tf32_pdl",
    "MINIWORLD_TOKEN_DIT_GEMM_SWIGLU": "token_dit_gemm_swiglu",
    "MINIWORLD_TOKEN_DIT_ROWS_CUDA": "token_dit_rows_cuda",
    "MINIWORLD_TOKEN_DIT_TRAIN": "token_dit_train",
    "MINIWORLD_TRANSITION_FUSED_SM100A": "transition_fused_sm100a",
    "MINIWORLD_TRANSITION_FUSED_SM80": "transition_fused_sm80",
    "MINIWORLD_TRANSITION_FUSED_SM90A": "transition_fused_sm90a",
    "MINIWORLD_TRANSITION_SM100_NVCC": "transition_sm100_nvcc",
    "MINIWORLD_TRANSITION_SM100_REPL": "transition_sm100_repl",
    "MINIWORLD_TRANSITION_WIDE_REPL": "transition_wide_repl",
    "MINIWORLD_TRANSITION_WIDE_SAVE_H": "transition_wide_save_h",
    "MINIWORLD_TRANSITION_WIDE_SM90A": "transition_wide_sm90a",
    "MINIWORLD_TRANSITION_WIDE_SMALL_FWD_TILES": "transition_wide_small_fwd_tiles",
    "MINIWORLD_TRANSITION_WIDE_SMALL_TILES": "transition_wide_small_tiles",
    "MINIWORLD_TDIT32_WGRAD_SPLIT": "",  # removed: the default behaviour is the only one
    "MINIWORLD_ATOM_RG_TMA": "",  # removed: the default behaviour is the only one
    "MINIWORLD_BIAS_ONLY_DIT_BWD_TAIL_NST": "",  # removed: the default behaviour is the only one
    "MINIWORLD_SWA_DIT_SM100_FUSED_BWD": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_SM80_FLAGS": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_SM80_NREP": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_Q_FWD": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_QG_FWD": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_QKV_FWD": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_TRAINING_FWD": "",  # removed: the default behaviour is the only one
    "MW_WIDE_INFER_CFG": "",  # removed: the default behaviour is the only one
    "TRIMUL_NATIVE_DRIVER": "",  # removed: the default behaviour is the only one
    "TRIMUL_NATIVE_PDL": "",  # removed: the default behaviour is the only one
    "MW_SLDDT_TILE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_BO_ROWS_BLOCKS_PER_SM": "",  # removed: the default behaviour is the only one
    "MINIWORLD_BIAS_ONLY_DIT_F32_RESB_MINB": "",  # removed: the default behaviour is the only one
    "MINIWORLD_A100_CLEANUP": "a100_cleanup",
    "MINIWORLD_ADALN_ATOM": "adaln_atom",
    "MINIWORLD_ADALN_ATOM_CFG": "adaln_atom_cfg",
    "MINIWORLD_ADALN_ATOM_TF32": "adaln_atom_tf32",
    "MINIWORLD_ADALN_BRANCH": "adaln_branch",
    "MINIWORLD_ADALN_FP32_ATOM_MIN_ROWS": "adaln_fp32_atom_min_rows",
    "MINIWORLD_ADALN_SM80": "adaln_sm80",
    "MINIWORLD_ADALN_SM80_ATOM_COND_WGRAD_FUSED": "adaln_sm80_atom_cond_wgrad_fused",
    "MINIWORLD_ADALN_SM80_FLAGS": "adaln_sm80_flags",
    "MINIWORLD_ADALN_SM80_NATIVE_PARAM_CAST": "adaln_sm80_native_param_cast",
    "MINIWORLD_ADALN_SM80_PACK_PARAM_CAST": "adaln_sm80_pack_param_cast",
    "MINIWORLD_ADALN_SM80_SAVE_PARAM_CAST": "adaln_sm80_save_param_cast",
    "MINIWORLD_ADALN_WARP_BLOCKS": "adaln_warp_blocks",
    "MINIWORLD_APB_SM80": "apb_sm80",
    "MINIWORLD_APB_SM80_SAVE_NATURAL": "apb_sm80_save_natural",
    "MINIWORLD_APB_SM80_TRAIN16": "apb_sm80_train16",
    "MINIWORLD_APB_SM80_TYPED_GRAD": "apb_sm80_typed_grad",
    "MINIWORLD_ATOM_DIT_SM80": "atom_dit_sm80",
    "MINIWORLD_ATOM_DIT_SM80_JOIN_THREE_CFG": "atom_dit_sm80_join_three_cfg",
    "MINIWORLD_ATOM_DIT_SM80_JOIN_TWO_CFG": "atom_dit_sm80_join_two_cfg",
    "MINIWORLD_AUGATTN_SM80": "augattn_sm80",
    "MINIWORLD_AUGATTN_SM80_ATOM_OUTPUT_FUSED": "augattn_sm80_atom_output_fused",
    "MINIWORLD_AUGATTN_SM80_ATOM_OUTPUT_TILED": "augattn_sm80_atom_output_tiled",
    "MINIWORLD_AUGATTN_SM80_ATOM_QKVG": "augattn_sm80_atom_qkvg",
    "MINIWORLD_AUGATTN_SM80_ATOM_QKVG_TILED": "augattn_sm80_atom_qkvg_tiled",
    "MINIWORLD_AUGATTN_SM80_FLAGS": "augattn_sm80_flags",
    "MINIWORLD_AUGATTN_SM80_FUSED": "augattn_sm80_fused",
    "MINIWORLD_AUGATTN_SM80_OPS": "augattn_sm80_ops",
    "MINIWORLD_AUGATTN_SM80_PBGEN": "augattn_sm80_pbgen",
    "MINIWORLD_AUGATTN_SM80_PBGEN_ROWS": "augattn_sm80_pbgen_rows",
    "MINIWORLD_AUGATTN_SM80_SHARED_CORE": "augattn_sm80_shared_core",
    "MINIWORLD_AUGATTN_SM80_SHARED_SCORE": "augattn_sm80_shared_score",
    "MINIWORLD_AUGATTN_SM80_SHARED_UNIFORM": "augattn_sm80_shared_uniform",
    "MINIWORLD_BIAS_ONLY_ATTN_SM80": "bias_only_attn_sm80",
    "MINIWORLD_BIAS_ONLY_DIT_PREP": "bias_only_dit_prep",
    "MINIWORLD_BIAS_ONLY_DIT_SM80_DPB": "bias_only_dit_sm80_dpb",
    "MINIWORLD_BIAS_ONLY_DIT_SM80_SG": "bias_only_dit_sm80_sg",
    "MINIWORLD_BIAS_ONLY_DIT_SM80_STAGES": "bias_only_dit_sm80_stages",
    "MINIWORLD_CONDTRANS_ATOM": "condtrans_atom",
    "MINIWORLD_CONDTRANS_ATOM_TF32": "condtrans_atom_tf32",
    "MINIWORLD_CONDTRANS_SM80": "condtrans_sm80",
    "MINIWORLD_CONDTRANS_SM80_ATOM_COND_WGRAD_FUSED": "condtrans_sm80_atom_cond_wgrad_fused",
    "MINIWORLD_CONDTRANS_SM80_ATOM_NARROW": "condtrans_sm80_atom_narrow",
    "MINIWORLD_CONDTRANS_SM80_ATOM_SMALL": "condtrans_sm80_atom_small",
    "MINIWORLD_CONDTRANS_SM80_ATOM_SPLIT": "condtrans_sm80_atom_split",
    "MINIWORLD_CONDTRANS_SM80_ATOM_SPLIT_CFG": "condtrans_sm80_atom_split_cfg",
    "MINIWORLD_CONDTRANS_SM80_ATOM_WGRAD_SPLIT": "condtrans_sm80_atom_wgrad_split",
    "MINIWORLD_CONDTRANS_SM80_ATOM_WGRAD_WIDE": "condtrans_sm80_atom_wgrad_wide",
    "MINIWORLD_CONDTRANS_SM80_FLAGS": "condtrans_sm80_flags",
    "MINIWORLD_CONDTRANS_SM80_NATIVE_PARAM_CAST": "condtrans_sm80_native_param_cast",
    "MINIWORLD_CONDTRANS_SM80_PACK_PARAM_CAST": "condtrans_sm80_pack_param_cast",
    "MINIWORLD_CONDTRANS_SM80_SAVE_PARAM_CAST": "condtrans_sm80_save_param_cast",
    "MINIWORLD_DENSE_CORE_DKV_ORDER": "dense_core_dkv_order",
    "MINIWORLD_DENSE_CORE_DQ_ORDER": "dense_core_dq_order",
    "MINIWORLD_DENSE_E274_DKV_ORDER": "dense_e274_dkv_order",
    "MINIWORLD_DENSE_E274_DQ_ORDER": "dense_e274_dq_order",
    "MINIWORLD_DROPOUT_SCALE_SM80": "dropout_scale_sm80",
    "MINIWORLD_GATED_SM80": "gated_sm80",
    "MINIWORLD_LNLINEAR_SM80": "lnlinear_sm80",
    "MINIWORLD_LOCAL_DIT_SM80": "local_dit_sm80",
    "MINIWORLD_MPNN_EDGE_FLAGS": "mpnn_edge_flags",
    "MINIWORLD_MPNN_EDGE_PTXAS": "mpnn_edge_ptxas",
    "MINIWORLD_MPNN_EDGE_SM80": "mpnn_edge_sm80",
    "MINIWORLD_MPNN_MSG_SM80": "mpnn_msg_sm80",
    "MINIWORLD_MPNN_MSG_SM80_BWD": "mpnn_msg_sm80_bwd",
    "MINIWORLD_MPNN_MSG_SM80_FLAGS": "mpnn_msg_sm80_flags",
    "MINIWORLD_MPNN_NODE_SM80_FLAGS": "mpnn_node_sm80_flags",
    "MINIWORLD_NORMS_SM80": "norms_sm80",
    "MINIWORLD_NORMS_SM80_FLAGS": "norms_sm80_flags",
    "MINIWORLD_OPM_SM80": "opm_sm80",
    "MINIWORLD_OPM_SM80_DG": "opm_sm80_dg",
    "MINIWORLD_OPM_SM80_EPI": "opm_sm80_epi",
    "MINIWORLD_OPM_SM80_MASK_COUNT_NATIVE": "opm_sm80_mask_count_native",
    "MINIWORLD_OPM_SM80_MASK_EPILOGUE": "opm_sm80_mask_epilogue",
    "MINIWORLD_OPM_SM80_NATIVE128_PIPELINE": "opm_sm80_native128_pipeline",
    "MINIWORLD_OPM_SM80_PB": "opm_sm80_pb",
    "MINIWORLD_OPM_SM80_PIPELINE": "opm_sm80_pipeline",
    "MINIWORLD_OPM_SM80_PL": "opm_sm80_pl",
    "MINIWORLD_OPM_SM80_SAVE_O": "opm_sm80_save_o",
    "MINIWORLD_OPM_SM80_TYPED_GRAD": "opm_sm80_typed_grad",
    "MINIWORLD_PWA_SM80": "pwa_sm80",
    "MINIWORLD_PWA_SM80_BROADCAST_NATIVE": "pwa_sm80_broadcast_native",
    "MINIWORLD_PWA_SM80_BROADCAST_PIPELINE": "pwa_sm80_broadcast_pipeline",
    "MINIWORLD_PWA_SM80_CONTRACT_GATE_FUSED": "pwa_sm80_contract_gate_fused",
    "MINIWORLD_PWA_SM80_DW_NATIVE": "pwa_sm80_dw_native",
    "MINIWORLD_PWA_SM80_DW_SHORT": "pwa_sm80_dw_short",
    "MINIWORLD_PWA_SM80_GLUE_TUNED": "pwa_sm80_glue_tuned",
    "MINIWORLD_PWA_SM80_GRAD_REDUCE": "pwa_sm80_grad_reduce",
    "MINIWORLD_PWA_SM80_PAIR_PARTIAL": "pwa_sm80_pair_partial",
    "MINIWORLD_PWA_SM80_PB": "pwa_sm80_pb",
    "MINIWORLD_PWA_SM80_SPLIT": "pwa_sm80_split",
    "MINIWORLD_PWA_SM80_TYPED_GRAD": "pwa_sm80_typed_grad",
    "MINIWORLD_SIGMOID_GATE_SM80": "sigmoid_gate_sm80",
    "MINIWORLD_SWA_DIT_SM80": "swa_dit_sm80",
    "MINIWORLD_SWA_DIT_SM80_FFN_TUNED": "swa_dit_sm80_ffn_tuned",
    "MINIWORLD_SWA_DIT_SM80_FLAGS": "swa_dit_sm80_flags",
    "MINIWORLD_SWA_DIT_SM80_INFER_TUNED": "swa_dit_sm80_infer_tuned",
    "MINIWORLD_SWA_DIT_SM80_NATIVE_DWD": "swa_dit_sm80_native_dwd",
    "MINIWORLD_SWA_DIT_SM80_NATIVE_GRAD_CAST": "swa_dit_sm80_native_grad_cast",
    "MINIWORLD_SWA_DIT_SM80_NATIVE_GROUPED5_DW": "swa_dit_sm80_native_grouped5_dw",
    "MINIWORLD_SWA_DIT_SM80_NATIVE_GROUPED8_ALLOC": "swa_dit_sm80_native_grouped8_alloc",
    "MINIWORLD_SWA_DIT_SM80_NATIVE_GROUPED8_DW": "swa_dit_sm80_native_grouped8_dw",
    "MINIWORLD_SWA_DIT_SM80_NATIVE_GROUPED_DW": "swa_dit_sm80_native_grouped_dw",
    "MINIWORLD_SWA_DIT_SM80_NATIVE_QKVG_DW": "swa_dit_sm80_native_qkvg_dw",
    "MINIWORLD_TOKEN_DIT_PREP": "token_dit_prep",
    "MINIWORLD_TOKEN_DIT_SM80_ALIAS_TRAIN_PACK": "token_dit_sm80_alias_train_pack",
    "MINIWORLD_TOKEN_DIT_SM80_DIRECT_GRAD": "token_dit_sm80_direct_grad",
    "MINIWORLD_TOKEN_DIT_SM80_NATIVE_TRAIN_PACK": "token_dit_sm80_native_train_pack",
    "MINIWORLD_TOKEN_DIT_TYPED_GRAD": "token_dit_typed_grad",
    "MINIWORLD_TRANSITION_BWD_SM80_FLAGS": "transition_bwd_sm80_flags",
    "MINIWORLD_TRANSITION_BWD_SM80_NREP": "transition_bwd_sm80_nrep",
    "MINIWORLD_TRANSITION_FUSED_BWD_SM80": "transition_fused_bwd_sm80",
    "MINIWORLD_TRANSITION_FUSED_FWD_SM80": "transition_fused_fwd_sm80",
    "MINIWORLD_TRANSITION_FWD_SM80_FLAGS": "transition_fwd_sm80_flags",
    "MINIWORLD_TRANSITION_WIDE_GRAD_CAST": "transition_wide_grad_cast",
    "MINIWORLD_TRANSITION_WIDE_GRAD_FP32": "transition_wide_grad_fp32",
    "MINIWORLD_TRANSITION_WIDE_SM80": "transition_wide_sm80",
    "MINIWORLD_TRANSITION_WIDE_SM80_FLAGS": "transition_wide_sm80_flags",
    "MINIWORLD_TRIATTN_SM80": "triattn_sm80",
    "MINIWORLD_TRIATTN_SM80_COMPACT": "triattn_sm80_compact",
    "MINIWORLD_TRIATTN_SM80_DQ_ROWS": "triattn_sm80_dq_rows",
    "MINIWORLD_TRIATTN_SM80_EXACT_GAMMA": "triattn_sm80_exact_gamma",
    "MINIWORLD_TRIATTN_SM80_FLAGS": "triattn_sm80_flags",
    "MINIWORLD_TRIATTN_SM80_GEMM_PAD": "triattn_sm80_gemm_pad",
    "MINIWORLD_TRIATTN_SM80_INTERNAL": "triattn_sm80_internal",
    "MINIWORLD_TRIATTN_SM80_NATIVE_DB": "triattn_sm80_native_db",
    "MINIWORLD_TRIATTN_SM80_NATIVE_OUT": "triattn_sm80_native_out",
    "MINIWORLD_TRIATTN_SM80_NATIVE_RNG": "triattn_sm80_native_rng",
    "MINIWORLD_TRIATTN_SM80_NATIVE_WO": "triattn_sm80_native_wo",
    "MINIWORLD_TRIATTN_SM80_ROWS": "triattn_sm80_rows",
    "MINIWORLD_TRIATTN_SM80_SOFTMAX": "triattn_sm80_softmax",
    "MINIWORLD_TRIATTN_SM80_TRAIN_CAPTURE_AWARE": "triattn_sm80_train_capture_aware",
    "MINIWORLD_TRIATTN_SM80_TRAIN_INNER": "triattn_sm80_train_inner",
    "MINIWORLD_TRIATTN_SM80_TUNED": "triattn_sm80_tuned",
    "MINIWORLD_TRIATTN_SM80_WIDE": "triattn_sm80_wide",
    "MINIWORLD_TRIMUL_SM80": "trimul_sm80",
    "MINIWORLD_TRIMUL_SM80_FLAGS": "trimul_sm80_flags",
    "MINIWORLD_TRIMUL_SM80_ROW_MASK": "trimul_sm80_row_mask",
    "MINIWORLD_TRIMUL_SM80_TUNED": "trimul_sm80_tuned",
    "MINIWORLD_TRIMUL_SM80_WIDE": "trimul_sm80_wide",
    "MINIWORLD_TRIMUL_SM80_WIDE512": "trimul_sm80_wide512",
    "MINIWORLD_TRIMUL_SM80_WIDE_FLAGS": "trimul_sm80_wide_flags",
}


def _warn_about_removed_environment() -> None:
    """Warn once about any variable of ``_REMOVED_ENVIRONMENT`` that is set: its value no longer has any effect."""
    present = sorted(name for name in _REMOVED_ENVIRONMENT if name in os.environ)
    if not present:
        return
    detail = ", ".join(
        f"{n} -> settings.{_REMOVED_ENVIRONMENT[n]}"
        if _REMOVED_ENVIRONMENT[n]
        else f"{n} (removed)"
        for n in present
    )
    warnings.warn(
        "miniworld_engine no longer reads these environment variables; use miniworld_engine.settings.configure(...) "
        f"(MiniWorld: train.engine_settings): {detail}",
        UserWarning,
        stacklevel=2,
    )


_warn_about_removed_environment()

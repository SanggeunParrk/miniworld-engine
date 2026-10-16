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
    #: Serve the bf16 window attention -- forward, and the backward's dq / dk / dv as ONE kernel -- with the hand-CUDA sm_90a
    #: kernels (`cuda/swa_attn.cu`); off (or any other card, or a failed build) takes `_swa_attn_fwd_kernel` and
    #: `_swa_attn_bwd_{dq,dkv}_kernel`.
    swa_dit_attn_cuda: bool = True
    #: Same for the bf16 out-projection backward (`cuda/swa_slab.cu`; else `_swa_oproj_bwd_kernel`).
    swa_dit_oproj_bwd_cuda: bool = True
    #: Same for the bf16 qkvg backward (`cuda/swa_slab.cu`; else `_swa_qkvg_bwd_kernel`).
    swa_dit_qkvg_bwd_cuda: bool = True
    #: Which hand-CUDA kernels serve the stages that have two (the qkvg forward, the out-projection + FFN forward, the FFN
    #: backward): "slab" (`cuda/swa_slab.cu`, 16-row slabs per 4-warp group, mma.sync -- built for the small row counts the
    #: block mostly runs at), "wgmma" (team-gm's 64-row wgmma kernels), or "auto" (by row count, see
    #: `kernels/swa_dit/dispatch.py:_SLAB_MAX_ROWS`).
    swa_dit_rows_impl: Literal["auto", "slab", "wgmma"] = "auto"
    #: FFN weight gradients: "mat" materialises [da | db] and h and takes dW from cuBLAS; "fused" recomputes them in
    #: the `_swa_ffn_dw_kernel` Triton kernel (correct, slower in Triton: 255 registers). Formerly SWA_FFN_DW.
    swa_dit_ffn_dw: Literal["mat", "fused"] = "mat"
    #: dtype of dq1, the gradient handed from the FFN half of the backward to the attention half, on the Triton FFN
    #: backward (the hand-CUDA one writes bf16). Formerly SWA_DQ1.
    swa_dit_dq1: Literal["bf16", "fp32"] = "bf16"
    #: Serve the fp32 block (q fp32) with the hand-CUDA sm_90a kernels of `kernels/swa_dit/cuda/swa_fp32.cu` on sm_90:
    #: every stage -- qkvg, window attention, out-projection + FFN and their backwards -- on tf32 wgmma with
    #: rna-rounded operands (cuBLAS's TF32; the FFN weights split in two tf32 halves) and a bf16-operand window
    #: attention (FlashAttention-4's). Only while torch's fp32 matmul precision allows TF32 ("high" / "medium"); under
    #: "highest", off, on any other card, or on a failed build the fp32 Triton kernels serve. The weight gradients are
    #: torch GEMMs either way.
    swa_dit_fp32_cuda: bool = True
    #: Serve the hoisted adaLN modulation (`swa_dit_hoist_modulation`: silu(c) Wmod^T and its backward) with the hand-CUDA
    #: sm_90a kernels of `kernels/swa_dit/cuda/swa_mod.cu` on sm_90, bf16 (any matmul precision) or fp32 (TF32 allowed). Off
    #: or on any other card, the fp32 GEMMs of torch serve.
    swa_dit_mod_cuda: bool = True
    #: The four weight-gradient GEMMs of the bf16 block backward (dWqkv | dWg, dWo, dWu, dWd) as ONE hand-CUDA launch
    #: (`kernels/swa_dit/cuda/swa_wgrad.cu`, clusters of 8 CTAs splitting K, summed in rank order) instead of four cuBLAS GEMMs.
    #: sm_90, bf16, on the all-CUDA backward only.
    swa_dit_wgrad_cuda: bool = True

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
    #: The generic sm_90a Transition kernels (any width multiple of 64, any hidden multiple of 128, ragged rows, fp32 on TF32);
    #: off leaves only the tuned fused / wide builds. Formerly ``MINIWORLD_TRANSITION_GENERIC_SM90A`` (default on).
    transition_generic_sm90a: bool = True
    #: Prefer the generic kernels where they beat a tuned build at small row counts. Formerly ``MINIWORLD_TRANSITION_GENERIC_SMALL``.
    transition_generic_small: bool = True
    #: Training bf16 forwards of the generic path keep a | b for the backward. Formerly ``MINIWORLD_TRANSITION_GENERIC_SAVE_AB``.
    transition_generic_save_ab: bool = True
    #: Force the cuBLAS parts of the generic bf16 path: comma separated ``f3`` / ``gate`` / ``dxn``, ``all`` or ``none``; None keeps
    #: the measured row-count plan. Formerly ``MINIWORLD_TRANSITION_GENERIC_CUBLAS``.
    transition_generic_cublas: str | None = None
    #: Fuse the input LN into the expand GEMM (measured 5-25 % slower: default off). Formerly ``MINIWORLD_TRANSITION_GENERIC_LNF``.
    transition_generic_lnf: bool = False
    #: Overlap the weight gradients with the dxn -> ln_bwd chain at the row counts where it pays. Formerly
    #: ``MINIWORLD_TRANSITION_GENERIC_DW_ASIDE`` (default on).
    transition_generic_dw_aside: bool = True
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
    # ---- H100 native-path switches (formerly MINIWORLD_* environment variables; name = the variable without its prefix, lower case)
    #: Formerly ``MINIWORLD_APB_H100_TRAIN`` (default on): the native H100 CUDA AttentionPairBias training path
    #: (`integrations/attention_pair_bias_b200.py`). Off keeps the legacy comparison path (explicit baseline only).
    apb_h100_train: bool = True
    #: H100 AttentionPairBias training: True recomputes the pair-weight (``to_bias.weight``) gradient in fp32 from bf16-rounded
    #: inputs (the "shadow": 1.45e-2 against an fp64 truth, about 2.4x the step); False (default) keeps the fused kernel's dWf
    #: (bf16 operands, fp32 accumulation: 1.9e-2, the same error as the bf16 PyTorch module, the Triton path and every other
    #: gradient of the block).
    apb_h100_exact_dwb: bool = False
    #: Formerly ``MINIWORLD_AUGATTN_TF32_SM90`` (default on): serve the fp32 token-DiT inference core with the hand-CUDA TF32 kernel
    #: (`attn_tf32.cu`) on sm_90; off keeps the Triton core.
    augattn_tf32_sm90: bool = True
    #: Formerly ``MINIWORLD_BIAS_ONLY_DIT_H100`` (default on): the native Hopper bias-only DiT kernels
    #: (`kernels/bias_only_dit/cuda`); off takes the sm_100 / composed route.
    bias_only_dit_h100: bool = True
    #: Formerly ``MINIWORLD_LOCAL_DIT_H100_TRAIN``: None serves the Hopper local atom DiT training block natively when the
    #: call is inside the native geometry and lets the composed path serve everything else; True raises instead of falling back;
    #: False disables the native Hopper training path.
    local_dit_h100_train: bool | None = None
    #: The H100 local atom DiT on warp-level tensor-core kernels (``integrations/local_dit_h100``): fp32 inference (TF32). Off keeps the
    #: previous routes.
    local_dit_h100_mma: bool = True
    local_dit_h100_mma_fp32: bool = True
    local_dit_h100_mma_bf16: bool = True
    #: programmatic dependent launch between the kernels of the H100 local atom DiT (the next kernel's launch overlaps the previous one's tail)
    local_dit_h100_pdl: bool = True
    #: Formerly ``MINIWORLD_SWA_H100_CUDA_ONLY``: None runs the native Hopper SWA DiT block for the shapes it is qualified for and
    #: falls back elsewhere; True forces it (an unsupported call raises); False disables it.
    swa_dit_h100_native: bool | None = None
    #: Formerly ``MINIWORLD_TOKEN_DIT_GEMM_SWIGLU_FP32`` (default off): allow the TF32 build of the fused GEMM + SwiGLU for fp32
    #: activations. Its TMA-fed operands are truncated to TF32 where cuBLAS rounds them (1.6e-3 against 4.2e-4); wins only at M = 3840.
    token_dit_gemm_swiglu_fp32: bool = False
    #: Formerly ``MINIWORLD_TRANSITION_D64_REPL``; 0 keeps the built-in value (26 at n = 2, 16 at n = 4): D64 backward weight replicas.
    transition_d64_repl: int = 0
    #: Formerly ``MINIWORLD_TRANSITION_D64_VARIANT``; 0 picks the D64 inference build by row count (13 / 12 = 64-row tile, LayerNorm in
    #: the wgmma A fragment, 3 / 2 CTAs per SM; 2 / 3 / 14 / 15 earlier builds, forced only).
    transition_d64_variant: int = 0
    #: Formerly ``MINIWORLD_TRANSITION_DW_REPL``; 0 keeps the built-in value (8 at n = 4, 15 at n = 2): D128 backward weight replicas.
    transition_dw_repl: int = 0
    #: Formerly ``MINIWORLD_TRIMUL_H100_B200STYLE`` (default off): serve TriMul on Hopper with the B200-style port
    #: (`h100_b200style*`, widths 256/384/512 plus the D64 b200gate kernels). Experimental; becomes a B item (bake in or delete)
    #: when the family is qualified. On since 2026-10-13: it is 1.2-2.1x faster than the general path at every length the preserved
    #: native path does not serve, and the cells where that path is faster keep it (`integrations/trimul_h100.py`).
    trimul_h100_b200style: bool = True
    #: Formerly ``MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS``: None = only the qualified native cells; True also serves the unqualified
    #: lengths 128..1024 and the preserved wide glue for the widths in ``trimul_h100_preserve_widths``; False disables every
    #: native TriMul training cell (formerly NATIVE_BUCKETS=0).
    #: Run the H100 TriMul contractions through cuBLAS with a zero workspace (no semaphore memset before each GEMM, same kernels,
    #: bitwise equal results; ``kernels/trimul_inproj/cuda/_h100_bmm.py``). Off falls back to ``torch.bmm``.
    trimul_h100_zero_workspace_bmm: bool = True
    trimul_h100_native_buckets: bool | None = None
    #: Formerly ``MINIWORLD_TRIMUL_H100_PRESERVE_D128`` / ``_PRESERVE_D256`` / ``_PRESERVE_WIDE`` (widths 384 and 512 together):
    #: pair widths whose preserved native glue is selected explicitly; takes effect with ``trimul_h100_native_buckets=True``.
    trimul_h100_preserve_widths: frozenset[int] = frozenset()

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
    "MINIWORLD_APB_ACC_TREE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_APB_H100_BETA_SHADOW": "",  # removed: the default behaviour is the only one
    "MINIWORLD_APB_H100_EXACT_DWB": "",  # removed: the default behaviour is the only one
    "MINIWORLD_APB_H100_OWNED_SGEMM": "",  # removed: the default behaviour is the only one
    "MINIWORLD_APB_H100_SHADOW_DB_ONLY": "",  # removed: the default behaviour is the only one
    "MINIWORLD_APB_H100_SHADOW_IO": "",  # removed: the default behaviour is the only one
    "MINIWORLD_APB_H100_SHADOW_PAIR_LN": "",  # removed: the default behaviour is the only one
    "MINIWORLD_APB_H100_SHADOW_PAIR_PROJECT": "",  # removed: the default behaviour is the only one
    "MINIWORLD_APB_H100_TRAIN": "apb_h100_train",
    "MINIWORLD_APB_PYTORCH_NULL_BETA": "",  # removed: the default behaviour is the only one
    "MINIWORLD_ATOM_H100_COND_WAVES": "",  # removed: the default behaviour is the only one
    "MINIWORLD_ATOM_H100_POST_PACKED": "",  # removed: the default behaviour is the only one
    "MINIWORLD_AUGATTN_TF32_SM90": "augattn_tf32_sm90",
    "MINIWORLD_BIAS_ONLY_DIT_H100": "bias_only_dit_h100",
    "MINIWORLD_BIAS_ONLY_DIT_H100_FP32_LOGITS": "",  # removed: the default behaviour is the only one
    "MINIWORLD_BIAS_ONLY_DIT_H100_INFER_ASYNC": "",  # removed: the default behaviour is the only one
    "MINIWORLD_BIAS_ONLY_DIT_H100_PIPELINE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_BIAS_ONLY_DIT_H100_PV_GROUP": "",  # removed: the default behaviour is the only one
    "MINIWORLD_BIAS_ONLY_DIT_H100_SHADOW_GAMMA": "",  # removed: the default behaviour is the only one
    "MINIWORLD_DIT_H100_CALLER_NATIVE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_LOCAL_DIT_H100": "",  # removed: the default behaviour is the only one
    "MINIWORLD_LOCAL_DIT_H100_OWNED_BLAS": "",  # removed: the default behaviour is the only one
    "MINIWORLD_LOCAL_DIT_H100_TRAIN": "local_dit_h100_train",
    "MINIWORLD_OPM_H100_CUBLAS_COUNT": "",  # removed: the default behaviour is the only one
    "MINIWORLD_OPM_H100_CUDA_ONLY": "",  # removed: the default behaviour is the only one
    "MINIWORLD_OPM_H100_EPILOGUE_HALF": "",  # removed: the default behaviour is the only one
    "MINIWORLD_OPM_H100_EPILOGUE_TMA": "",  # removed: the default behaviour is the only one
    "MINIWORLD_OPM_H100_EPILOGUE_WIDE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_OPM_H100_OWNED_BLAS": "",  # removed: the default behaviour is the only one
    "MINIWORLD_OPM_H100_TMA_PROLOGUE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_PWA_H100_CUDA_AUX": "",  # removed: the default behaviour is the only one
    "MINIWORLD_SWA_H100_ATTN_STAGES": "",  # removed: the default behaviour is the only one
    "MINIWORLD_SWA_H100_ATTN_WAVES": "",  # removed: the default behaviour is the only one
    "MINIWORLD_SWA_H100_B200_FUSION": "",  # removed: the default behaviour is the only one
    "MINIWORLD_SWA_H100_CUDA_ONLY": "swa_dit_h100_native",
    "MINIWORLD_SWA_H100_FUSED_OPROJ_BWD": "",  # removed: the default behaviour is the only one
    "MINIWORLD_SWA_H100_OWNED_LT_WGRAD": "",  # removed: the default behaviour is the only one
    "MINIWORLD_SWA_H100_REUSE_FFN_BWD": "",  # removed: the default behaviour is the only one
    "MINIWORLD_SWA_H100_TILED_ATTENTION": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TOKEN_DIT_GEMM_SWIGLU_FP32": "token_dit_gemm_swiglu_fp32",
    "MINIWORLD_TOKEN_DIT_H100_CALLER_NATIVE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TOKEN_DIT_H100_EMPTY_UNIFORM": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TOKEN_DIT_H100_GRAD_REDUCE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TOKEN_DIT_H100_INFER_ASYNC": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TOKEN_DIT_H100_OWNED_GEMMS": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TOKEN_DIT_H100_PACK_VECTOR": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TOKEN_DIT_H100_TRAIN_NATIVE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TOKEN_DIT_H100_TRAIN_PACK_ASYNC": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TOKEN_DIT_H100_TRAIN_RPB": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TOKEN_DIT_TRAIN_FP32": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_D64_REPL": "transition_d64_repl",
    "MINIWORLD_TRANSITION_D64_VARIANT": "transition_d64_variant",
    "MINIWORLD_TRANSITION_DW_REPL": "transition_dw_repl",
    "MINIWORLD_TRANSITION_H100_AFFINE_PARALLEL": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_AFFINE_WARP_STORE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_COMPACT_LN": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_COMPACT_LN_D384": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_CUDA_AUX": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_DW_EARLY_RELEASE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_DW_OVERLAP": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_DXLN_FULL_ROW": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_FUSED_D384": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_FUSED_D384_MC": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_GATE_FULL_STAGE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_GATE_K64": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_GATE_MULTICAST": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_OWNED_MM": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_PREPARE_FUSED": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_SAVED_AB_NO_H_STAGE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_SAVED_AB_TMA": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_SAVE_AB": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_SPLIT_DX": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRANSITION_H100_SPLIT_FWD": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_CANDIDATE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_FRONTEND": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_KV_TILE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_NATIVE_RNG": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_OWNED_BLAS": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_PIPELINE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_PROJECT_LN": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_ROW_GROUP": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_ROW_VECTOR": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_SPLIT_WGRAD": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIATTN_H100_TAIL": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_B200STYLE": "trimul_h100_b200style",
    "MINIWORLD_TRIMUL_H100_BIDI512_INFER_CANDIDATE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_CUDA_AUX": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_D128_GRAPH_WORKSPACE_CANDIDATE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_D256_REG_SOURCE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_D64_GRAPH_WORKSPACE_CANDIDATE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_D64_OWNED_BMM": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_D64_UNI_AFFINE_SPLIT": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_FAST_GLUE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_FUSED_GLUE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_INFER_FAST_AUX": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_INFER_OWNED_BLAS": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_K3W_NATIVE_RESIDUAL": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS": "trimul_h100_native_buckets",
    "MINIWORLD_TRIMUL_H100_PRESERVE_D128": "trimul_h100_preserve_widths",
    "MINIWORLD_TRIMUL_H100_PRESERVE_D256": "trimul_h100_preserve_widths",
    "MINIWORLD_TRIMUL_H100_PRESERVE_WIDE": "trimul_h100_preserve_widths",
    "MINIWORLD_TRIMUL_H100_TRAIN_CUDA": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_TRAIN_FUSED_PREP": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_WARP_LN": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_WIDE_DEFAULT_CANDIDATE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_WIDE_FUSED_GLUE": "",  # removed: the default behaviour is the only one
    "MINIWORLD_TRIMUL_H100_WIDE_OWNED_BLAS": "",  # removed: the default behaviour is the only one
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

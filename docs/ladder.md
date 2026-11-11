# The performance ladder

For every requirement case that more than one engine backend serves, the engine's backends must be ordered by speed:

    cuda  <  triton  <  pytorch          (median time of the same case, lower is faster)

`cuda` is the hand-written path for the card (H100: sm_90a, A100: sm_80), `triton` is the portable rung, `pytorch` is the reference. A rung that
is slower than the rung below it on some case is a ladder violation: either the faster-looking rung is mis-routed for that case
or the kernel has a performance bug. A violation is never resolved by dropping a rung or a case from the requirements.

## How it is checked

- Per card: the check runs on one card's tokens; the cards are not compared with each other.
- Data: the speed tokens of the card (`src/miniworld_engine/judge/tokens/<gpu>/*.jsonl`, `kind: speed`, `timing: graph`,
  `status: ok`), i.e. the measurer's CUDA-graph timing with the L2 cache flushed before every call (docs/measure.md).
- Pairs: for one case (`case_key` without the backend), compare the medians of `cuda` vs `triton`, `triton` vs `pytorch` and
  `cuda` vs `pytorch`. A pair counts as a violation when the faster rung is more than 2 % slower (`slow > fast * 1.02`); the
  margin absorbs run-to-run noise of graph timing (≈ 0.5 %) and clock drift between jobs.
- Only cases both rungs serve are compared; a case one rung refuses (`unsupported`) or skips is not a violation.
- External backends (`anthropic`, `cueq`) are baselines, not rungs: they are compared in reports, never part of the ladder.
- Eager timings (`timed_as: eager`, a capture fallback) are not comparable with graph timings and are left out.

## Exceptions

| class | why |
|---|---|
| `LayerNorm`, `RMSNorm` | user decision (2026-11-10): a single normalisation is memory-bound and runs at a few microseconds, where the rungs differ by launch and scheduling effects rather than by kernel quality; they are measured and reported, not held to the ladder. |
| small L | user decision (2026-11-10): a miss that occurs only at L <= 256 (the shortest lengths, where a step is tens of microseconds and launch / scheduling effects dominate) is accepted; every miss at L >= 384 must be fixed. |
| `ParallelResidual` (`cuda` vs `triton`) | user decision (2026-11-11): one elementwise pass at the HBM bandwidth wall (triton ~3.1 TB/s of 3.35 on H100); the hand-CUDA backend exists for the H100-in-CUDA rule and may be slower than `triton` (eval, L >= 384: 1-5 %). It must still beat `pytorch`; train (its backward kernel) is faster than `triton`. |

Classes with only a `pytorch` backend have no ladder; `BidirectionalTriangleAttention` has `cuda` and `pytorch` only (cuda < pytorch).

## Status

The status is recomputed from the tokens after every measurement cycle (only the cases the requirements ask for: 2992 speed cases per
card with graph timings). Tokens at b1ffb4a16 (2026-11-11), both cards with 0 owed cases:

| card | ladder | cuda vs the external baselines (target, not a ladder rule) |
|---|---|---|
| H100 | no violation outside the exceptions; the small-L misses are cuda vs triton of AttentionPairBias (1 case), OuterProductMean (3), PairEmbedCA (2), at most 3.7 % | only LayerNorm misses (exception): 8 cases vs `anthropic` |
| A100 | no violation outside the exceptions; the small-L misses are cuda vs triton of OuterProductMean (1 case) and PairEmbedCA (3), at most 6.1 % | LayerNorm misses (exception): 30 cases vs `anthropic`, 8 vs `cueq`; one Transition case at L <= 256 vs `anthropic` (17.5 %) |

Fixed (2026-11-12; was open 2026-11-11): A100 `AttentionPairBias` `cuda`, float32 training with a wide pair (d_pair 256 / 384 / 512) was
0.98-1.12x triton at L 384-640 (S = 5; ~1.0 at S = 1 for d_pair 512). Profile: the pair side of the backward was 70-80 % of the step (per row: the
one-pass CUDA-core kernel at 0.35 TB/s, or at L <= 384 a LayerNorm recompute + cuBLAS GEMMs + the LayerNorm backward + a copy of the pair-sized
gradient). It is now one TF32 tensor-core launch for the batch (`pair_bias_gen_bwd_tc`: one read of the pair, one write of its gradient, ~1.45 TB/s;
the pair gradient in 3xTF32); graph timing, A100 80GB PCIe: cuda / triton 0.43-0.51 at S = 1 and 0.40-0.51 at S = 5 for L 384-768 (r42 / r45 / r47;
before: 0.86-1.01 and 0.86-1.12), 0.29-0.38 at L 1024-2048 (r42 / r45; at L 2048 and S = 5 pytorch and triton do not fit 80 GB). The rest of the step stays per batch row (a row's
results equal the one-row call's bitwise); at L 128 and S = 5 cuda is 1.09-1.24x triton (before 1.69-1.83; the per-row launches; L <= 256 is the small-L
exception), at L 256 and S = 5 0.61-0.67 (before 1.21-1.28). The earlier batched GEMM attempt (`req-apb-batch`, never measured) was not used: it kept 3-4 passes over the pair and
recovered dWf from a z - a mean (cancellation for a pair with a large mean).

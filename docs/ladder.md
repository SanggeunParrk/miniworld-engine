# The performance ladder

For every requirement case that more than one engine backend serves, the engine's backends must be ordered by speed:

    cuda  <  triton  <  pytorch          (median time of the same case, lower is faster)

`cuda` is the hand-written path for the card (H100: sm90), `triton` is the portable rung, `pytorch` is the reference. A rung that
is slower than the rung below it on some case is a ladder violation: either the faster-looking rung is mis-routed for that case
or the kernel has a performance bug. A violation is never resolved by dropping a rung or a case from the requirements.

## How it is checked

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

Classes with only a `pytorch` backend (e.g. `TriangleMultiplicationRF3`, `BidirectionalTriangleAttention`) have no ladder.

## Status

The status is recomputed from the tokens after every measurement cycle (not stored in this page). The H100 check of 2026-11-10
(tokens at main 57c6dc26d) found the pairformer-side classes (`TriangleAttention*`, `TriangleMultiplication`,
`BidirectionalTriangleMultiplication`, `MSAPairWeightedAveraging`, `PairformerBlock`) with no violation, and violations in
`Transition` and `OuterProductMean` (mostly `triton` vs `pytorch`), `TokenDiT*` (the `dit_variants` `cuda` path slower than
`pytorch` on about half the Chai-1 / Boltz-2 / RoseTTAFold3 cases; `triton` vs `pytorch` on Protenix), `PairEmbed*` (`cuda`
slower than `triton`) and `AttentionPairBias` (`triton` vs `pytorch`).

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
| small L | user decision (2026-11-10): a miss that occurs only at L <= 256 (the shortest lengths, where a step is tens of microseconds and launch / scheduling effects dominate) is accepted; every miss at L >= 384 must be fixed. |

Classes with only a `pytorch` backend have no ladder; `BidirectionalTriangleAttention` has `cuda` and `pytorch` only (cuda < pytorch).

## Status

The status is recomputed from the tokens after every measurement cycle (only the cases the requirements ask for). H100, tokens at main
5383def09 (2026-11-10): no violation outside the exceptions above (the remaining misses: APB / OuterProductMean / PairEmbedCA cuda vs
triton at L <= 256, at most 3.7 %). The same check against the external baselines (cuda vs `anthropic` / `cueq`, not a ladder rule but
the engine's target): only LayerNorm misses (exception).

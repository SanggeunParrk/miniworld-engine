# B200 (sm_100a, card `b200_sm100`): status and how to read it

This directory holds the per-module notes of the B200 port (rounds 1-3 and the S90 rounds, 2026-11-16 .. 2026-11-18). Every note is a
dated log: newest passes are at the end of a file, a later pass supersedes an earlier one where it says so, and each pass names the
`cuda:b200_sm100` version it ended at. This page is the entry point: where to look, what the limits of the card are, and which version
each class stands at.

## What shipped, in one paragraph

The B200 is a routed card: `kernels/<family>/cuda/` picks the `sm100` / `b200_*` sources on a compute-capability 10.0 device and the
H100 (`sm90`) sources stay byte-identical to engine 2.5.14, except `bias_only_dit/cuda/bo_wgrad.cu`, which is a B200-only kernel
(`bo_wgrad_sm100`, tcgen05 / TMEM; it is only built through the sm100 kernel loader) that happens to live in a shared directory. Every
B200 kernel is hand CUDA (tcgen05, TMEM, TMA, programmatic dependent launch); Triton stays where it was and gained B200-only branches
under `triton:b200_sm100` versions. Accuracy follows the engine standard (rms / max against fp64), never bit identity with H100.

## Per-module notes

| class family | note | where the blocker table is |
|---|---|---|
| TriangleMultiplication, RF3, Bidirectional | [trimul.md](trimul.md) | "TriMul inference: blockers per case group" (inference); "Gaps" and the S90 training parts (training) |
| TriangleAttention (all variants) | [triattn.md](triattn.md), [triattn_s90_inf_sol.md](triattn_s90_inf_sol.md) | "TriAttn inference: blockers per case group", "TriAttn training: blockers per case group"; per-case speed-of-light table in `triattn_s90_inf_sol.md` |
| Transition, LayerNorm, RMSNorm, InputFeatureEmbedder | [transition.md](transition.md) | "S90 consolidation ... blockers per case group" (inference); "What bounds the small cases" (timer grid, launch floors) |
| Atom DiT (local / sliding window), SWAAtomTransformer, ESMFold2 DiT | [atom-dit.md](atom-dit.md) | "Part 4 ... the backward / forward limits measured", "Part 20 ... what bounds the one-sample chain", "Ladder after this round" |
| Token DiT family, AttentionPairBias | [token-dit.md](token-dit.md) | "Gaps (after round 3 part 3)" |
| Bias-only token DiT, TokenPairInit | [bias-only.md](bias-only.md) | "Gaps" |
| OuterProductMean, MSAPairWeightedAveraging | [msa.md](msa.md) | "Gaps" |
| Sampler glue, smooth lDDT, pair-embed builders, ParallelResidual, ProteinMPNN, PairformerBlock | [misc.md](misc.md) | per-class tables in the file |

Per-pass evidence that is too large for the notes lives in subdirectories: `triattn_s90_train/` (training probes, cluster barrier
microbenchmark `cb/`), `s90_transition_inf/` (inference accuracy / speed drivers), `trimul_s90_train_sol*.txt` (training speed-of-light
tables).

## Rejected variants

`rejected/` holds 165 diffs of variants that were built, measured and not adopted (60 transition, 27 trimul, 26 triattn, 15 each for
apb / local DiT / swa, 7 token DiT). The name is `<family>_<pass>_<what>.diff` and the note of that family names the measurement that
rejected it; nothing in the tree depends on them. A rejected variant is not reachable from the code: it is in git history and here only.

## Measured limits of the card (rules the notes rely on)

These are measurements, not guesses; each links to the pass that produced it.

| limit | value | where |
|---|---|---|
| kernel boundary under programmatic dependent launch (empty / ~3 us kernels, graph replay) | 0.73-0.91 us (1.4-1.6 us with 200 FMA iterations of work); plain launch chain 1.3-1.4 (1.8) | [triattn.md](triattn.md), "S90 inference, part 10 ... the kernel boundary" |
| software grid barrier | about the cost of a PDL boundary with an entry trigger (~1.5 us each): no ~5 us launch to win | [trimul.md](trimul.md), training part 9; [triattn.md](triattn.md), S90 training part 6 |
| cluster barrier | measured with `triattn_s90_train/cb/cb.cu` (arrive.release / wait.acquire pairs, one CTA an SM) | [triattn.md](triattn.md), "Cluster barrier cost" |
| TMA operation | ~590-740 clk per SM per op whatever its size and whichever lane issues it (the op rate is the limit, not the issue lanes); 16 KiB boxes ~28 B/clk, 32 KiB ~53 B/clk. Rule: merge ops into larger boxes, do not add issue lanes | [trimul.md](trimul.md) (k1w grouped weight boxes), [triattn.md](triattn.md) "S90 inference, part 13" |
| event timer | steps of 2.048 us; a replay median is round(t / 2.048 us) ticks and an empty graph replay is 3 ticks, so a case within one tick of the 1.5x-vs-H100 threshold is a tie, not a miss | [trimul.md](trimul.md) "The timer grid"; [transition.md](transition.md) "What bounds the small cases" |
| shared memory | 227 KB per CTA; a 232,448 B static assert is used where the plan fills it | per kernel |
| TMEM | 512 columns per SM; kernels that fill it say so in their note | per kernel |

## Versions at the tip

`cuda:b200_sm100` entries of `backend_versions` (the only version the B200 work bumps; no module `version` changed and no other
card's `backend_versions` changed against engine 2.5.14, except the new `triton:b200_sm100` entries listed below). Convention where a
class's counter is shared by inference and training: **inference bumps are even, training bumps are odd**, and a merge keeps both
(Transition, TriMul and TriAttn follow it: Transition 60 (inference), TriMul 73 (training; 72 was the last inference bump),
TriAttn 67 (training; 66 inference)).

| class | `cuda:b200_sm100` |
|---|---|
| TriangleMultiplication, TriangleMultiplicationRF3, BidirectionalTriangleMultiplication | 73 |
| TriangleAttention, AF3, Chai1, Chai1Confidence, RF3 | 67 |
| BidirectionalTriangleAttention | 4 |
| Transition | 60 (`triton:b200_sm100` 22) |
| AttentionPairBias | 29 |
| AtomDiTBoltz2 / Chai1 / ESMFold2Pair / OpenFold3 / Protenix / RoseTTAFold3 | 27 |
| AtomDiTESMFold2 / AtomDiTMiniWorld | 18 |
| SWAAtomTransformer | 26 |
| TokenDiTBoltz2 / Chai1 / MiniWorld / Protenix / RoseTTAFold3 | 24 (`triton:b200_sm100` 2) |
| TokenDiTMiniWorldBiasOnly | 3 |
| InputFeatureEmbedder | 27 |
| MSAPairWeightedAveraging | 21 |
| OuterProductMean | 4 |
| LayerNorm | 22 |
| PairEmbedAdaLN / PairEmbedCA / PairEmbedGeom | 20 (`triton:b200_sm100` 20 on AdaLN) |
| PairformerBlock, ParallelResidual | 2 |
| AlignEuler, AugmentNoise, PoolAtoms, SmoothLddtLoss | 2 |
| TokenPairInit | 3 |
| ProteinMPNN | 1 |

## Where the code is

- Sources: `src/foldjet_engine/kernels/<family>/cuda/` (`sm100*`, `b200_*` directories and files), the integrations
  `kernels/integrations/*_b200.py` / `*_sm100.py`, the shared graph RNG writer `kernels/_graph_rng_sm100.{py,cu}` (the one
  `aten::fill_.Scalar` override on the B200).
- Prebuild: `python -m foldjet_engine.kernels._prebuild --arch sm100 --list | --check | --jobs N` plans and builds every cubin /
  extension the routed shapes need, from the requirements rows.
- Tests: `tests/gpu/test_*_b200.py` (GPU), the CPU gate is the repository's `tests/cpu`.
- Tokens: `src/foldjet_engine/judge/tokens/b200_sm100/`; `python -m foldjet_engine.judge --gpu b200_sm100 owed` says what a
  version change owes.

## Debug / probe switches

Timing-only ablation macros that were default off and unrouted (the `WF_NO*` / `WT_NO*` / `FR_NO*` / `WG_NO*` families of the wide
TriAttn kernels, `K3W_ABL_*`, `K3W_T2`, `TA_FTL`, probe `#ifdef` blocks, the fused small-L TriMul backward `b7f`) were removed in the
finalisation pass. A kernel that stays behind an environment switch is documented in its note ("A / B" switches such as
`FOLDJET_ENGINE_*_B200_*`, `MW_LD*`); timer instrumentation macros (`*_TL`, `TRACE`) stay because Python builds them for the measurements
the notes quote.

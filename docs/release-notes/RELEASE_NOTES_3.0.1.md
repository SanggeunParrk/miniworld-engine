# miniworld-engine 3.0.1

Base: 3.0.0 (`6bedf9f0d`). Integration branch `rc/3.0.1` (bf16 training / inference work of the H100 branches, merged and verified together).
FP8 is unchanged: default bf16 / fp16 routes are as in 3.0.0 except for the changes below; `engine-fp8` behaves as in 3.0.0. The per-module
speed figures in the table below are the per-branch measurements quoted from the branch docs (`docs/h100/*.md`) and commit messages (H100,
CUDA graph, interleaved with a same-cubin control, "cool / sustained" as stated in each doc) unless a row says it was re-measured on the
integrated tree; the whole-tree judge re-measure is the section "Judge re-measure of the integrated tree".

## Output changes (judge owed) and version bumps

`backend_versions["cuda:h100_sm90"]` only (no module `version` bumped, `PairformerBlock` backend version not bumped):

| class | 3.0.0 | 3.0.1 | cause |
|---|---|---|---|
| TriangleAttention | 11 | 12 | row-split query-side training backward (dbias summation order) |
| TriangleAttentionAF3 | 11 | 13 | + 64-channel training core |
| TriangleAttentionRF3 | 12 | 14 | + 64-channel training core |
| TriangleAttentionChai1 / Chai1Confidence | 7 | 9 | as RF3 (parent class 14) |
| TokenDiTProtenix / TokenDiTMiniWorld | 3 | 5 | shared conditioning (4), bf16 pair bias on the AttentionPairBias kernels (5) |
| TokenDiTBoltz2 / RF3 / Chai1 | 6 | 7 | bf16 pair bias |
| Transition | 6 | 7 | D768 / D384 / D512 training on the generic kernels, chunked wide inference in place |
| OuterProductMean | 3 | 4 | wide TMA prologue outside the TMA table, `normalize_before_proj=True` cells only (outputs differ by at most 1 bf16 ulp; the AF3 order is unchanged) |
| MSAPairWeightedAveraging | 1 | 2 | wide backward aux kernels (bit-identical), native builds at every served training shape |
| AtomDiTMiniWorld | 2 | 3 | bf16 modulation table (training) |
| SWAAtomTransformer / InputFeatureEmbedder | 3 | 4 | as AtomDiTMiniWorld |

Requirements kernel rows: `triangle_attention_dh_bwd_q_sm90` 3 -> 4 (note: this also owes the triton rows of the TriangleAttention
AF3 / RF3 / Chai1 / Chai1Confidence classes, see owed counts), new `triangle_attention_d64_fwd_train_sm90`; the `apb_pair_bias_bwd*` rows
list the TokenDiT classes; `parallel_residual_{fwd,bwd}_sm90` list `PairformerBlock`; new `PairformerBlock` rows `chai1`, `chai1_confidence`.

## Per-module summary

| module | change | measured (branch docs) |
|---|---|---|
| TriAttn training, query side (all widths, S >= 512) | `TB_Q8` row-split kernel, half the TMA reduce-adds, wave-fit slab count | module step 0.93-0.96x of 3.0.0 (L512-768, 64-384 channels); dq bitwise unchanged, dbias differs in fp32 summation order (rms 1e-7 rel.) |
| TriAttn training, 64-channel heads (RF3, Chai-1) | wgmma bf16 forward core + two-row query-side backward | step L384/512/768/1024 RF3 2.70 -> 2.59, 5.47 -> 5.14, 15.9 -> 14.2, 35.2 -> 31.2 ms; Chai-1 3.08 -> 2.94, 6.17 -> 5.71, 17.5 -> 15.9, 38.5 -> 34.6 ms (sustained) |
| token DiT training, shared conditioning | conditioning GEMMs on L rows, sample sum as hi + lo | A48 step vs 3.0.0 (cool / sustained): L128 0.88 / 0.90, L768 0.91 / 0.92, L1536 0.94 / 0.94; A1 unchanged; not bit-identical (summation order) |
| token DiT pair bias | AttentionPairBias tensor-core kernels for the bf16 training pair bias (128 / 256 / 512 ch) and wide inference hoist | kernel level L768 d128: fwd 135 -> 63 us, bwd 233 -> 130 us (floor 57 / 113); outputs change at bf16 rounding |
| atom DiT training (tr_bwd90 grouped) | dmod slabs alias the second weight buffer (2 CTA/SM) | tr_bwd90 400 -> 275 us; block step 0.966x (A48 x 4096); bit-identical |
| SWA / atom DiT training | bf16 modulation table | step 1.13-1.147x (window rows), 1.045-1.11x (global rows); A48 x 4096, 3 blocks 11.09 -> 9.80 ms |
| PWA training (wide) | fused dropout-mask / dv-permute aux kernels, native builds at every served shape, glue3 mask prefetch | 8x32 d_msa 64: N512 S8192 0.90x of 3.0.0 (16.09 -> 14.50 ms); bit-identical (pair side stays Triton beyond the six shapes) |
| Transition training | generic kernels from 16384 rows (D384 / D512) and every row count (D768 n2) | D768 step 1.2-1.85x vs the wide build; D512 8-17 %; D384 tie (5-14 % at 24576-65536 rows) |
| Transition chunked inference (D384 / D512) | blocks write into the output rows (no DtoD copy) | 3456 / 3968 tokens 45.8 / 60.5 -> 40.1 / 53.2 ms; bitwise |
| OPM inference, `normalize_before_proj=True` (Boltz order) | wide TMA prologue outside the TMA table | re-measured on the integrated tree (one node, interleaved, graph, L2 flush; cool / sustained): L1024 / S1024 4.27 / 4.43 -> 3.59 / 3.9 ms, L1024 / S2048 7.92 / 8.66 -> 6.53 / 7.3, L1536 / S1024 8.94 / 9.8-10.4 -> 7.90 / 8.8, L1536 / S2048 16.6 / 19.0-19.7 -> 14.4 / 16.8; the AF3 order (`normalize_before_proj=False`) is unchanged (always the wide prologue); the earlier branch-doc figures (4.54 -> 3.83 etc.) did not reproduce in this protocol and are not claimed |
| AttentionPairBias ragged L | pair read unpadded; Chai-1 entry bf16 form cached per tensor | removes the `F.pad` of the pair (81 % of a 13.2 ms call at L2976); bitwise; entry cache: cast 4.94 ms -> 0.68 ms replay at L2048 under `static_inputs()` |
| glue | edge_index kernel (one launch), ParallelResidual cuda backward writes leaf gradients in one pass | see docs (`docs/h100/dit_variants.md`, `local_atom_dit.md`) |
| PairformerBlock(parallel=True) cuda | fused 5-branch ParallelResidual for the pair and 2-branch for the single deltas; fuses when the updates run cuda | docs: parallel pairformer fused residual (Chai-1 pairformer block ~1.2x claimed by the branch; see the integrated-tree check in the final report) |
| TriMul wide training | frozen cuBLASLt algorithms, only bit-equal candidates | gradients bit-identical to 3.0.0; warnings gone |

## Judge re-measure of the integrated tree (H100, `h100_sm90`, 2026-11-18, tree `rc/3.0.1`)

Cycle via `tools/slurm/cycle.sh` (4 G groups + 2 kit groups K1 / K2 on node01, 1 GPU / 10 min per job; two passes, the second only for the cases the
first pass cut at its time limit). Owed before: accuracy 2069 + speed 2069 (adapter bumps listed above, kernel `triangle_attention_dh_bwd_q_sm90`
3 -> 4, 96 new `PairformerBlock` rows); owed after: **0 / 0**, pending 0. **RED: 0.** The 108 extra `triton` owed cells (TriangleAttentionAF3 48,
Chai1 24, Chai1Confidence 12, RF3 24) come from the kernel row `triangle_attention_dh_bwd_q_sm90` 3 -> 4 (the Triton rung lists it); their outputs
did not change (lights unchanged, see below).

Accuracy, re-measured `cuda` cases (GREEN / YELLOW; max error as a multiple of tol; skipped = OOM of the PyTorch reference at long L):

| class | GREEN | YELLOW | max err / tol |
|---|---|---|---|
| AtomDiTESMFold2 | 22 | 14 | 4.85 |
| InputFeatureEmbedder | 39 | 105 | 3.58 |
| MSAPairWeightedAveraging | 76 | 21 | 1.46 |
| OuterProductMean | 220 | 1 | 1.005 |
| PairformerBlock (new rows chai1, chai1_confidence) | 10 | 6 | 1.44 |
| SWAAtomTransformer | 32 | 44 | 4.62 |
| TokenDiTBoltz2 / Chai1 / MiniWorld / Protenix / RF3 | 27 / 84 / 13 / 149 / 29 | 21 / 30 / 11 / 119 / 19 | 3.11 / 1.97 / 3.39 / 3.17 / 3.66 |
| Transition | 235 | 63 | 1.64 |
| TriangleAttention / AF3 / Chai1 / Chai1Confidence / RF3 | 144 / 90 / 39 / 19 / 36 | 20 / 2 / 9 / 5 / 12 | 1.43 / 1.13 / 1.25 / 1.10 / 1.27 |

Lights changed against the 3.0.0 tokens (cases that existed before), `cuda`: GREEN -> YELLOW 16 (SWAAtomTransformer 3, TokenDiT Boltz2 1 / Chai1 1 /
MiniWorld 1 / Protenix 8, TriangleAttentionChai1 1, RF3 1; all within 1.12 x tol) and YELLOW -> GREEN 14 (TokenDiT MiniWorld 1 / Protenix 9,
TriangleAttentionAF3 1, Chai1 3); borderline cases move both ways at about 1.0 x tol. The `pytorch` rows of InputFeatureEmbedder moved
GREEN -> YELLOW 11 / YELLOW -> GREEN 16 and SWA 1 YELLOW -> GREEN at 1.00-1.08 x tol (the baseline against its own fp64 gold, same code: rerun
variation). No case moved to RED. `triton` rungs: lights unchanged in all four TriAttn classes.

Speed, `cuda` median ms against the 3.0.0 token of the same case (geometric mean per class; < 1 = faster): AtomDiTESMFold2 0.976, IFE 0.978,
PWA 0.968, OPM 1.010, SWA 0.978, TokenDiT Boltz2 0.986 / Chai1 0.846 / MiniWorld 0.998 / Protenix 0.912 / RF3 0.982, Transition 0.998, TriAttn 0.997,
AF3 0.983, Chai1 0.986, Chai1Confidence 0.991, RF3 0.979. The same cases' `pytorch` baselines (untouched code) moved by 0.998-1.005 on
average, 0.925-1.154 at the extremes: that is the noise of a re-measure on a shared node. **Slower than the 3.0.0 token, plainly:** 85 `cuda`/`triton`
cells are more than 3 % slower in raw ms (36 more than 5 %, 7 more than 10 %: OPM d_pair 256 L640 +15.5 %, Transition d_hidden 256 L768 +13.7 %,
TokenDiTProtenix intellifold2 L128 +11.3 %, TriAttention Chai1Confidence triton L2048 +10.3 %, OPM count_offset L768 +10.0 %, TriAttention protenix
d_pair 256 L640 +9.6 %). Normalised to the same case's PyTorch time 21 `cuda` cells (of 2069) are more than 5 % slower: OPM 12, Transition 5,
TriAttention 3, TokenDiTProtenix 1, TriAttentionChai1 1 (worst 0.873 of the 3.0.0 speedup: OPM d_msa 128 / d_pair 256 L640 eval). These are on code
the integration did not change for those routes (OPM d_pair 256 / 512, Transition eval at D128 / 256 / 384, TriAttn d_pair 256 eval; a same-job
interleaved check of OPM d_pair 128 cells found the 3.0.0 and 3.0.1 trees equal to 0.3 %), so they are read as re-measure noise, not as regressions;
no row was dropped and no tolerance loosened. The raw numbers are in `judge/tokens/h100_sm90/` and `benchmarks/h100_sm90/`.

New `PairformerBlock` rows (chai1, chai1_confidence; cuda ms, eval): chai1 L128 0.419, L512 4.42, L1024 19.9 (4.8x / 10.7x / 15.7x of PyTorch,
2.0x / 2.3x / 2.2x faster than the Anthropic kit); train chai1 L128 1.71, L512 21.0, L768 53.7 (3.6x / 6.3x / 8.1x). The branch's "~1.2x" for the fused
parallel residual is a 3.0.0-vs-3.0.1 claim and has no old token to compare with; it is not stated here.

## Accuracy decisions

- token DiT pair bias: 8 cases moved GREEN -> YELLOW, all within 1.12x tolerance, RED 0: accepted as bf16 rounding noise.
- PWA: the CUDA pair backward stays at the six qualified shapes (NATIVE_PAIR_ANY_SHAPE = False): at d_pair 128 it moved two GREEN rows to
  YELLOW (ln_pair.weight 0.96 -> 1.03, to_bias.weight 0.89 -> 1.02 of tol).

## Prebuild regeneration (release build)

Kernels wheels (sm90 / sm80 / sm120, cu129, cp312) rebuilt incrementally from the 3.0.0 release cache with GCC 12.4 (CUTLASS 4.2, MathDx): sm90a 41
built (re-keyed), 559 artifacts, 0 absent (3.0.0: 560); sm80 2 built (`dit_variants_edge`, `parallel_residual`), 72; sm120a 2 built, 64. Manifest names
(additions only): sm90 +2 (`miniworld_pwa_wide_aux`, `triattn_h100_d64t`); the 3 generic PWA names no longer planned (`miniworld_pwa_dgv_bwd`,
`_glue3`, `_pair3`: every served training shape is native) are carried byte-for-byte from the 3.0.0 sm90 wheel (as `triattn_h100_core_h16_v2` is
from 2.5.5), so no name of 3.0.0 is dropped; sm80 / sm120 +0 / -0 names.

## Not in 3.0.1 / deferred

FP8 training routes; TriAttn Q4/Q5/Q8 follow-ups; TriMul G3/G4; OPM d_pair >= 256 backward kernel; PWA structural rewrite; atom DiT register
work; IFE 7168 step.

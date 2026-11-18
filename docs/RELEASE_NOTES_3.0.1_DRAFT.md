# miniworld-engine 3.0.1 (DRAFT, not a release)

Status: local integration branch `rc/3.0.1` on top of 3.0.0 (`6bedf9f0d`). Nothing is pushed, tagged or uploaded. FP8 is not part of this
release: default bf16 / fp16 routes are as in 3.0.0 except for the changes below. All speed numbers are the per-branch measurements quoted
from the branch docs (`docs/h100/*.md`) and commit messages (H100, CUDA graph, interleaved with a same-cubin control, "cool / sustained"
as stated in each doc); they were NOT re-measured on the integrated tree except where `docs/` of the final tree says so.

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
| OuterProductMean | 3 | 4 | wide TMA prologue outside the TMA table (outputs differ by at most 1 bf16 ulp) |
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
| OPM inference | wide TMA prologue outside the TMA table | L1024 / S1024 4.54 -> 3.83 ms, L1024 / S2048 8.56 -> 7.55, L128 / S4096 0.58 -> 0.26 (same-cubin control) |
| AttentionPairBias ragged L | pair read unpadded; Chai-1 entry bf16 form cached per tensor | removes the `F.pad` of the pair (81 % of a 13.2 ms call at L2976); bitwise; entry cache: cast 4.94 ms -> 0.68 ms replay at L2048 under `static_inputs()` |
| glue | edge_index kernel (one launch), ParallelResidual cuda backward writes leaf gradients in one pass | see docs (`docs/h100/dit_variants.md`, `local_atom_dit.md`) |
| PairformerBlock(parallel=True) cuda | fused 5-branch ParallelResidual for the pair and 2-branch for the single deltas; fuses when the updates run cuda | docs: parallel pairformer fused residual (Chai-1 pairformer block ~1.2x claimed by the branch; see the integrated-tree check in the final report) |
| TriMul wide training | frozen cuBLASLt algorithms, only bit-equal candidates | gradients bit-identical to 3.0.0; warnings gone |

## Accuracy decisions

- token DiT pair bias: 8 cases moved GREEN -> YELLOW, all within 1.12x tolerance, RED 0: accepted as bf16 rounding noise.
- PWA: the CUDA pair backward stays at the six qualified shapes (NATIVE_PAIR_ANY_SHAPE = False): at d_pair 128 it moved two GREEN rows to
  YELLOW (ln_pair.weight 0.96 -> 1.03, to_bias.weight 0.89 -> 1.02 of tol).

## Prebuild regeneration (release build)

Re-keyed or new artifacts versus 3.0.0 (`_prebuild --check`): sm90a 41 re-keyed + 2 new (`miniworld_pwa_wide_aux`, `triattn_h100_d64t`),
3 generic (non-native) PWA builds no longer planned (`miniworld_pwa_dgv_bwd`, `_glue3`, `_pair3`: every served training shape is native);
sm80 and sm120a: `parallel_residual`, `dit_variants_edge`. Full list in the final report of the integration.

## Not in 3.0.1 / deferred

FP8 training routes; TriAttn Q4/Q5/Q8 follow-ups; TriMul G3/G4; OPM d_pair >= 256 backward kernel; PWA structural rewrite; atom DiT register
work; IFE 7168 step.

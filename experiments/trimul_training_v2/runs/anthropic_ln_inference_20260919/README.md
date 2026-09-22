# Inference-only input LayerNorm split/fused experiment

2026-09-19 · node02 H100 · B1 C128/H256 · bidirectional packed contraction · L384/768.

## Scope

User correction: **do not retune the fused original Anthropic baseline**.

- Fused: directly call unchanged upstream `k1_body` and `k3_body` from pinned
  `f4f62fa6592ae4938d49b1757bea0cfeff9f468e` with the shipped C128/H256 BF16 defaults.
  K1 `(2,64,8,2)` and K3 `(2,64,4,1)`. Input LN is computed in each kernel as in the original.
- Split: standalone input LN produces x_n once, shared by modified K1 and K3.
- Output LN stays fused in K3 in every row.
- **No mean/rstd, normalized contraction, projection/gate/preactivation training saves.**
  x_n exists only in the split row because it is a required forward intermediate.
- No dropout or RNG. Mask and residual enabled identically.
- Same packed cuBLAS contraction for all rows. Prepacked weights, CUDA Graph replay.
  This isolates kernel inference, not Python overhead or per-call weight packing.
- This adapts the original K1/K3 to the existing bidirectional contraction. It is not a timing of the upstream single-direction public API.

## Implementation

`original_k1.cu`, `original_k3.cu`: wrappers only, original bodies/settings unchanged.
`k1.cu`: upstream body with input LN disabled and tunable register budget/schedule.
`k3.cu`: upstream body with input LN disabled, retains fused output LN. Since its input is
x_n, initially read the original BF16 residual from global in the epilogue.
`k3_tma.cu`: additionally tested and selected; reuse the shared input stage for a TMA residual load after x_n reaches registers. No extra training stores; original inference rounding preserved.
`separate_ln*.cu`: original `ln_fragment` directly included; mean/rstd stores removed.
`triton_ln.py`: existing engine LN arithmetic/tiling, with only mean/rstd stores removed for inference.
No statistics allocation or training-only tensor writes in either LN implementation.
These are isolated experiment sources; production dispatch is unchanged.

## Search

Only the split path is retuned:

- K1: 66 valid initial tile/ring/chunk combinations; default scheduling/register share initially.
- K1 refinement: best three geometries, reduced register budgets, two scheduling modes,
  and warpgroup start offsets 0/2/4. Invalid ring/resource combinations pruned.
- K3: 44 valid tile/ring/accumulator/register-partition/LN-scheduling combinations.
- Standalone Anthropic-derived LN: 12 candidates (vector 64/128/256/512 threads,
  TMA loads, TMA stores, affine serialization on/off).
- Standalone Triton LN: all 800 unique registered `(BLOCK_M1,BLOCK_K,num_warps,num_stages)`
  configurations explicitly launched. No 24-candidate miss cap.

Each candidate is checked numerically, then timed. Top four candidates are remeasured in
alternating order for 8 rounds. Final comparison uses 18 rounds × 200 replays, all six order permutations, after 200 warmup replays per path. Component timings use 12 rounds × 30 replays.
Per-candidate failures, samples, settings and build/ptxas artifacts are retained locally.
Shape selection is independent at L384 and L768. This finite search is not a proof of a global optimum.

## Reproduce on allocated node02 GPU

```
bash runs/anthropic_adoption_20260919/env.sh python runs/anthropic_ln_inference_20260919/check.py
bash runs/anthropic_adoption_20260919/env.sh python runs/anthropic_ln_inference_20260919/tune.py
```

The original/default baseline does not depend on tuning results. Compile flags and source
hash-based cubin paths are recorded under build/. Full summaries will be in results.json.

## Final result

Original fused baseline remains unchanged. Split K1 uses 192-token tiles and 152 consumer registers (original 128-token tile / 232).
L384 tile is 6x32; L768 1x192. Shape/ring search plus register/schedule/start-offset refinement was performed.
K3 global residual loads were slower; an additional 44-candidate sweep of TMA residual staging removed most of that regression.
The first TMA prototype had a barrier-release condition incompatible with residual=1; the validation process was killed and this condition was fixed before successful measurements.

| Path | L384 ms | L768 ms |
|---|---:|---:|
| original | 0.277 | 1.151 |
| split-anthropic-ln | 0.282 | 1.154 |
| split-triton-ln | 0.281 | 1.160 |

L384 original is about 1.6% faster than the Triton-LN split path; L768 differences are under 1% and within run variation.
The shorter first measurement changed the L768 ranking; do not claim a meaningful L768 winner. All raw samples are retained.
Standalone LN: Triton 28.24/102.06 us vs Anthropic-derived CUDA 29.23/104.60 us at L384/768.
These are standalone kernel timings, not guaranteed end-to-end gains.

## Verification

- All 800 Triton candidates completed without numerical failures at both shapes.
- Final native-split outputs bitwise equal to the original at tested sizes.
- Triton split relative L2 around 8e-5; LN adversarial constant/small-variance/offset inputs checked.
- Final selected configs at L64/72: memcheck 0 errors, racecheck 0 hazards.
- binary-selected.json records resources at L768. Split K1/K3 have zero local bytes; unmodified original K3 reports 40 local bytes in this build.
- Final repeat protocol and telemetry.csv retained; no causal attribution of variation solely to clocks.
- Global production dispatch and existing model training unchanged.

## K3 recomputation experiment

`bench_recompute.py` keeps the tuned split LN/K1 unchanged, swaps only K3 to the unchanged original kernel with original x, so K3 recomputes input LN and reuses raw x for residual. No retuning of the original.

Same-process five-way comparison, 20 rotating/reversing rounds, 200 graph replays each, seed 67, dropout off and no training saves. All outputs validated. Anthropic-derived LN routes are bitwise equal to original; Triton routes have relative L2 below 8e-5.

| Path | L384 us | L768 us |
|---|---:|---:|
| original | 279.06 | 1154.71 |
| anthropic-share | 282.84 | 1153.21 |
| anthropic-recompute | 285.14 | 1169.10 |
| triton-share | 283.65 | 1165.49 |
| triton-recompute | 284.94 | 1176.34 |

No speed benefit observed from K3 recomputation. Recompute was slower in 19/20 and 17/20 rounds with Anthropic LN (L384/L768), and 13/20 and 15/20 with Triton LN. Raw samples and paired differences are in recompute-results.json. Do not interpret tiny median differences as a universal ranking.

# Anthropic LayerNorm split/fused A/B

2026-09-19, node02 H100. Experimental training forward, not a default dispatch change.

## Source finding

Pinned upstream `anthropics/uplifting-biomolecular-modeling` revision
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`, native v5 `csrc/tmn_kernels.cuh:185-272`.
`ln_fragment` is an explicitly optimized device function, not a standalone LN kernel:

- Two rows per four-lane subgroup, using the MMA register fragment layout directly.
- Balanced per-thread sums and two shuffle steps; FP32 mean and centered variance.
- Retain packed BF16 registers, re-convert during each pass instead of keeping FP32 values live.
- Register fences prevent compiler hoisting; SERIAL affine mode uses two interleaved dependency chains to control live registers and avoid spills.
- Shared gamma/beta vector loads, fused affine and immediate BF16 repacking.
- K1/K3 combine this function with TMA staging and producer/consumer scheduling. TMA/WGMMA are not arithmetic instructions for LN itself.

`ln_stock` alternatives reproduce a reference library's summation order; do not treat them as universally faster variants.

## A/B construction

`separate_ln.cu` and `separate_ln_tma.cu` include the **unchanged original** `ln_fragment`.
Separate LN candidates: vector loads with 128/256 threads; TMA+ldmatrix loads with 128 threads;
TMA loads plus TMA stores. Each has SERIAL off/on (eight candidates).
TMA variants use original K1 row-major and K3 channel-major operand layouts; TMA store uses the original stmatrix layout.

`fused_front.cu`: restore original input LN in the previous saved-front derivative;
emit normalized input and separate contiguous mean/rstd, plus unchanged a/b and preactivation saves.
`anthropic_k3_training.cu`: existing original-K3 derivative, output LN fused and all saves preserved.
Separate F2 and F567 use `anthropic_saved.py`. cuBLAS contractions unchanged.

Compare four combinations (input LN separate/fused × output LN separate/fused).
Both paths save normalized input/output, mean/rstd, a/b, preactivations, projection, gate and contraction output.
K3 reads the saved normalized input in both paths, so this is **not** the original inference policy that recomputes input LN.
Default front config `(2,64,6,1,0)`; output `(2,64,4,1,232)`, fused output SERIAL=1.
No exhaustive GEMM retuning or change to the existing backward.

## Measurement

`bench.py`: B1, C128/H256, L384/768, BF16, same fixed mask and dropout 25% scale, residual enabled.
8 alternating CUDA Graph rounds; full fwd 16 replays per round. **Weights prepacked** for all rows.
RNG generation, optimizer, weight packing and backward are excluded. Do not compare absolute values with earlier packing-included reports.
All twelve returned output/saved tensors are bitwise equal across all four variants at L384 and L768.
Small-shape checks at L64/72 also compare all individual saves.
`results.json` preserves round samples, chosen LN candidates, component and full-forward times.
`ln-reference.json` separately compares extracted LN with existing Triton LN; that reference uses at most 24 heuristic candidates on cache miss.

## Reproduction

Run only on allocated node02 GPU, not the login node:

```
bash runs/anthropic_adoption_20260919/env.sh python runs/anthropic_ln_ab_20260919/check_bulk.py
bash runs/anthropic_adoption_20260919/env.sh python runs/anthropic_ln_ab_20260919/bench.py
bash runs/anthropic_adoption_20260919/env.sh python runs/anthropic_ln_ab_20260919/bench_ln_reference.py
```

Source and cubins are isolated in this experiment directory. This does not change the production selection or existing training job.

## Results (training forward, ms)

| L | both separate | input separate/output fused | input fused/output separate | both fused |
|---:|---:|---:|---:|---:|
| 384 | 0.474 | 0.448 | 0.464 | 0.445 |
| 768 | 1.823 | 1.761 | 1.793 | 1.737 |

Memcheck: 0 errors. Racecheck: 0 hazards (L64/72).

Standalone extracted LN did not beat existing Triton decisively: L384 input 30.10 vs 28.93 us, output 59.58 vs 58.52 us; L768 input 106.36 vs 104.88 us, output 214.86 vs 215.11 us. See ln-reference.json.

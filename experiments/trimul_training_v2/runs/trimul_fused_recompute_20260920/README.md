# TriMul: inference forward + on-chip backward recomputation

## Outcome

Implemented CUDA/TMA/WGMMA B1–B4 and B7–B12 prototypes with activation reconstruction inside their consuming kernels. The measured prototype is slower than the saved-activation training path and is **not enabled in production**. This is not an optimized limit of the recomputation design.

Comparison baseline: the immediately preceding **Anthropic-derived saved training forward + optimized CUDA B1–B4/B7–B12**, with unchanged cuBLAS contraction backward. This is neither public Anthropic inference nor cuEquivariance.

H100 80GB node02, batch1, BF16 C128, outgoing/incoming hidden128 each, L384/768, dropout25%, mask and residual enabled. CUDA graphs, 600 interleaved measurements per path/scope. Includes live weight packing, all 11 gradients; excludes optimizer, RNG generation, CPU dispatch and compilation. Separate kernel traces are diagnostic and need not sum exactly to graph medians.

| L | Path | Forward ms | Backward ms | Measured total ms |
|---|---|---:|---:|---:|
| 384 | 직전 저장형 | 0.438 | 0.711 | 1.150 |
| 384 | 새 on-chip 재계산형 | 0.284 | 1.491 | 1.758 |
| 768 | 직전 저장형 | 1.736 | 2.755 | 4.486 |
| 768 | 새 on-chip 재계산형 | 1.104 | 5.823 | 6.913 |

## Exact storage and fusion policy

Forward: Anthropic-derived no-save K1 → cuBLAS outgoing/incoming → no-save K3 with matching training dropout/residual and BF16 rounding. Retain `left/right/tri`, input/weight/mask references and packed weights. No `x_n`, raw input projection/gate, output-normalized tri, output projection/gate, mean or reciprocal-standard-deviation activation buffers.

B1–B4: reconstruct input LN and output gate; reconstruct output LN/projection only where the derivative consumes them. DX skips the output projection GEMM; dW-projection roles also skip it. Reconstructed values stay in registers/shared memory. Output LN statistics are computed inside the same kernel.

B5–B6: unchanged cuBLAS contraction gradients. Trace confirms 2 forward + 4 backward contractions, no repeated forward contraction.

B7–B12: reconstruct input LN and projection/gate, immediately apply GLU backward, input/weight derivatives and input-LN backward. Input LN statistics are recomputed inside this kernel. One cooperative launch per fused region; no standalone activation-restoration or gradient-reduction launches. Weight-gradient partial reduction buffers are still global scratch, not forward activations. `dg/dtri/dleft/dright` remain gradient outputs between regions.

Logical retained intermediate activation bytes (input/weights/mask/output excluded):

| L | Saved path MB | New retained left/right/tri MB | Removed MB |
|---|---:|---:|---:|
|384|719.585|226.492|493.093|
|768|2878.341|905.970|1972.371|

These are tensor-size accounting, **not total training peak-memory measurements**. The comparison harness keeps both implementations and reference tensors live.

## Backward region traces

| L | Region | Saved CUDA us | New CUDA us | Ratio |
|---|---|---:|---:|---|
| 384 | B1–B4 | 168.5 | 570.9 | 3.39x slower |
| 384 | B7–B12 | 364.0 | 727.2 | 2.00x slower |
| 768 | B1–B4 | 605.9 | 2250.2 | 3.71x slower |
| 768 | B7–B12 | 1383.5 | 2831.7 | 2.05x slower |

## NCU

Warm-cache, full section set, cache/clock control disabled; profiling is a separate run. Percentages below are hardware throughput/activity metrics, not a computed percentage of the theoretical optimal complete algorithm. Full raw reports and CSVs are in this run directory.

| L | Region | DRAM throughput | SM throughput | Tensor pipe activity |
|---|---|---:|---:|---:|
| 384 | B1–B4 | 18.7% | 31.7% | 9.6% |
| 384 | B7–B12 | 12.9% | 34.9% | 22.0% |
| 768 | B1–B4 | 21.3% | 32.0% | 9.8% |
| 768 | B7–B12 | 12.9% | 36.7% | 23.2% |

Both kernels have about12.5% active-warp occupancy (one256-thread CTA/SM). B1 uses255 registers/thread; B7 uses238. Ptxas reports **zero register spills**. A two-role B1 candidate with spills was rejected, even though it was numerically correct.

Source inspection explains the next optimization work: B1 uses a single row buffer with load → LN → recompute GEMM → derivative GEMM dependencies; it no longer retains the original two-slot row prefetch. DW/DX roles repeat some reconstruction. B7's extra x_n shared tile and register pressure prevent the old two-CTA occupancy and repeatedly stage weights. This profile does not support a DRAM- or Tensor-Core-roofline claim, much less SoL90. Reducing global activation bytes alone did not compensate for the lost overlap and repeated computation.

## Configuration search

B1:64-row tiles,256 threads,132 CTAs; DW_SPLITS={20,24,28,32,36,40}, with3 DW groups; winner28 at both lengths. B7:64-row/64-hidden tiles,256 threads,132 CTAs; DW_SPLITS={4,6,8,10,12,14}, with8 DW groups; winner8 at both lengths. Register/shared stores were vectorized with stmatrix. This is **a CTA-role partition search**, not exhaustive autotuning of row/column tiles, pipeline stages and register budgets. Do not call it fully tuned.

## Verification and remaining accuracy limitation

- Initial output is bit-exact; all11 gradients meet relative-L2 <=5e-4 against the independent existing backward reference at both lengths. Region-level tests retain their stricter per-output bounds.
- After changing input, weights, upstream gradient, dropout scale and mask, captured graphs equal freshly executed new kernels bit-for-bit for output and all11 gradients. New-vs-saved maximum relative-L2 is0.00871% at384 and0.02685% at768.
- **The mutated L768 independent-reference check does not fully pass**: dWL relative-L2 is0.05485% for the new path and0.05413% for the saved baseline, exceeding the unchanged0.05% target. FP32 reduction segments2/4/8/16 did not resolve it. The failure is recorded in JSON; it was not converted to a pass by raising tolerance. It remains a limitation before production adoption.
- Memcheck: B1 and B7 independently at384, combined forward/backward at768:0 memory errors, numerical checks pass. Racecheck: both regions at384:0 errors,0 warnings. Initial B1 shared-buffer reuse had a synchronization bug that was fixed by a CTA barrier after both warp-groups finish TMA dtri writes; these are post-fix results.
- Trace: zero activation-restoration kernels,6 total cuBLAS contractions. No register spills in accepted cubins. The profiler's report-export helper prints an environment utf-8-sig warning after saving; .ncu-rep and CSV export both succeeded.

## Reproduction

Use node02 inside a two-GPU Slurm allocation. Existing training and Claude allocations were not stopped. This allocation was released after measurement.

```bash
export MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/trimul_fused_recompute_20260920/bench.py --length 384
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/trimul_fused_recompute_20260920/bench.py --length 768
```

`plans.py` builds cubins with SHA256 source/header/config keys and rejects spills. `b1_fused.cu`, `b7_fused.cu`, `common_recompute.cuh`, and `b1_saved_math.inc` contain the implementation. JSON includes source hashes, numerical results, graph replay checks, timing distributions, selected configs and kernel traces. `derive.py` records derivation from prior math. No engine default/dispatch was changed.

Attribution: this work follows and adapts Anthropic's Apache-2.0 TMA/WGMMA, LayerNorm fragment and matrix-layout primitives plus Miniworld's existing backward math. We are extending that inference work into training; these are not independent original inference kernels.

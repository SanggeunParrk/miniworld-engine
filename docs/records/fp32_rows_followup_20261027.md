# float32 registry rows, follow-up: Transition single-stream weight copies, AugmentedAttention training glue (2026-10-27)

Branch `perf/final-fp32-rows`, on top of 776a454b (see `fp32_rows_ladder_20261027.md`). Same harness, H100, compile + CUDA graph, median of 3.

## Transition fp32 inference: TF32-rounded weights once per parameter version

`fused_generic_sm90a._fwd_launch` rounded the three weights to TF32 into fresh copies on every call (`prep`, 13 us of 40 at D768). The inference entry now takes the copies from
`_weights.tf32_rounded_triple`: filled on the first call, reused while the three weights are the same tensor objects at the same version. Guards (same family as `transposed_contiguous`, plus a liveness guard):
the entry holds weak references to the three weights and serves only those very objects, so an address freed and reused by another module cannot hit; a new version replaces the entry; nothing is filled
under grad mode with a parameter requiring grad, under `torch.compile` tracing, or during a CUDA-graph capture (warm up once before capturing; a capture reads a hit). The training forward is unchanged.

| row | mode | L | CUDA ms before -> after | portable ms | c/p before -> after |
|---|---|---|---|---|---|
| trc_fp32_single_D768 | inference | 128 | 0.0442 -> 0.0314 | 0.0405 | 1.091 -> 0.774 |
| trc_fp32_single_D768 | inference | 384 | 0.0451 -> 0.0322 | 0.0453 | 0.994 -> 0.711 |
| trc_fp32_single_D768 | inference | 768 | 0.0551 -> 0.0416 | 0.0524 | 1.051 -> 0.795 |
| trc_fp32_single_D384 | inference | 128 / 384 / 768 | 0.0249 / 0.0253 / 0.0257 -> 0.0214 / 0.0216 / 0.0221 | 0.0285 / 0.0286 / 0.0314 | 0.87 / 0.89 / 0.82 -> 0.75 / 0.75 / 0.71 |

Test: `test_fp32_inference_reuses_the_rounded_weights_and_never_serves_a_stale_or_foreign_copy` (reuse on a repeat call, rebuild after an in-place update, three modules on possibly reused addresses, no fill under grad).
The portable / PyTorch column of these rows (1.02 - 1.43) is the Triton agent's and is unchanged.

## AugmentedAttention token fp32 training: glue passes folded away

Per training step at L128, A = 48 the TF32 path made these extra fp32 passes over [A L, 768]: q/k/v rounding (+ v transpose), two transposes (q, k), one transpose (dO), the dO rounding pass, and in the pair backward a gather of the
key-permuted dbias into natural order. Now: `prep_k<true>` writes q^T and k^T in the same pass (training forward only; inference unchanged), `dd_k` writes dO^T from shared memory in the pass that rounds dO, and the
pair backward kernels read the permuted dbias directly (`dbpos`). 3 transpose launches and the gather launch are gone: total kernel time of the step 1784 -> 1750 us (portable 1731).

| row | mode | L | CUDA ms | portable ms | c/p (before 776a454b -> now) |
|---|---|---|---|---|---|
| aat_fp32 (d_pair 128) | training | 128 | 1.525 | 1.499 | 1.034 -> 1.017 |
| aat_fp32 | training | 384 / 768 | 4.651 / 10.876 | 5.166 / 13.685 | 0.91 / 0.80 -> 0.90 / 0.79 |
| aat_fp32_c768 (d_pair 256) | training | 128 | 1.695 | 1.678 | 1.020 -> 1.010 |
| aat_fp32_c768 | training | 384 / 768 | 5.237 / 12.308 | 5.771 / 15.107 | 0.92 / 0.82 -> 0.91 / 0.81 |
| aat_fp32 / aat_fp32_c768 | inference | 128 | 0.0888 / 0.1080 | 0.0908 / 0.1067 | 0.96 / 0.997 -> 0.98 / 1.012 (c768 within run-to-run spread; unchanged code path) |

OPEN: L128 training stays 1 - 2 % above portable, not at the 0.95 target. What is left is not glue that can be fused away: the step is ~1.5 ms of which ~0.9 ms is cuBLAS GEMMs and autograd adds common to both rungs;
the CUDA-only work is q/k/v rounding+transposes 61 us (bandwidth bound: 171 MB), dO rounding+transpose 30 us, pair backward 30 us (16 384 rows, latency bound), partial-sum reductions ~15 us, against a faster core
(fwd + dq/dk/dv 237 us vs Triton 284 us). Reaching 0.95 would need the rounding folded into the projections' GEMM epilogues (the cuBLAS calls of the module), which is outside this kernel's reach.

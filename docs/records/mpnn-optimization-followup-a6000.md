# MPNN optimization follow-up on A6000 (2026-09-13)

After the exact B8/L8192/K48/D128 cache build, two structural backward changes were tested without modifying production source or tuning data. All measurements ran on allocated A6000 devices on gpu04.

Full-model comparisons reuse the existing attribution harness: native BF16 parameters, gradients and AdamW states, FP32 LayerNorm parameters, no autocast, no checkpointing, dropout 0.25, witnessed fullgraph compilation, training CUDA Graph disabled. Time includes forward, loss, backward, eager AdamW and zero_grad. Seven timing samples per run and five finite-gradient/memory validation steps. Inputs are seeded synthetic proteins, not dataset throughput.

## Whole-model comparisons

Each baseline/candidate group uses the same GPU UUID. Chunk order is baseline, 524288 rows, 3145728 rows, baseline; cast order is A/B/B/A. Baselines and repeated cast candidates are the median of their per-run medians. Different experiments can use different physical A6000 devices.

| Policy | Candidate | Baseline ms | Candidate ms | Time reduction | Peak GiB before → after |
| --- | --- | ---: | ---: | ---: | ---: |
| all_compute | half | 402.56 | 401.30 | 0.31% | 24.943 → 25.005 |
| all_compute | whole | 402.56 | 401.56 | 0.25% | 24.943 → 25.630 |
| all_compute | fusedcast | 404.12 | 405.01 | -0.22% | 24.943 → 24.943 |
| current | half | 414.68 | 413.55 | 0.27% | 21.427 → 21.490 |
| current | whole | 414.68 | 413.01 | 0.40% | 21.427 → 22.114 |
| current | fusedcast | 417.94 | 417.91 | 0.01% | 21.427 → 21.427 |

## Chunk experiment

The current projection backward handles 3145728 edge rows in twelve 262144-row chunks. The reusable BF16 GELU buffer limits temporary activation memory; partial BF16 weight-gradient GEMMs accumulate in FP32. Larger chunks reduce launches but increase temporary memory and change partial-GEMM rounding.

| Chunk rows | Isolated dX+dW ms | Incremental peak GiB | dW relative L2 vs baseline |
| ---: | ---: | ---: | ---: |
| 262144 | 6.959 | 0.813 | 0.000000 |
| 524288 | 6.737 | 0.875 | 0.002860 |
| 1048576 | 6.769 | 1.000 | 0.005531 |
| 3145728 | 6.639 | 1.500 | 0.005519 |
| 262144 | 6.967 | 0.813 | 0.000000 |

All isolated dX outputs were bitwise equal. The isolated numerical check uses seeded BF16 random inputs; it is not a training-convergence test. Larger chunks were exploratory runtime overrides, not new full-grid cache builds: unseen buckets used the ordinary bounded 24-candidate fallback, and other shapes may reuse a populated logical bucket. Warmup is outside timing. Zero reported runtime cache misses alone does not prove exact physical-profile tuning for these experimental shapes. Existing production-shape cache entries were preserved.

## Cast experiment

The original loop converts each BF16 partial dW matrix to FP32 before adding it into its FP32 accumulator. The experiment keeps the first conversion and lets subsequent FP32 in-place adds consume the BF16 partial directly. It removes eleven standalone conversions per twelve-chunk call, preserves chunk sizes and GEMM precision, and leaves all Triton launch workloads unchanged. A separately registered temporary custom op keeps fullgraph compilation working.

The isolated baseline and candidate dX and dW outputs were bitwise equal. All cast whole-model runs had zero runtime cache misses. The measured gain must be judged against the repeated whole-model baseline, not inferred from removed launch counts.

## Scope and artifacts

No production kernel, default policy or tuning cache was changed by these experiments. The earlier native-BF16 work and 14-operation/40-profile cache remain intact. This is a bounded optimization assessment for this A6000 workload, not a claim of global optimality or A5000 validation.

Raw results and runnable experimental scripts are under `.scratch/mpnn-native-bf16/chunk-study-20260913`; the JSON record also embeds those scripts and every full-model sample, GPU UUID, source identity, precision/compile evidence and memory validation step.

[Full record](mpnn-optimization-followup-a6000.json). [Validated cache and module comparisons](mpnn-target-cache-a6000.md).

## Decision

Keep the validated production implementation and caches. Larger chunks produced only 0.25–0.40% observed whole-step reductions while adding 0.0625–0.6874 GiB of peak allocation and changing weight-gradient rounding. Expanded candidates had one run per policy and were not exhaustively tuned; these results do not establish a stable benefit sufficient to replace the validated implementation. The cast A/B/B/A comparison showed no meaningful gain (all-compute 0.22% slower, current 0.008% faster).

This concludes the bounded tuning pass for A6000 B8/L8192. Wider decoder gather/mask/reduction fusion remains an untested structural direction. The previously validated encoder advantage and whole-model speed/memory tradeoff remain the basis for choosing compute versus current policy; this follow-up does not replace the prior matched PyTorch comparison.

A subsequent [training CUDA Graph ON/OFF comparison](mpnn-training-cudagraph-a6000.md)
validated replay semantics and measured only 0.05–0.32% whole-step reductions at
this same B8/L8192 workload. The training graph-OFF default remains intact.

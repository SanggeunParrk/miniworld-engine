# MPNN training CUDA Graph comparison on A6000 (2026-09-13)

Explicit opt-in experiment. Production training graph defaults, kernels and tuning caches were not changed. The standard module harness forbids dropout training graphs; this isolated experiment uses the existing fullgraph compile witness and shared `bench_time` timer, with additional graph-specific training validation.

B8/L8192/K48/D128, three encoder and three decoder layers. Native BF16 ordinary parameters, gradients and AdamW moments; FP32 LayerNorm parameters. No autocast or checkpointing, dropout 0.25. Resident seeded synthetic inputs with fixed shapes. Compile warmup and graph capture are excluded from steady-state timing.

Graph ON captures forward + cross-entropy loss + backward. Both modes execute the same eager AdamW outside the graph; the table includes its cost. OFF clears gradient references each step; ON captures fresh-gradient writes and validates overwrite across replays. It does not measure a captured optimizer or data transfer.

## Results

Each policy runs OFF/ON/ON/OFF on one A6000 UUID, in fresh processes. A cell is the median of two per-run medians, each from seven timing samples. Policies run on separate A6000 GPUs, so ON/OFF changes within each row are the matched comparison. RNG is reset to the same seed immediately before optimizer warmup after mode-specific validation.

| Policy | Graph OFF ms/step | Graph ON ms/step | OFF / ON | Time reduction | Peak reserved GiB OFF → ON |
| --- | ---: | ---: | ---: | ---: | ---: |
| pytorch | 448.69 | 447.25 | 1.003x | 0.32% | 34.750 → 34.812 |
| all_compute | 406.22 | 406.04 | 1.000x | 0.05% | 25.844 → 25.887 |
| current | 417.44 | 416.15 | 1.003x | 0.31% | 22.348 → 21.746 |

`reserved` includes the CUDA Graph private pool. Allocator counters during replay alone omit the captured temporary-allocation peak: a ~0.04 GiB replay allocated counter does not mean this model trains in that much memory. The JSON includes capture-time allocation peaks, graph pool/reserved memory, and device-used memory after each validation step. Driver/CUDA context overhead is outside PyTorch reserved memory.

## Training correctness

All ON runs compared compiled OFF and graph replay with unchanged weights and identical seeded dropout. Loss, gradient relative error, repeated-seed replay, advancing dropout RNG, writable resident label buffers, stable gradient addresses, finite native gradient dtypes, BF16 AdamW moments, optimizer step counter, and actual parameter updates were checked. Five additional training steps per run checked finite gradients and memory.

| Policy | Maximum ON/OFF gradient relative L2 | Maximum reseed replay relative L2 |
| --- | ---: | ---: |
| pytorch | 0.00059963241 | 0.00059978618 |
| all_compute | 1.5166326e-08 | 1.447487e-08 |
| current | 1.5156381e-08 | 1.5150312e-08 |

All final runs had zero runtime cache misses and witnessed fullgraph execution. This verifies the sampled training steps and graph replay semantics; it is not a convergence study or validation of variable-shape graph management.

## Artifacts

Final Slurm array: 1682429. Pilot: 1682422. Preparation arrays 1682423 and 1682426 were intentionally stopped; their partial results are excluded. The latter exposed a benchmark warmup stream/loss-reference issue, fixed before final measurements. Final logs are asserted free of that stream mismatch warning.

Raw data and scripts: `.scratch/mpnn-native-bf16/graph-study-20260913/final` and its parent. The JSON embeds the final runner and Slurm script, all timing samples, compile evidence, GPU UUIDs, source identities, graph numerical checks and memory records.

[Complete record](mpnn-training-cudagraph-a6000.json). [Graph-OFF cache and module comparison](mpnn-target-cache-a6000.md).

## Decision

Retain the existing training graph-OFF default for this workload. All three observed step-time reductions were at most 0.32%; the compute policy changed by only 0.05%. Replay is functional and training checks passed, but these measurements do not establish a material speed advantage. The small effect suggests host dispatch is not the dominant whole-step cost at this size. Smaller shapes and capturing the optimizer were not tested.

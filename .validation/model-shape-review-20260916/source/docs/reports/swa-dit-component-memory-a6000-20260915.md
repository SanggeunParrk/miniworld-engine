# SWA DiT component memory audit — A6000

## Findings

In this grid, modulation and residual-gate replacements reduce full-block peak memory even though their isolated operations do not save memory. Thus their slower block timings from the latency audit come with a memory benefit. Q/K norm+RoPE and SwiGLU save memory both in isolation and in the block. Attention sigmoid gate shows no memory saving.

Individual savings cannot be added to predict the all-five result. The full-block peak and tensor lifetimes change with the combination of operations.

## Measurement definition

All values are **incremental peak allocated memory during a forward+backward step**. The standard harness measures `max_memory_allocated - memory_allocated_at_step_start`. It uses 3 warmup steps and the median of 10 measured steps per process, then this report aggregates 3 independent processes. The step clears previous gradients internally, so the starting allocation can include the previous step's gradients. Inputs, weights and other already-live allocations are excluded; these numbers are not total GPU/process peak, reserved memory, or inference memory.

For each pair, PyTorch and engine have identical seeded inputs, weights and upstream gradients. Both use fullgraph compilation, AOT backward and the same FA2 attention core. One-at-a-time block replacements preserve all other operations. A lower isolated peak does not necessarily lower the full block peak, and replacement savings must not be summed: the peak can occur at a different point in the step.

- Slurm job `1699247`, `NVIDIA RTX A6000`, `GPU-4c79f963-f619-9df0-44dd-11bdd6e3b3c7`, host `gpu01`; one allocated GPU throughout.
- Stable source hash `ccc77e767aba44de90a687fd4c76e2b187fae1a5d5e26617d1f52a91b80cb94f`.
- BF16 inputs/weights, TF32 disabled, A=48, B=1, atoms3072/4096, D128, H4, head dimension32, SwiGLU hidden256; one block, half-window64, mask probability0.125, no dropout, CUDA Graph OFF.
- Active modulation gates (normal weight std0.01); no optimizer step. This matches the component latency audit workload.
- 33 benchmark processes and 132 successful rows, each with output and gradient comparison against compiled PyTorch. Full-block reference uses the original production PyTorch SWADiTBlock. Maximum output/gradient relative Frobenius errors: 0.004160 / 0.008227.
- Current runtime dispatch and bounded fallback tuning during warmup; no kernel or dispatch modifications for this measurement. FP32 and H100 are outside this audit.

## Isolated operation memory

Positive savings mean lower incremental peak. Modulation measures one branch with SiLU conditioning and three projections; the full PyTorch block combines all six projections into one GEMM.

| Operation | Atoms | PyTorch MiB | Engine MiB | Saved MiB | Saved % |
|---|---:|---:|---:|---:|---:|
| RMSNorm + modulation | 3072 | 180.000 | 180.375 | -0.375 | -0.21% |
| RMSNorm + modulation | 4096 | 240.000 | 240.563 | -0.563 | -0.23% |
| Q/K norm + RoPE | 3072 | 112.500 | 72.000 | +40.500 | +36.00% |
| Q/K norm + RoPE | 4096 | 150.000 | 96.000 | +54.000 | +36.00% |
| SwiGLU FFN | 3072 | 359.875 | 287.812 | +72.062 | +20.02% |
| SwiGLU FFN | 4096 | 479.875 | 383.812 | +96.062 | +20.02% |
| Residual gate | 3072 | 36.000 | 36.000 | +0.000 | +0.00% |
| Residual gate | 4096 | 48.000 | 48.000 | +0.000 | +0.00% |
| Attention sigmoid gate | 3072 | 36.000 | 36.000 | +0.000 | +0.00% |
| Attention sigmoid gate | 4096 | 48.000 | 48.000 | +0.000 | +0.00% |

## Whole-block memory with one replacement

Each row replaces only the named operation in the PyTorch block (both branches for modulation/residual). "All five" replaces all five operations.

| Replacement | Atoms | PyTorch MiB | Modified block MiB | Saved MiB | Saved % |
|---|---:|---:|---:|---:|---:|
| RMSNorm + modulation | 3072 | 1085.156 | 942.031 | +143.125 | +13.19% |
| RMSNorm + modulation | 4096 | 1447.031 | 1255.031 | +192.000 | +13.27% |
| Q/K norm + RoPE | 3072 | 1085.156 | 1044.656 | +40.500 | +3.73% |
| Q/K norm + RoPE | 4096 | 1447.031 | 1393.031 | +54.000 | +3.73% |
| SwiGLU FFN | 3072 | 1085.156 | 1050.594 | +34.562 | +3.19% |
| SwiGLU FFN | 4096 | 1447.031 | 1399.844 | +47.188 | +3.26% |
| Residual gate | 3072 | 1085.156 | 1017.407 | +67.749 | +6.24% |
| Residual gate | 4096 | 1447.031 | 1355.595 | +91.437 | +6.32% |
| Attention sigmoid gate | 3072 | 1085.156 | 1086.031 | -0.875 | -0.08% |
| Attention sigmoid gate | 4096 | 1447.031 | 1447.031 | +0.000 | +0.00% |
| All five | 3072 | 1085.156 | 864.594 | +220.562 | +20.33% |
| All five | 4096 | 1447.031 | 1152.969 | +294.062 | +20.32% |

![Memory comparison](../../benchmarks/modules/swa_dit/artifacts/component_memory_20260915/component_memory.svg)

[All measurements](../../benchmarks/modules/swa_dit/artifacts/component_memory_20260915/measurements.csv) · [Aggregates with min/max](../../benchmarks/modules/swa_dit/artifacts/component_memory_20260915/summary.csv) · [Paired savings](../../benchmarks/modules/swa_dit/artifacts/component_memory_20260915/comparisons.csv) · [Original CSV manifest](../../benchmarks/modules/swa_dit/artifacts/component_memory_20260915/raw_files.json)

## Reproduce

Use the commands in the [component latency audit](swa-dit-component-audit-a6000-20260915.md#reproduce), replacing `metric=time` with `metric=memory`. Keep `cudagraph=disabled`, `compile=true`, active gates and the same shape grid; run three independent process pairs per case in one Slurm GPU allocation.

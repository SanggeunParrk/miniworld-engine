# SWA DiT fullgraph training — A6000

FA2 backward now uses the same fixed-capacity scatter/gather packing as forward. True per-sequence lengths exclude trailing storage from attention. Removing `nonzero` and the host maximum-length read lets AOTAutograd trace the recomputation and native FA2 backward custom op. No new compiler-disable boundary or fallback was added.

## Validation

- Slurm GPU job `1698624`: **212 tests passed** in 408.33 s (SWA GPU numerics, compile tests, and benchmark contracts). [Validation log](../../benchmarks/modules/swa_dit/artifacts/esmfold2_fullgraph_20260915/validation_1698624.log). Changed-file Ruff and project-environment type checks passed.
- Strict `torch.compile(..., fullgraph=True)` forward and backward: FP32/BF16, zero and active gates; input, conditioning, and all parameter gradients checked.
- Independent legacy FA2 packing reference: front masks, holes, empty rows, all-empty batches, and Q-only gradients. Captured one forward and one backward FX graph across changed masks; backward contains the native FA2 backward operation and no `nonzero` or host scalar read.
- SWA DiT benchmark now requires `module_forward:fullgraph`; any graph break fails the compiled benchmark.
- FA4/H100 was not available on this cluster and was not GPU-validated.

## Measurement

- Slurm job `1698632`, `NVIDIA RTX A6000`, `GPU-c6b0efd9-3781-ef92-b83a-055284404408`, host `gpu01`.
- Source hash `0602b29dee0e4fc76e22d3fb2b539b012db7559b90a33e3fc64d9a72cb0c410e`.
- Standard `benchmarks/runners/bench.py`: BF16 inputs and weights (`bf16-mixed` label), TF32 off, B=1, A=48, 1 block, width128, 4 heads, half-window64, mask probability0.125, no dropout.
- Training includes forward and backward, fresh gradients per step, fullgraph compilation with AOT backward, CUDA Graph OFF. Default zero-initialized gates; active-gate numerical validation is separate.
- Each result is the median of 3 independent processes after warmup; min/max and run IDs are retained. Both implementations ran on the same allocated GPU.
- Memory is incremental peak allocated MiB during the step, excluding already-live allocations; it is not total process peak.
- Runtime cache misses used the standard bounded fallback tuning during warmup. These are not measurements after rebuilding the complete tuned cache.
- The earlier partial-compile run used another physical A6000 (`13b61e95...`); its absolute times are historical context rather than a controlled same-card before/after comparison. Inference was not rerun for this backward fix.

| Tokens / atoms | PyTorch ms | Engine ms | Speedup | PyTorch extra peak MiB | Engine extra peak MiB |
|---|---:|---:|---:|---:|---:|
| 384 / 3072 | 14.7420 | 14.6883 | 1.004x | 1085.16 | 864.59 |
| 512 / 4096 | 19.4140 | 18.6778 | 1.039x | 1447.03 | 1152.97 |

[Aggregated CSV](../../benchmarks/modules/swa_dit/artifacts/esmfold2_fullgraph_20260915/summary.csv) · [Raw manifest](../../benchmarks/modules/swa_dit/artifacts/esmfold2_fullgraph_20260915/raw_files.json)

![Fullgraph training latency](../../benchmarks/modules/swa_dit/artifacts/esmfold2_fullgraph_20260915/latency.svg)

Previous architecture-corrected inference and partial-compile training measurements: [earlier report](swa-dit-esmfold2-a6000-20260915.md).

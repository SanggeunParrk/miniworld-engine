# ESMFold2 SWA DiT — A6000 rerun

> Training measurements below are superseded by the [fullgraph backward fix and rerun](swa-dit-fullgraph-a6000-20260915.md). The inference measurements remain from this original run.

Measured the corrected ESMFold2 block through the standard `benchmarks/runners/bench.py` harness.

- Slurm job: `1698091`; GPU/host: `('13b61e95-69d6-f2ca-fe4c-e4593d53b746', 'gpu01')`.
- Source hash: `1752dfed2d8bef6e4ac21b91cd31e9b199358d6aa9f9cc8180d3f1bc42082490`.
- BF16 inputs and weights (`bf16-mixed` harness label), TF32 disabled; B=1, 1 block, width 128, 4 heads, half-window 64.
- Token lengths 384/512 correspond to atom lengths 3072/4096. Augmentation: inference 5, training 48. Front-packed padding probability 0.125; no dropout.
- Both implementations use `torch.compile` and FlashAttention-2. Inference timing uses manual CUDA Graph replay; training timing includes forward + backward with fresh gradients and no graph.
- Compilation follows the harness `module_forward:partial_allowed` policy. The FA2 backward path triggers `aten.nonzero` graph breaks in training; these timings include that current execution path, not a promised full-graph implementation.
- Each cell is the median of 3 independent benchmark processes. CSV includes min/max and run IDs.
- Memory is the harness incremental peak allocated memory during a step, in MiB, with CUDA Graphs disabled. It excludes allocations already live before the step; it is not total model/process peak memory.
- Default zero-initialized adaLN gates are used, as in the production constructor. Active-gate output/gradient correctness was verified separately before this benchmark.
- The changed kernel source invalidated some tuned cache entries; the runtime used its normal fallback candidate search during warmup. These are current runtime results, not results after a full cache rebuild.
- Previous hybrid-block results are archived beside the new raw data and are not a valid performance baseline for this architecture.

| Mode | Tokens / atoms | PyTorch ms | Engine ms | Speedup | PyTorch extra peak MiB | Engine extra peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| inference | 384 / 3072 | 0.7076 | 0.5980 | 1.183x | 79.73 | 64.85 |
| inference | 512 / 4096 | 0.9236 | 0.8059 | 1.146x | 106.47 | 86.47 |
| training | 384 / 3072 | 26.8882 | 19.5482 | 1.375x | 1445.16 | 828.59 |
| training | 512 / 4096 | 35.6434 | 25.1290 | 1.418x | 1927.91 | 1104.97 |

[Aggregated CSV](../../benchmarks/modules/swa_dit/artifacts/esmfold2_rerun_20260915/summary.csv) · [Raw CSV manifest](../../benchmarks/modules/swa_dit/artifacts/esmfold2_rerun_20260915/raw_files.json)

![Latency and speedup](../../benchmarks/modules/swa_dit/artifacts/esmfold2_rerun_20260915/latency.svg)

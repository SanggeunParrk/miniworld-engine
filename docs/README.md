# docs

Pages are written for a reader using the kernels, building a cache or reading a benchmark.
Start with the repo [README](../README.md) for the layout and quickstart.

| folder | what it answers |
|---|---|
| [status/](status/README.md) | **per GPU: which op is finished for which shapes** (the judgement column is the maintainer's) — [H100](status/h100.md) · [B200](status/b200.md) · [A100](status/a100.md) |
| [gpus/](gpus/README.md) | how to run on each GPU cluster: partitions, QoS, env setup, GPU-specific pitfalls — [H100](gpus/h100.md) ([dispatch](gpus/h100-dispatch.md)) · [B200](gpus/b200.md) · [A100](gpus/a100.md) · [A6000/A5000](gpus/ampere-workstation.md) |
| [getting-started/](getting-started/) | [what has run where](getting-started/supported.md) · [troubleshooting](getting-started/troubleshooting.md) · [reproducing a report](getting-started/reproducing-a-report.md) |
| [kernels/](kernels/) | per-op design notes ([TriMul module](kernels/triangle-multiplication-module.md), [trimul_inproj](kernels/trimul-inproj.md), [tm1](kernels/tm1.md)/[tm2](kernels/tm2.md), [triangle attention](kernels/triangle-attention.md), [LayerNorm](kernels/layernorm.md), [LN+Linear](kernels/layernorm-linear.md), [RMSNorm-AdaMod](kernels/rmsnorm-adamod.md), [bias-only attention](kernels/bias-only-attention.md)), [numeric thresholds](kernels/thresholds.md), [lab-notebook convention](kernels/lab-notebooks.md) |
| [autotune/](autotune/) | the dispatch cache ([policy](autotune/dispatch-cache.md)), [key convention](autotune/autotune-key.md), [grid sweep](autotune/grid-sweep.md), [L2 swizzle](autotune/l2-swizzle.md), [training shape policy](autotune/training-shape-policy.md), [sweep-grid page](autotune/sweep-grid.html) |
| [benchmarks/](benchmarks/README.md) | harness conventions, [cautions](benchmarks/cautions.md), [measurement contract](benchmarks/measurement-contract.md), [results](benchmarks/results.md) |
| [anthropic/](anthropic/) | integration of Anthropic's published kernels: [integration](anthropic/integration.md), [payload](anthropic/trimul-payload.md) |
| [standards/](standards/) | [library](standards/library-standards.md) and [product](standards/product-standards.md) standards, [naming](standards/naming.md), [project direction](standards/project-direction.md) |
| [releases/](releases/) | what each version changed and its qualification status |
| [records/](records/README.md) | dated measurements, audits and verdicts — evidence, never edited after the fact. Older records, design proposals and analyses are at tag `archive/docs-20260928` |
| [assets/](assets/README.md) | figures |

Rules: a page reflects the current source (records are the exception); a performance number
names its GPU, baseline and timing mode; per-GPU facts go in `status/` or `gpus/`, not here.

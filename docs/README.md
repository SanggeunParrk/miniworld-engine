# docs

| | |
|---|---|
| [gpus/](gpus/README.md) | one page per GPU: how to run there, and **which op is finished for which shapes** (judgement column: the maintainer's) — [H100](gpus/h100.md) ([dispatch](gpus/h100-dispatch.md)) · [B200](gpus/b200.md) · [A100](gpus/a100.md) · [A6000/A5000](gpus/ampere-workstation.md) · [what has run where](gpus/supported.md) · [troubleshooting](gpus/troubleshooting.md) |
| [kernels/](kernels/) | per-op notes — [TriMul module](kernels/triangle-multiplication-module.md), [trimul_inproj](kernels/trimul-inproj.md), [tm1](kernels/tm1.md)/[tm2](kernels/tm2.md), [triangle attention](kernels/triangle-attention.md), [LayerNorm](kernels/layernorm.md), [LN+Linear](kernels/layernorm-linear.md), [RMSNorm-AdaMod](kernels/rmsnorm-adamod.md), [bias-only attention](kernels/bias-only-attention.md), [Anthropic kernels](kernels/anthropic-integration.md) ([payload](kernels/anthropic-trimul-payload.md)); tuning — [dispatch cache](kernels/autotune-dispatch-cache.md), [key](kernels/autotune-key.md), [grid sweep](kernels/autotune-grid-sweep.md), [L2 swizzle](kernels/autotune-l2-swizzle.md), [training shapes](kernels/autotune-training-shape-policy.md), [sweep page](kernels/autotune-sweep-grid.html), [thresholds](kernels/thresholds.md); writing kernels — [contributing](kernels/contributing.md), [naming](kernels/naming.md), [lab notebooks](kernels/lab-notebooks.md) |
| [CHANGELOG.md](CHANGELOG.md) | what each version changed |
| [standards.md](standards.md) | project direction, library and product standards |

Benchmark docs live beside the harness in [`benchmarks/`](../benchmarks/README.md). Older docs,
records and release pages are in git (README, "Research history").

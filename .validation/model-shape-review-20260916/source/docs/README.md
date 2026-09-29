# docs

Index of the written docs. Each page is for a *consumer* — someone using the kernels, building
the cache, or reading a benchmark — not a changelog. Start with the repo [`README.md`](../README.md)
for the layout and quickstart; the pages here go deeper.

## Start here

| page | what it answers |
|---|---|
| [supported.md](supported.md) | which cards this has actually been run on |
| [reproducing-a-report.md](reproducing-a-report.md) | reproduce a committed benchmark on another machine |
| [troubleshooting.md](troubleshooting.md) | what to do when a step does not do what the README says |

## Benchmarking

| page | what it answers |
|---|---|
| [benchmarks.md](benchmarks.md) | the benchmark conventions: harness, CSV-is-truth, results/artifacts/plots layout |
| [benchmarking-cautions.md](benchmarking-cautions.md) | traps: compile vs cudagraph vs reduce-overhead, graph-break kernels, mask cost |
| [operations/dispatch-cache.md](operations/dispatch-cache.md) | the runtime autotune/dispatch cache: what it is, how it is built, the policy |

## Standards

| page | what it answers |
|---|---|
| [library-standards.md](library-standards.md) | what a tier-1 kernel library owes its consumers |
| [product-standards.md](product-standards.md) | what a product owes someone who is not its author |

## Kernel notes (`kernels/`)

Conventions that span kernels:

| page | what it answers |
|---|---|
| [kernels/naming.md](kernels/naming.md) | the kernel naming rules |
| [kernels/autotune-key.md](kernels/autotune-key.md) | the autotune-key convention (what a cache entry is keyed on) |
| [kernels/thresholds.md](kernels/thresholds.md) | every numeric literal that decides something, and why |
| [kernels/grid-sweep.md](kernels/grid-sweep.md) | the full config grid sweep — design and preparation |
| [kernels/l2-swizzle.md](kernels/l2-swizzle.md) | `GROUP_M`, the L2-swizzle axis, and which kernels skip it |
| [kernels/cute-autotune-and-config-pinning.md](kernels/cute-autotune-and-config-pinning.md) | the cute autotune bypass and configs pinned for correctness |

Per kernel:

| kernel | page |
|---|---|
| tm1 / tm2 (gated GEMMs) | [tm1.md](kernels/tm1.md) · [tm2.md](kernels/tm2.md) |
| layernorm / layernorm_linear | [layernorm.md](kernels/layernorm.md) · [layernorm-linear.md](kernels/layernorm-linear.md) |
| rmsnorm_adamod (adaLN ladder) | [rmsnorm-adamod.md](kernels/rmsnorm-adamod.md) |
| triangle attention | [triangle-attention.md](kernels/triangle-attention.md) |
| trimul (module / inproj) | [triangle-multiplication-module.md](kernels/triangle-multiplication-module.md) · [trimul-inproj.md](kernels/trimul-inproj.md) |
| bias-only attention | [bias-only-attention.md](kernels/bias-only-attention.md) |

## Design proposals (`design/`)

Forward-looking plans, not yet the shipped path:

- [design/residual-fusion.md](design/residual-fusion.md) — residual fusion follow-ups
- [design/layernorm-linear-fused-dgrad-lnbwd.md](design/layernorm-linear-fused-dgrad-lnbwd.md) — fused dgrad GEMM + LN-backward
- [design/layernorm-linear-warp-specialized-stats.md](design/layernorm-linear-warp-specialized-stats.md) — warp-specialized LN stats
- [design/quack-0.5.0-cute-port.md](design/quack-0.5.0-cute-port.md) — port the cute backend to quack 0.5.0

## Reports (`reports/`)

Rendered benchmark reports, one per GPU, generated from the tracked results tables by
`benchmarks/runners/report_gpu.py` (per-module sweep figures, a summary table and a summary
figure). Regenerate after the tables change; do not hand-edit the numbers.

| page | what it answers |
|---|---|
| [reports/a6000-module-sweeps.md](reports/a6000-module-sweeps.md) | every module's L and d_pair sweep on the RTX A6000, inference and training, against compiled PyTorch and the vendor baselines |
| [reports/swa-dit-component-memory-a6000-20260915.md](reports/swa-dit-component-memory-a6000-20260915.md) | SWA operation memory and one-at-a-time block peak-memory savings |
| [reports/swa-dit-component-audit-a6000-20260915.md](reports/swa-dit-component-audit-a6000-20260915.md) | SWA operation-by-operation speed and full-block substitution audit |
| [reports/swa-dit-fullgraph-a6000-20260915.md](reports/swa-dit-fullgraph-a6000-20260915.md) | FA2 fullgraph backward fix and A6000 training rerun |
| [reports/swa-dit-esmfold2-a6000-20260915.md](reports/swa-dit-esmfold2-a6000-20260915.md) | corrected ESMFold2 SWA DiT: A6000 time and incremental peak memory, three repetitions |

## Records (`records/`)

Measurement records — a number taken at a point in time, kept for evidence. See
[records/README.md](records/README.md) for the full list (latency tables, tiling/naming/cache audits).

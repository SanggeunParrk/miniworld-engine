# Benchmark execution contract

The official path is `miniworld-engine bench` → `runners/bench.py` → CSV schema 2 → `runners/plot_csv.py`. A requested option is not evidence that it ran. This contract was introduced after the September 2026 audit found that 16 of 17 kernel benchmark targets ignored `compile=True` while recording it as true.

## What a successful row establishes

- The requested implementation produced a finite timing or memory measurement and observable finite output tensors. Training checks include the forward result and available input/parameter gradients.
- `compiled` comes from an executed Inductor executable observed through the same callable that is subsequently timed. Calling `torch.compile`, tracing a graph, or compiling only a reference is insufficient. A requested compile with no observed executable becomes `unsupported`, with no valid measurement value.
- `compile_scope` distinguishes module forward from standalone callable compilation and records `fullgraph` versus `partial_allowed`. Module compilation allows graph breaks. It does **not** establish that an entire forward/backward step was fused or that custom CUDA/Triton operators were lowered by Inductor.
- `cudagraph` records the timer actually used. Manual capture includes a replay/output consistency check before timing. An inference request for `graphed` currently uses manual capture and records `manual`; `graphed` training is explicitly unsupported. `auto` selects manual for inference timing, including SWA/FlashAttention targets, and disabled for training and memory. Capture failures are reported without silently switching graphs off. A6000/FA2 SWA no-grad capture/replay is GPU-tested; those tests do not verify FA4 GPU capture.
- `measurement_scope` distinguishes forward, backward-only, and forward+backward. Backward-only autograd over a prebuilt eager graph rejects compile requests rather than claiming compiled backward.
- Actual input and parameter dtypes and captured input tensor shapes are stored. A precision request alone does not determine these values. Unsupported FP32 backend requests are rejected explicitly. `tokens` and `batch_size` remain blank where the harness cannot establish their semantics reliably.
- Inference sets modules to eval mode and disables autograd. Training clears the requested gradients inside each step, including graph capture, so replay uses the same fresh-gradient semantics as eager timing. This benchmark measures forward/backward, without an optimizer update.

Request fields (`compile_requested`, `cudagraph_requested`, `mode_requested`, `n_layers_requested`) are separate from observed fields. Failures and unsupported combinations are preserved as rows and produce a nonzero process exit. Empty sweeps, NaN timing values, and missing execution evidence cannot yield a successful run.

## Compilation, timing, and validation

The compile backend is Inductor with `dynamic=False`. Inductor's internal CUDA graphs are disabled so that the harness controls graph capture explicitly. Compilation, correctness probes, warmup/capture setup, and first replay checks happen before the timing window. The ordinary time helper uses Triton's `do_bench` with warmup=10 ms, rep=100 ms, and median/p20/p80 quantiles. Audit smoke/matrix wrappers deliberately shorten this interval and are not publishable speed comparisons.

The compile witness adds a small Python wrapper/ContextVar lookup to compiled execution. A separate A6000 BF16 L128 TriangleMultiplication training A/B found 0% PyTorch and 0.18% MiniWorld median change when bypassing only the witness body (three repetitions, outputs and gradients checked). This does not quantify every target or shape. Manual graph replay avoids executing that Python wrapper per replay. Do not describe the no-graph numbers as uninstrumented production latency.

Pure kernel compiled output is compared against the exact eager callable. Graph replay is compared against its pre-capture result, including available training gradients. Relative Frobenius limits are 2% for BF16/FP16 and 0.01% for FP32. These are **execution-consistency** checks, not substitutes for per-kernel FP32-reference accuracy gates. Module accuracy fields remain blank where no independent mathematical reference comparison was performed. A passing execution-contract row must not be reported as complete numerical certification.

Memory mode measures peak allocated-memory growth of an eager/compiled non-graph step. Graph-memory requests are unsupported. It does not claim total process memory or compilation memory.

## Provenance and result reuse

Every run records a unique `run_id`, full configuration hash, source hash, framework versions, device name, and a `.run.json` configuration/settings sidecar. The run filename includes hashes and a unique ID to prevent silent overwrites between repetitions or omitted filename dimensions. Source identity is checked before and after each measurement; changing sources mid-run fails that row.

The plot reader rejects mixed requested conditions, duplicate implementation/x-coordinate rows, and unverifiable legacy schema by default. A deliberate historical plot requires `--allow-legacy-unverified` and is visibly marked. This does not delete or declare all old measurements numerically wrong.

Autotune candidate caches are distinct from performance CSVs. Changing benchmark/reporting code does not by itself require rebuilding unchanged kernel tiles. Module/dispatch source changes do require refreshing the derived reachability plan and checking existing cache identities and required keys. Never bypass those checks by merely restamping stale derivation evidence.

## Audit coverage and limitations

The audit inventories all 17 official kernel targets (52 implementation branches), 11 module targets, CLI dispatch, plotting, five standalone benchmark/compile-diagnostic runners, 27 root Slurm launchers, scratch probes, and kernel development-note entrypoints. The A6000 execution matrix uses isolated GPU allocations and subprocesses. CPU regression checks inject no-op compilation, reference-only compilation, non-finite output, corrupted replay, overlapping GPU scheduling, and shell failures.

A6000 kernel contract matrix: 228 cases, 181 successful executions and 47 explicit unsupported outcomes, no unresolved failures. Unsupported cases comprise 18 architecture restrictions, 16 compiled prebuilt-autograd backward requests, 12 unsupported FP32 requests, and one K-tiled launcher without an opaque compile entry. This matrix covers every declared implementation; it is not a full Cartesian product of every shape, dtype, mask, and graph option.

The full per-target/implementation tables and raw allocation/CSV evidence are in the workspace audit reports. Blackwell/Hopper-specific numerical execution and external-checkout developer probes were not run on another GPU. Retired probes that compared identical current implementations or included module construction in a fallback timer now fail with an explanation.

Remaining production-kernel numerical issues and the separately tracked incomplete A6000 autotune keys are not made complete by these benchmark-harness checks. Keep those statuses separate from execution-contract coverage.

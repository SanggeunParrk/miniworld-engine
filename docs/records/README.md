# Records

Dated findings. Each describes what was true when it ran and is **not** updated when the code
changes — the same rule as `src/miniworld_engine/kernels/<family>/notes/`, for the ones that belong
to no single kernel.

| | what it records |
|---|---|
| [mpnn-b8-l8192-memory-fit.md](mpnn-b8-l8192-memory-fit.md) | B8/L8192 full-model training with checkpoint API forbidden: capped-A6000 memory-fit proxy, two AdamW steps, and compute-policy capacity limits; A5000 direct validation not performed. |
| [mpnn-batch-accumulation-a6000.md](mpnn-batch-accumulation-a6000.md) | Fixed-effective-batch A6000 experiment: full ProteinMPNN and node-message throughput, real accumulation, peak memory, and compiled-gradient parity. |
| [mpnn-node-compute-a6000.md](mpnn-node-compute-a6000.md) | Saved-projection node-message training, A6000 latency and memory tradeoff, tail-mask fix and build-driver coverage. |
| [mpnn-a6000-optimization.md](mpnn-a6000-optimization.md) | MPNN cold-autotune gradient and inference compile fixes, A6000 seven-family comparison, before/after performance and saved-storage tradeoffs. |
| [a6000-production-audit.md](a6000-production-audit.md) | Current A6000 final audit: corrected DiT dispatch, strict build/shard checks, package validation and repeated measurements. |
| [a6000-atom-dit-4096.md](a6000-atom-dit-4096.md) | Historical ordinary pair-bias atom DiT at 4096 atoms (superseded above), A5 graph inference and A48 no-graph training on one A6000. |
| [workspace-artifact-relocation.md](workspace-artifact-relocation.md) | Team-GM cleanup: 1,067 MiniWorld work entries relocated into engine scratch; provenance paths and movement manifest. |
| [a6000-swa-gate-output-fusion.md](a6000-swa-gate-output-fusion.md) | SWA output GEMM/gate fusion: controlled inference/training experiment, cached inference dispatch, and build-plan status. |
| [a6000-dit-qk-fusion-cache-audit.md](a6000-dit-qk-fusion-cache-audit.md) | DiT after the AdaLN repair, paired Q/K RMSNorm+RoPE forward/backward, production cache wiring and workload provenance. |
| [a6000-swa-inference-attribution.md](a6000-swa-inference-attribution.md) | SWA inference overhead traced to Q/K copies and separate RMSNorm/RoPE launches; live RoPE cache coverage verified. |
| [a6000-adaln-ct-training-dispatch.md](a6000-adaln-ct-training-dispatch.md) | A6000 training slowdown traced to AdaLN forward dispatch, component-swap controls, and SWA backend clarification. |
| [build-all-production-contract.md](build-all-production-contract.md) | shared A5/A48 shapes, automatic preflight/plan refresh, A6000 1,007/1,007 coverage, and explicit native-backend gaps. |
| [a6000-l384-module-bench-latest.md](a6000-l384-module-bench-latest.md) | historical consolidated module table; newer DiT/build results are above, dropout .25 training, refreshed triangle inference and remaining cross-GPU build gaps. |
| [a6000-rmsnorm-validation-fix.md](a6000-rmsnorm-validation-fix.md) | closes the RMSNorm precision qualification with unrounded reference gradients, a justified BF16 forward band, and FP64 checks. |
| [a6000-training-dropout025.md](a6000-training-dropout025.md) | real dropout 0.25 training defaults, RNG/gradient validation and L384 native comparisons. |
| [a6000-trimul-mask-fusion.md](a6000-trimul-mask-fusion.md) | fused masking for outgoing/incoming/bidirectional, full-gradient validation, paired before/after native timings, and two refreshed A6000 caches. |
| [a6000-trimul-training-cause.md](a6000-trimul-training-cause.md) | historical dtype mismatch, training-forward/backward attribution, and a same-GPU mask A/B that reverses the cuEquivariance ranking. |
| [a6000-trimul-directions-and-pytorch-reference.md](a6000-trimul-directions-and-pytorch-reference.md) | A6000 outgoing/incoming/sequential/bidirectional comparisons, equivalent cuEquivariance composition, and removal of MiniWorld kernels from the PyTorch SWA reference. |
| [a6000-small-input-followup.md](a6000-small-input-followup.md) | deferred L128 launch-overhead investigation, evidence and acceptance conditions. |
| [a6000-l384-cache-built.md](a6000-l384-cache-built.md) | four added L384/A48 keys, explicit RMSNorm precision qualification, and the completed native benchmark. |
| [a6000-l384-cache-missing.md](a6000-l384-cache-missing.md) | four missing L384/A48 forward/backward cache keys and the augmentation gap in the build plan. |
| `naming-audit.md` | the defects found while renaming 111 kernels to `docs/kernels/naming.md`'s rules. The old names are its *subject*, so they stay. Current names: `registry.csv`; the mapping: `docs/kernels/rename-map.tsv`. |
| `tiling-audit.md` | one sweep of every kernel's tile axes. Kernel names are the ones `registry.csv` held at the time. |
| `pairformer-b200-latency.md` | Pairformer pair-track latency on B200 (sm100). |
| `pairformer-h100-latency.md` | the same on H100 (sm90). |
| `where-the-cache-build-spends-its-time-a6000.md` | the compile/bench split of an A6000 rebuild, and the three things that were idling: a second autotune key compiling on one core, a pool at 50% occupancy, and compile never overlapping bench. |
| `cache-coverage-replay-a6000.md` | the 363 lookups the module matrix asks for and the shipped cache does not serve, against a static coverage check that reports zero missing. Work list for the pending rebuild. |

The two latency files were under `benchmarks/runners/`, which `docs/benchmarks.md` forbids —
"do not add curated markdown reports under `benchmarks/`; write durable explanations under
`docs/`" — and nothing cited them. They are the only evidence in this repository of anything
running on sm90 or sm100 hardware, which `docs/supported.md` lists as never exercised here, so
they are kept rather than deleted. They are also not a substitute for a device manifest: no
`#provenance`, no commit, no way to know what code produced them.

The two audits sat in `docs/kernels/` while each opened by saying it was a record and not current
documentation. `docs/` is for pages written to be read as true now.

`cache-coverage-replay-a6000.md` is kept for a different reason than the others: it is not
superseded, it is *pending*. It is the first output of `dev audit --replay`, which had existed
with no caller, and it stays until a rebuilt cache makes it empty.

- [A6000 AdaLN workload-aware cache repair and SWA fusion](a6000-adaln-workload-cache-fix.md)

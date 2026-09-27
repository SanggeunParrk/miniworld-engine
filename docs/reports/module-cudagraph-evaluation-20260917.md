# Module compile + CUDA Graph evaluation (2026-09-17)

**Recommendation:** keep compilation enabled and retain the current graphed inference baseline. For training, publish graph OFF and ON separately rather than replacing every training number with ON. Small modules can be strongly host-limited; large modules often show only a small change. Graph replay latency is useful as a module device-execution comparison, but does not predict a full optimizer/data-loading/distributed training step.

No production benchmark default, dropout policy, or autotuning configuration was changed.

## Method

- Source: `4d28918d16730370b57762acb518a725daaaa6b9`, frozen worktree; PyTorch 2.10.0+cu128, RTX A6000.
- Slurm jobs: 1709436 (two GPUs, one subprocess per GPU), 1709439 (initial PyTorch AdaLN controls), 1709458 (serial MiniWorld AdaLN/TriMul confirmations). These ran on gpu01, separate from cache builds on gpu02/gpu04.
- Reused the official `benchmarks/runners/bench.py` module construction, inputs, backend selection, compile witness and backward functions. Only `measured_result` was replaced in the experiment process; no new module implementation was written.
- Both sides use the same compiled callable, weights, shapes and selected kernels in one process. Inductor internal CUDA graphs are disabled; ON manually captures forward or forward+backward. Actual executed compile evidence is required (`module_forward:partial_allowed`), not inferred from the request.
- Seven alternating OFF/ON timing rounds, each using the official Triton timer (10 ms warmup, 100 ms measurement, median). Tables show the median of the seven round medians. Compilation, initial autotuning, capture and correctness checks are excluded. The timed OFF step does not collect diagnostics or build result dictionaries.
- Input BF16 without autocast, existing module-specific affine parameter policy preserved. TriMul reports BF16+FP32 parameters; the other tested modules report BF16 parameters. This preserves the current benchmark setup rather than introducing a new precision policy.
- L=128/384, batch 1, pair width 128, token width 768, conditioning width 384. AdaLN/ConditionedTransition augmentation=5 inference and 48 training. Trunk inputs are [1,L,L,128] (no augmentation axis). One layer, mask probability 0.2. TriMul direction is the harness default outgoing.
- Training TriMul/TriangleAttention use dropout=0.25; inference dropout=0. Training includes input/parameter gradients and fresh-gradient semantics, without an optimizer.
- Two seeds check graph outputs and captured input/parameter gradients against the same ungraphed compiled callable. Consecutive dropout replays must change the output. These execution-consistency checks passed for all 32 conditions; this is not independent mathematical certification of all kernels.
- The initial two PyTorch AdaLN training measurements were discarded and repeated with the final timing hook. Large small-shape gains were rechecked serially after the matrix; those confirmation numbers are used below for MiniWorld AdaLN/TriMul training.

## Training

| Module | Backend | L | Compile, graph OFF (ms) | Compile, graph ON (ms) | Time reduction |
|---|---|---:|---:|---:|---:|
| adaptive_layernorm | pytorch | 128 | 0.4608 | 0.4475 | 2.89% |
| adaptive_layernorm | pytorch | 384 | 1.2247 | 1.2134 | 0.92% |
| adaptive_layernorm | miniworld | 128 | 0.9457 | 0.4127 | 56.36% |
| adaptive_layernorm | miniworld | 384 | 1.1254 | 1.1116 | 1.23% |
| conditioned_transition | pytorch | 128 | 2.1335 | 2.1079 | 1.20% |
| conditioned_transition | pytorch | 384 | 6.0815 | 6.0795 | 0.03% |
| conditioned_transition | miniworld | 128 | 2.1253 | 2.0931 | 1.52% |
| conditioned_transition | miniworld | 384 | 5.7728 | 5.7697 | 0.05% |
| triangle_multiplication | pytorch | 128 | 0.9513 | 0.8069 | 15.18% |
| triangle_multiplication | pytorch | 384 | 14.1481 | 14.1604 | -0.09% |
| triangle_multiplication | miniworld | 128 | 2.1151 | 0.5253 | 75.16% |
| triangle_multiplication | miniworld | 384 | 4.4882 | 4.3715 | 2.60% |
| triangle_attention | pytorch | 128 | 0.9974 | 0.8755 | 12.22% |
| triangle_attention | pytorch | 384 | 13.8547 | 13.7513 | 0.75% |
| triangle_attention | miniworld | 128 | 1.4904 | 0.5980 | 59.88% |
| triangle_attention | miniworld | 384 | 6.4799 | 6.4456 | 0.53% |

## Inference

| Module | Backend | L | Compile, graph OFF (ms) | Compile, graph ON (ms) | Time reduction |
|---|---|---:|---:|---:|---:|
| adaptive_layernorm | pytorch | 128 | 0.0297 | 0.0276 | 6.90% |
| adaptive_layernorm | pytorch | 384 | 0.0666 | 0.0635 | 4.62% |
| adaptive_layernorm | miniworld | 128 | 0.1823 | 0.0276 | 84.83% |
| adaptive_layernorm | miniworld | 384 | 0.1690 | 0.0522 | 69.09% |
| conditioned_transition | pytorch | 128 | 0.1157 | 0.1085 | 6.19% |
| conditioned_transition | pytorch | 384 | 0.3246 | 0.3174 | 2.21% |
| conditioned_transition | miniworld | 128 | 0.3809 | 0.0932 | 75.54% |
| conditioned_transition | miniworld | 384 | 0.3922 | 0.2191 | 44.13% |
| triangle_multiplication | pytorch | 128 | 0.2519 | 0.2376 | 5.69% |
| triangle_multiplication | pytorch | 384 | 4.9029 | 4.9132 | -0.21% |
| triangle_multiplication | miniworld | 128 | 0.3574 | 0.1085 | 69.63% |
| triangle_multiplication | miniworld | 384 | 0.9544 | 0.9339 | 2.15% |
| triangle_attention | pytorch | 128 | 0.2877 | 0.2724 | 5.34% |
| triangle_attention | pytorch | 384 | 5.0662 | 5.0519 | 0.28% |
| triangle_attention | miniworld | 128 | 0.1884 | 0.1403 | 25.54% |
| triangle_attention | miniworld | 384 | 1.5493 | 1.5360 | 0.86% |

## Interpretation and limits

A small percent difference near zero should be treated as timing noise, not a reliable optimization. The much larger changes in some L128 cases are consistent with eliminating host dispatch/launch gaps; this experiment does not profile those gaps into individual Python functions.

The production harness currently rejects training dropout with graphs before capture, and its replay check compares against an earlier stochastic invocation without reseeding. That blanket restriction is a harness limitation, not evidence that these two dropout-bearing modules cannot be graphed. The isolated same-seed output/gradient checks and varying-replay checks demonstrate feasibility for these tested shapes/backends. Supporting it officially requires integrating those checks; simply changing the YAML flag is insufficient.

Latest-kernel caches were still being rebuilt. This experiment freezes the pre-build cache state and warms any heuristic choices before both timings, so OFF/ON comparisons use the same selection. Absolute latencies and PyTorch/MiniWorld rankings here are not the final tuned-cache benchmark. After caches are qualified, regenerate any official cross-backend comparison with equal graph settings.

This covers four representative modules on A6000, not all modules, cuEquivariance, other GPU architectures, dynamic bucket switching, graph memory overhead, DDP, optimizer updates, gradient accumulation or whole-model training. Keep graph-OFF results when the deployed training loop does not use graphs, and graph-ON results when evaluating repeated fixed-shape replay.

## Local evidence

Raw per-case JSON includes seven paired timings, compile evidence, input shapes, parameter dtypes, replay errors, capture setup time and (where applicable) dropout variation. It is preserved with logs and the isolated runner under `.bench/graph-eval-20260917/`. `summary.json` selects the final control/confirmation rows; initial results remain archived separately.

Measurement script SHA-256: `0de209d4ffb31a89f603f963b549421e5d8064d8f6c115a5b11448bd4be8f260`.

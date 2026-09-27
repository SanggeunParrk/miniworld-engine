# MPNN targeted cache build and comparison — A6000

Only the native BF16 B8/L8192/K48/D128 training workloads used by the actual
feature, encoder, decoder and full-model comparisons were built.
Inputs are seeded synthetic proteins/module states from the existing benchmark harness.
Ordinary parameters, gradients and AdamW moments are BF16; LayerNorm parameters are FP32.
No autocast or gradient checkpointing. Real fullgraph compilation was witnessed;
training CUDA Graphs are disabled and dropout is 0.25 where applicable.

## Coverage

14 operations, 40 exact physical workload profiles.
32,004 registered candidate/profile combinations were checked, including observed exclusions.
The inventory includes tensor shape, stride, dtype, scalar kernel arguments and
implementation identity. B1 registry drivers were not substituted for these calls.
Every profile was checked against the full registered grid after merging;
no unsearched candidates or runtime cache misses remain in the checked contexts.
Resource and compile-budget failures are observed exclusions, not successful
kernel timings. Predictive exclusions were disabled. Top five finite candidates
per physical profile were retained by the repository cache writer.

| Operation | Profiles | Candidate/profile combinations |
| --- | ---: | ---: |
| mpnn_edge_tail_bwd_dx_saveact_triton | 2 | 6804 |
| mpnn_edge_tail_bwd_dx_gather_saveact_triton | 1 | 3402 |
| mpnn_edge_tail_fwd_gemm_b2b_saveact_triton | 1 | 3402 |
| mpnn_edge_tail_fwd_gemm_gather_saveact_triton | 1 | 3402 |
| mpnn_message_bwd_dx_triton | 1 | 3402 |
| mpnn_message_fwd_gemm_triton | 1 | 3402 |
| mpnn_node_message_bwd_dx_triton | 13 | 2730 |
| mpnn_node_message_bwd_recompute_triton | 13 | 2730 |
| mpnn_relative_position_bwd_reduce_triton | 1 | 840 |
| mpnn_node_message_fwd_gemm_triton | 2 | 420 |
| mpnn_edge_tail_fwd_gemm_layernorm_saveact_triton | 1 | 378 |
| mpnn_message_bwd_reduce_dbias_triton | 1 | 378 |
| mpnn_message_fwd_gelu_reduce_triton | 1 | 378 |
| mpnn_edge_tail_bwd_layernorm_saveact_triton | 1 | 336 |

Pre-build fallback and post-build tuned full-model outputs/parameter gradients
were compared with identical seeded dropout and unchanged model weights:

| Policy | Output relative L2 | Gradient relative L2 |
| --- | ---: | ---: |
| full:all_compute | 0 | 0.000242836 |
| full:current | 0 | 0.000239101 |

All checked contexts also had finite gradients with the required native dtypes.
This is a cache-selection regression check; earlier compiled-vs-PyTorch numerical
checks are documented in the compiler attribution record. It is not a convergence test.

## Actual modules

Seven samples per cell, median forward + input/parameter backward, no optimizer.
All new rows were measured sequentially on the same A6000. Isolated module costs
are not additive whole-model costs. Encoder is the general nonzero-node layer.

| Module / backend | Earlier run ms | Post-build ms | Compiled PyTorch / backend |
| --- | ---: | ---: | ---: |
| decoder / compute | 40.44 | 38.07 | 1.016x |
| decoder / pytorch | 39.21 | 38.69 | 1.000x |
| encoder / compute | 73.80 | 72.17 | 1.266x |
| encoder / memory | 87.15 | 80.46 | 1.136x |
| encoder / pytorch | 91.23 | 91.38 | 1.000x |
| features / compute | 73.80 | 73.47 | 1.023x |
| features / memory | 77.49 | 77.01 | 0.976x |
| features / pytorch | 75.02 | 75.15 | 1.000x |

## Full model

Three encoder and three decoder layers. Seven step timing samples plus five
finite-gradient/allocated-memory validation steps. The timing scope includes
forward + FP32 cross-entropy + backward + eager AdamW + zero_grad.

| Policy | Earlier run ms | Post-build ms | Compiled PyTorch / backend | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| pytorch | 447.62 | 447.33 | 1.000x | 34.74 |
| all_compute | 420.75 | 402.57 | 1.111x | 24.94 |
| no_node | 423.00 | 409.08 | 1.094x | 27.19 |
| no_tail | 454.64 | 440.62 | 1.015x | 30.50 |
| no_decoder | 414.50 | 401.70 | 1.114x | 26.86 |
| no_position | 422.63 | 403.92 | 1.107x | 24.94 |
| current | 432.75 | 413.24 | 1.083x | 21.43 |

All-compute retains RBF. Current regenerates RBF and recomputes the first encoder
node-message activation. The no_* policies start from all-compute and replace
only that named custom group with compiled PyTorch. The PyTorch row uses the
same optimized graph/model algorithm, with custom MPNN kernels disabled.
The earlier fallback column is from separate prior allocations, so its delta
can include run-to-run variance. Compare the new PyTorch and custom rows for
the matched post-build result. No A5000 fit or other-size cache coverage is inferred.

## Reproduction and artifacts

Source hash: `64903f6d1ad3189393f7e9ca4e97a2cdc4f5353fdbacd618a59578e5714c2fed`.
Inventory job 1682293; parallel build job 1682295; two-operation recovery,
verification and sequential comparison job 1682315; matched cache A/B job 1682320.
Jobs 1682293, 1682315 and 1682320 completed with exit 0. Job 1682295 ends
with exit 1 because its original coordinator retains the two intentionally
stopped worker statuses; both were successfully rebuilt and validated by 1682315.
During the initial build, the bucket-only compile settlement key treated
different chunk-offset JIT specializations as already compiled. The next
specialization then compiled serially in the parent expected-cache-hit path.
The target runner now adds physical workload identity to compile settlement
keys. Only the two affected processes were stopped; their finished profiles
were merged first and reused. Other operation builds continued. Kernel source
and benchmark arithmetic were unchanged. This safeguard is in the target runner;
it does not claim that the general builder was changed by this task.
The independent EMIT_BIAS=0 edge-tail backward profile was also split into
a separate shard and built on a completed unit's idle GPU/CPU slots within
job 1682295. The original unit reuses its shared completed tuning round.
Pending extra allocations 1682335 and 1682340 were cancelled before execution.
Raw inventory, per-operation shards/logs, round restart cache, coordinator,
Slurm script and benchmark results are under `.scratch/mpnn-native-bf16/target-b8-20260912`.
The runner is `benchmarks.runners.mpnn_target_cache`: inventory, targeted build,
verify, and compare-block. Full-model comparisons reuse mpnn_attribution and
the existing shared measurement/compile harness. Run only on allocated nodes.

[Complete records](mpnn-target-cache-a6000.json).
[Earlier fallback/compiler analysis](mpnn-compiler-attribution-a6000.md).

## Matched cache A/B

Each pair runs sequentially on the same A6000 with unchanged model and
measurement source. Fallback bypasses only MPNN cache reads inside its own
child process, reproducing the ordinary heuristic 24-candidate behavior.
It does not delete or modify the built cache. Tuned runs require zero cache misses.
Both use the same seven-sample full-step/validation contract as above.

| Policy | Fallback 24 ms | Tuned ms | Fallback / tuned | Step time reduction |
| --- | ---: | ---: | ---: | ---: |
| all_compute | 415.18 | 402.74 | 1.031x | 3.00% |
| current | 427.06 | 413.38 | 1.033x | 3.20% |

## Candidate selection within each completed tuning grid

This table compares the best finite candidate in the full grid against the
best of the ordinary heuristic 24, using timings from the same tuning round.
It is a kernel-candidate comparison, not a full-model speedup. Ranges cover
different physical profiles of one operation. Compile/resource/launch-budget
exclusions are recorded as non-finite outcomes and never selected as winners.

| Operation | Profiles | Heuristic-best / full-grid-best |
| --- | ---: | ---: |
| mpnn_edge_tail_bwd_dx_saveact_triton | 2 | 1.020–1.035x |
| mpnn_edge_tail_bwd_dx_gather_saveact_triton | 1 | 1.029x |
| mpnn_edge_tail_fwd_gemm_b2b_saveact_triton | 1 | 1.001x |
| mpnn_edge_tail_fwd_gemm_gather_saveact_triton | 1 | 1.003x |
| mpnn_message_bwd_dx_triton | 1 | 1.005x |
| mpnn_message_fwd_gemm_triton | 1 | 1.705x |
| mpnn_node_message_bwd_dx_triton | 13 | 1.048–1.255x |
| mpnn_node_message_bwd_recompute_triton | 13 | 1.000–1.040x |
| mpnn_relative_position_bwd_reduce_triton | 1 | 1.339x |
| mpnn_node_message_fwd_gemm_triton | 2 | 1.088–1.110x |
| mpnn_edge_tail_fwd_gemm_layernorm_saveact_triton | 1 | 1.132x |
| mpnn_message_bwd_reduce_dbias_triton | 1 | 1.000x |
| mpnn_message_fwd_gelu_reduce_triton | 1 | 1.380x |
| mpnn_edge_tail_bwd_layernorm_saveact_triton | 1 | 1.000x |

A [subsequent structural optimization check](mpnn-optimization-followup-a6000.md)
compared larger backward chunks and folded gradient casts. Neither candidate
was adopted; the production implementation and the caches documented here remain intact.

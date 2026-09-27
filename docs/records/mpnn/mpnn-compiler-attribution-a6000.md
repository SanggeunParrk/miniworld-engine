# MPNN compiler and kernel attribution — A6000 native BF16

Measured on allocated RTX A6000 cards on gpu04. Each table compares policies
sequentially on the same card; the four experiments use separate allocations.
Native BF16 ordinary parameters, FP32 LayerNorm parameters/statistics, no autocast;
fullgraph compilation observed, training CUDA Graphs disabled. Dropout-bearing
modules use p=0.25. Geometry and loss remain FP32. No production backend was changed.

## Actual modules: B8/L8192, K48, D128

These are real BackboneFeatures+input projection, EncoderLayer and DecoderLayer
objects. Seven forward+input/parameter-backward measurements, no optimizer.
Encoder uses a general nonzero-node layer; decoder past/future masks split
synthetic local/random neighbors. These isolated times are not additive full-model costs.

| Module | PyTorch ms | Compute ms | PyTorch / compute | Memory policy ms |
| --- | ---: | ---: | ---: | ---: |
| features | 75.02 | 73.80 | 1.016x | 77.49 |
| encoder | 91.23 | 73.80 | 1.236x | 87.15 |
| decoder | 39.21 | 40.44 | 0.970x | n/a |

Feature memory regenerates RBF. Encoder memory changes only its node-message
backend to preactivation recomputation; its edge tail remains compute.

## Existing composite/kernel-family runner: N65536, K48, D128

Five forward+backward timing samples per cell, no optimizer. Synthetic B1
flattened nodes, 20% masked edges, half local/half random indices. This matches
B8/L8192 element count, not its precise neighbor graph. Accuracy probes use
dropout zero; training timings keep dropout 0.25 where applicable.

| Family | PyTorch ms | Backend | Backend ms | Speedup |
| --- | ---: | --- | ---: | ---: |
| message | 17.53 | triton_compute | 15.19 | 1.154x |
| message | 17.53 | triton_memory | 19.58 | 0.895x |
| edge_mlp | 26.96 | triton_compute | 24.12 | 1.118x |
| edge_mlp | 26.96 | triton_memory | 24.37 | 1.106x |
| edge_tail | 58.14 | triton_compute | 40.68 | 1.429x |
| edge_tail | 58.14 | triton | 59.86 | 0.971x |
| edge_layernorm | 6.53 | memory | 6.12 | 1.068x |
| node_message | 30.20 | triton | 37.08 | 0.815x |
| node_message | 30.20 | triton_compute | 26.72 | 1.130x |
| relative_position | 4.86 | triton | 1.72 | 2.828x |
| relative_position | 4.86 | index_add | 4.84 | 1.004x |
| edge_dropout | 5.58 | bitpack | 5.70 | 0.978x |

Standalone edge MLP, compressed LayerNorm and bitpack dropout are alternative
component experiments. Their savings must not be added to edge-tail savings:
the compute edge tail already contains the corresponding operations and its own
packed dropout saves. Node message also contains message reduction.

## Full-model leave-one-group-out comparison

B8/L8192, 3 encoder + 3 decoder layers, fullgraph forward/loss + AOT backward
+ eager AdamW (lr 1e-4), seven samples plus five finite-gradient validation steps.
All-compute retains RBF. Every ablation starts from that exact setup and changes
only the named group; current additionally restores RBF regeneration and the
first encoder node recomputation policy. All parameter dtypes and optimizer
moment dtypes were checked. Source identity and actual kernel dispatch were checked.

| Policy | Step ms | Difference vs all compute (ms) | Warm validation peak allocated GiB |
| --- | ---: | ---: | ---: |
| pytorch | 447.62 | +26.87 | 34.74 |
| all_compute | 420.75 | +0.00 | 24.94 |
| no_tail | 454.64 | +33.89 | 30.50 |
| no_node | 423.00 | +2.25 | 27.19 |
| no_decoder | 414.50 | -6.25 | 26.86 |
| no_position | 422.63 | +1.88 | 24.94 |
| current | 432.75 | +12.00 | 21.43 |

Positive ablation delta means removing the custom group made the model slower;
negative means replacing that group with compiled PyTorch improved throughput.
Deltas include changes in adjacent compiler fusion and are not additive isolated
kernel costs. Small deltas are descriptive single-allocation results.

## Decoder confirmation in A/B/B/A order

Fresh process per row, same A6000 UUID within this additional allocation.
A = all compute; B = decoder message returned to compiled PyTorch. Each row
has seven timing samples, the same full-model correctness/dispatch contract,
and five finite-gradient validation steps.

| Order | Policy | ms | Sample min–max ms |
| --- | --- | ---: | ---: |
| 1 | all_compute | 418.84 | 417.52–419.72 |
| 2 | no_decoder | 413.38 | 412.49–414.18 |
| 3 | no_decoder | 414.88 | 414.32–415.17 |
| 4 | all_compute | 421.06 | 420.66–421.69 |

## Actual compiler flow

The generated Inductor Python/Triton code and a separate warmed GPU profile were
captured with TORCH_LOGS=output_code and torch.profiler. Profiling was outside
the timing samples. No assumption that a custom op equals a single GPU launch.

- Feature geometry: the compiler fuses neighbor coordinate loads/differences,
  RBF distance/exp/cast, and normalization/masking. KNN still uses its distance
  GEMM and top-k. The retained-RBF compute variant changes relative-position backward.
- PyTorch encoder/decoder W1: external GEMM, then a generated kernel containing
  gather + query/neighbor/mask additions + GELU. In decoder-compute, GELU moves
  inside the opaque message op, leaving a separate gather/add producer.
- PyTorch node/message reduction: W2 external GEMM followed by fused GELU, mask,
  neighbor reduction and division. The output projection occurs at node count.
- PyTorch edge update: external GEMMs; bias/dropout/residual/LayerNorm share one
  generated kernel. Kernel names containing addmm do not imply GEMM is fused:
  inspecting the body shows loads from its already-computed GEMM output.
- PyTorch decoder backward: GELU derivative, masks, neighbor atomic accumulation
  and query reduction share generated kernels. The opaque message backward splits
  the GELU derivative away from this surrounding fusion.
- MiniWorld message backward reuses a bounded activation scratch by processing
  3,145,728 edge rows in 12 chunks of 262,144. Across six message backward uses,
  the profile records 72 projection-dX kernels plus their partial dW GEMMs.
  This saves storage but increases launch count. It is present even in compute mode.

These code observations identify optimization boundaries. An isolated chunk-size
ablation was not performed, so its exact contribution to elapsed time is not claimed.

## Full-model GPU profile (one separate step; kernel durations, not benchmark medians)

CPU launch correlation IDs assign each actual GPU kernel to forward/backward/AdamW.
Only trace category kernel is summed, avoiding double-counting CPU op annotations.

| Path | Forward GPU ms / launches | Backward GPU ms / launches | AdamW GPU ms |
| --- | ---: | ---: | ---: |
| pytorch | 165.95 / 196 | 283.16 / 442 | 0.230 |
| all_compute | 148.10 / 159 | 269.55 / 821 | 0.235 |

## Validation and artifacts

All 19 family cases passed their existing eager numerical checks and witnessed
compiled finite-gradient timing. All eight actual-module cases passed compiled
output/gradient comparisons against eager pure PyTorch at B2/L128 with dropout
zero (relative L2 below 2%). Seven full-model policies passed the corresponding
compiled-vs-eager comparison (relative L2 below 2%, gradient cosine above 0.999).
BF16 rounding differences across compiler fusion boundaries are recorded.

Timing shapes, masks and scopes differ across the tables and are explicit.
These results do not measure inference, convergence, or an exhaustive tuning grid.
The existing cache and ordinary 24-candidate heuristic fallback were used.

[Full records and compiler/profile summaries](mpnn-compiler-attribution-a6000.json).
Raw logs, profiler traces and extracted compiler graphs are under
`.scratch/mpnn-native-bf16/{modules-1682272,blocks-1682277,attribution-1682275,repeat-1682284}`.
Successful Slurm jobs: 1682272, 1682277, 1682275, 1682284. The initial new
attribution runner failed its setup check in job 1682273 due to a wrong module
attribute; it produced no timing and was corrected before these measurements.

Reproduction uses `benchmarks.runners.mpnn_compare` (families),
`benchmarks.runners.mpnn_blocks` (actual modules) and
`benchmarks.runners.mpnn_attribution` (full-model ablations), all wrapping the
existing shared `measured_result` and `compile_for_benchmark` harness.
Run only on allocated compute nodes. Scripts and measured runner snapshots are
preserved in `.scratch/mpnn-native-bf16/attribution-scripts/`.

No production kernel or default policy was edited by this attribution task.
The separate dirty main worktree and existing A5000 build job were preserved.

Returning decoder messages to PyTorch saved 6.25 ms in the main ablation but
increased the warm validation peak from 24.94 to 26.86 GiB (+1.92 GiB). This is
a measured speed/storage tradeoff; no A5000 fit is inferred from these A6000 rows.

Final runner lint/type checks passed (Ruff and ty against the actual PyTorch
2.10 environment), and `git diff --check` passed. The measured attribution runner
is archived with its recorded hash; afterward, its unused `config` local was
renamed `_config` for lint only. The benchmark arithmetic and policy are unchanged.

## Follow-up: persistent tuning-cache audit

At the time of the fallback measurements, the MPNN worktree contained
**zero MPNN tuning JSON files** under
`src/miniworld_engine/autotune/data/` (including ignored/untracked files).
The registry declares 22 MPNN kernels; registry/config-grid/validation-manifest
presence is not evidence of a built tuning cache. In particular, this is broader
than missing native-BF16 keys in an existing MPNN JSON file.

The all-compute measurement log reports `no tuned autotune cache` for 12 unique
operations. Runtime selected and timed a heuristic subset of 24 candidates:
node forward has 210 registered candidates, message GEMM/dX and several edge-tail
GEMM/dX kernels have 3,402, and relative-position backward has 840. Other reduction
and norm kernels have 336 or 378. Initial compilation/tuning was excluded from
the reported step times, but the selected configurations were not winners from
a completed full-grid persistent cache build.

The tables are valid measurements of that fallback setup. They do not establish
fully tuned kernel performance. Compiler fusion boundaries and chunked backward
launches are observed facts, but the relative cost of suboptimal tile selection
had not been isolated in these measurements. The subsequent targeted build
covers the 14 operations and 40 physical profiles actually used by the native
BF16 B8/L8192 training comparison. It verified all 32,004 candidate/profile
combinations, zero runtime cache misses, and seeded output/gradient agreement.
The new matched module, full-model, and cache A/B measurements are recorded in
[the targeted cache comparison](mpnn-target-cache-a6000.md). Use that follow-up
for tuned performance; the tables above remain the historical fallback results.

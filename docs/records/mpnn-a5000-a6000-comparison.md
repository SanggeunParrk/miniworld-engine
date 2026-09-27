# MPNN: A5000 current policy vs A6000 compute policies — 2026-09-12

The same final kernel source and existing measurement runner were used for all four rows.
A6000 results were remeasured sequentially on one allocated card, job **1682218**, gpu01.
The A5000 row is the completed confirmation from job **1680997**, gpu02.

**Workload:** B=8, L=8192, K=48, all widths=128, 3 encoder + 3 decoder layers,
fixed coordinates, BF16 autocast, FP32 parameters/gradients/AdamW, dropout 0.25.
Fullgraph forward and AOT backward; CUDA Graphs off; no gradient accumulation or
checkpoint API calls. Time includes forward, backward and the AdamW update.

| Device / policy | Median ms/step | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|
| A5000 current | 546.61 | 21.846 | 21.973 |
| A6000 current | 442.93 | 21.846 | 22.068 |
| A6000 all compute, regenerate RBF | 433.11 | 23.315 | 23.943 |
| A6000 all compute, retain RBF | 427.81 | 25.362 | 25.975 |

Current policy on A6000 is **1.234×** the measured A5000 throughput (18.97% less step time).
On the same A6000, all-compute plus retained RBF reduces step time by **3.41%** versus the current policy.
Equivalently, the current policy costs **3.54%** extra step time while saving **3.516 GiB** allocated memory.

## What differs

- Current: first encoder node uses `triton`; other encoder nodes, all encoder edge tails,
  and all decoder messages use their compute backends. `feature_backend="memory"`
  retains distances and regenerates RBF for the feature weight gradient.
- All compute, regenerate RBF: every encoder node also uses `triton_compute`;
  the feature memory policy remains unchanged.
- All compute, retain RBF: same compute layers, with `feature_backend="pytorch"`.
  The ordinary compiled feature path retains expanded RBF for backward, trading
  more memory for avoiding RBF regeneration. Packed dropout masks remain enabled.

## Validation and measurement limits

Each candidate passed seven timing samples, kernel-dispatch witnessing and twenty
additional optimizer steps with finite losses and every parameter gradient finite.
Checkpoint API calls were zero in every completed record. All A6000 candidates
used the same card, source hash, runner hash and environment, each in a fresh process.

Allocated/reserved columns include real warmup and validation steps, excluding the
timer-only flush-buffer peak (stored separately in the JSON). A6000 had no artificial
allocator cap; A5000 training used 22 GiB and its timer used 22.5 GiB. Thus the cross-GPU
rows compare their measured operating setups, not a simultaneous hardware-only experiment.

The A6000 per-sample ranges were:
- A6000 current: 441.04–443.85 ms.
- A6000 all compute, regenerate RBF: 432.71–434.35 ms.
- A6000 all compute, retain RBF: 426.50–428.19 ms.

These new measurements supersede the older allocation for comparing A6000 policies.
They do not identify why historical absolute timings differed. No new exhaustive
kernel-cache tuning was performed; missing MPNN entries use the existing heuristic shortlist.

The JSON includes exact policy arrays, configurations, sample timings, dispatch evidence,
source hashes and raw-record paths. The source remains the MPNN implementation at `17492754`;
this experiment changes no kernels or default backend selection.

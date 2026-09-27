# MPNN handoff — 2026-09-13

MPNN optimization is paused at the validated native-BF16 implementation. No new
kernel experiment is running. The larger backward chunks and folded gradient
casts were not adopted; training CUDA Graphs remain off by default.

## Start here

- [Native BF16 contract and correctness](mpnn-native-bf16.md): ordinary parameters,
  gradients and AdamW moments are BF16; LayerNorm parameters remain FP32. No autocast.
- [Exact A6000 cache and module comparison](mpnn-target-cache-a6000.md): B8/L8192/K48/D128,
  3 encoder + 3 decoder layers, dropout 0.25, no checkpointing, actual fullgraph
  compilation, training graphs off. Fourteen operations and forty physical profiles
  were searched and replayed with zero runtime cache misses.
- [Compiler attribution](mpnn-compiler-attribution-a6000.md),
  [structural follow-up](mpnn-optimization-followup-a6000.md), and
  [training graph ON/OFF comparison](mpnn-training-cudagraph-a6000.md).

| Matched post-cache A6000 policy | Full training ms/step | Peak allocated GiB |
| --- | ---: | ---: |
| Compiled PyTorch | 447.33 | 34.74 |
| MiniWorld all-compute, retain RBF | 402.57 | 24.94 |
| MiniWorld current memory policy | 413.24 | 21.43 |

These are the matched results from the cache record, not medians pooled across
later GPU allocations. Whole-step time includes loss, backward, AdamW and gradient
clearing on resident synthetic inputs. The current policy regenerates RBF and
recomputes the first encoder node-message activation. The full cache build covers
these exact A6000 profiles, not every size or every GPU.

The later ON/OFF graph experiment found only 0.05–0.32% whole-step reductions.
Its graph captures forward/loss/backward, keeps eager AdamW outside, and checks
advancing dropout RNG, fresh gradients and optimizer updates. Graph private-pool
memory is included in the report; replay-only allocated counters are not model peaks.

## Remaining scope

- Wider decoder gather/mask/reduction fusion was identified but not implemented.
- No native-BF16 B8/L8192 A5000 fit is claimed. The older A5000 546.61 ms / 21.846 GiB
  result uses FP32 parameters with BF16 autocast; see the separate GPU comparison.
- The general A5000 AF3-style cache build/validation is a separate workstream and
  must not be inferred from the targeted A6000 MPNN cache.
- Historical ~245–274 ms entries use smaller total workloads; they are not an
  earlier matched B8/L8192 native-BF16 full-training baseline.

## Reproduce or resume

Run GPU work only in a Slurm allocation. Benchmark entry points:
`benchmarks.runners.mpnn_training`, `mpnn_attribution`, `mpnn_blocks`, and
`mpnn_target_cache`. The records contain exact commands, source/runner identities,
raw result paths and policy definitions. Optimization/graph follow-up JSON records
also embed their experimental scripts. Ignored `.scratch/` compiler artifacts and
raw logs are retained locally; they are not required to import the package.

# A5000 cache-build memory-access investigation

Date: 2026-09-12. Source baseline: `fb6777417c0a34e383e0984978fec89169c18e28` plus the fixes described below. Device: NVIDIA RTX A5000, sm86, Triton 3.6.0, gpu02. This record does not qualify the complete A5000 cache or attribute the unreproduced faults to faulty hardware.

## Confirmed LayerNormLinear fault

Three failed bidirectional TriangleAttention inference units (BF16, d_pair=128, L=384/512/640) reached the same LayerNormLinear schedule:

| Parameter | Value |
| --- | --- |
| K / N | 128 / 520 |
| BLOCK_K / BLOCK_M1 / BLOCK_N | 128 / 64 / 128 |
| num_warps / num_stages | 1 / 1 |

Direct launch at M=256 reproduces an illegal memory access on two allocated A5000 cards. Compute Sanitizer reports an **invalid shared-memory write of 16 bytes** in `_lnl_fwd_kernel` (`fused.py`, `tl.dot`). The sanitizer run reports 34 errors. This is the same schedule already excluded on A6000. The pruning predicate now excludes it on A5000 too, restricted to the observed K/N/dtype/architecture. The JIT body and config grid are unchanged.

Verification:

- Adjacent schedules `(warps, stages) = (1,2), (2,1), (4,1)` pass sanitizer at M=256.
- All three pass numerical checks at M=147456, 262144 and 409600 (L squared). Output relative L2 error against the FP32 reference is 0.163–0.167%; the check compares the first 32 output rows after launching the entire tensor.
- All three formerly failing TriangleAttention module units now finish rebuilding with `unit ran=1` and exit code 0.

An isolated reproducer must bypass pruning to exercise the bad raw JIT schedule. The following call uses `M=256`, `K=128`, `N=520`, BF16 contiguous `x[M,K]`, `w[N,K]`, and `y[M,N]`, FP32 `g[K]`/`b[K]`, and a single-use CUDA process:

```python
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.layernorm_linear.triton.fused import _lnl_fwd_kernel

_lnl_fwd_kernel.fn[(4,)](
    x, w, x, g, b, y, 256, 520, 128, 1e-5, 128, 1, 128, 1, 520, 1,
    HAS_BIAS=False, shape_key=both_key(256, K=128, N=520),
    BLOCK_K=128, BLOCK_M1=64, BLOCK_N=128, num_warps=1, num_stages=1,
)
torch.cuda.synchronize()
```

Run the reproducer under `compute-sanitizer --tool memcheck`; a fault is the expected result for that raw schedule. Production dispatch must prune it instead.

## Four other original failures

| Module | Mode / shape | Kernel active at the original failure |
| --- | --- | --- |
| Pairformer, d_pair=128 | train, L=384, persistent LN backward | Transition SwiGLU recompute backward |
| Pairformer, d_pair=256 | train, L=384, persistent LN backward | Transition SwiGLU forward |
| Pairformer, d_pair=256 | eval, L=512, fused gate | Triangle multiplication output projection |
| AugmentedAttention, d_single=768, d_cond=384, d_pair=128, heads=16 | train, L=512, BF16 core | AdaLN forward gate |

All four original failures were on physical GPU UUID `GPU-bbfd1018-c7ea-a60c-5567-ec618e5dcada`. Driver records report Xid 31 MMU faults, unlike the deterministic LayerNormLinear shared-memory fault (Xid 13). That correlation is not sufficient to diagnose a hardware defect.

The original logs did not identify the exact failing config. Probes therefore covered the last recorded successful candidate and adjacent candidates inferred from the search order. They do not establish that those were the faulting configs. These probes passed sanitizer and reference comparisons. Under an additional 18 GiB allocation, the backward, trimul and AdaLN next-candidate probes passed ten launches; Transition stages=6 was rejected for 102400 bytes shared memory against the 101376-byte device limit. A Transition-backward probe also passed 100 launches on the original physical GPU.

All four original module units passed their complete replays in job 1680191 (`unit ran=1`, exit code 0). The d_pair=128 Pairformer training replay ran on the original faulting physical GPU. Replays reuse only matching GPU/compiler/kernel identities and exact recorded workload identities from each original shard. Their remaining candidates are compiled and measured through the real module builder. Final per-unit results are recorded in the companion JSON. A passing replay is evidence of successful rebuilding, not an established root cause for the original MMU faults.

## Builder fixes

- The launch-budget warmup propagates a fatal CUDA error immediately, preserving the first fault instead of benchmarking the poisoned context again.
- The fatal-error log includes the operation, config and cache key, so a recurrence can be reproduced precisely.
- Shards record `_unit_complete` only when the module/driver ran successfully and capture reported no recording errors. Resume requires that marker as well as nonempty timings and compatible provenance. Historical or partial shards remain eligible for validated timing merge, but do not prove that a unit completed.
- Failed child processes release their claims even when earlier kernels produced timings. Explicit `--reclaim` also releases claims without a verified completed shard.
- Both child paths clean up their precompile pool in `finally`, including errors.

OOM and shared-memory capacity rejections remain expected exclusions on this 24 GB card. They are not reclassified as illegal-access bugs.

## Artifacts and recovery

Original build: job 1678627, `.scratch/gpu02-a5000-fb677741/`. It was stopped to isolate the faults; its shards, round cache and gpu02 compiler cache (`/tmp/mw-a5000-build-1678627/triton`) were retained.

Diagnostic scripts, raw sanitizer outputs, numerical probe JSONL and module-replay logs are under `.scratch/a5000-memory-debug/`. The companion JSON stores compact validation results. Recovery uses the normal provenance-checking merger for completed measurements, then `build all` without `--rebuild`; it does not relabel old shards as completed or invalidate valid JIT measurements.

## Final validation and resumed build

- All seven previously failing module units rebuilt successfully.
- CPU builder/autotune suites: **1257 passed, 1 skipped**; focused regression suite: 83 passed. Scoped Ruff and ty passed.
- A real A5000 TriangleAttention rerun using the new builder writes `_unit_complete=true`, finishes with `unit ran=1`, and exits without the old pool-finalizer traceback. Earlier replay processes had imported the old cleanup code before that fix.
- Updated sm86 module-to-kernel plan: **5258 invocations, 0 errors, 1008 required rows, 53 kernels**.
- Resume job **1680229** started on gpu02 with six A5000s. It verified successful replay results and merged 310 original/replay shards (1078 written workload buckets, zero skipped records) through the normal provenance checks. Incremental `build all` selected 38 module units for 43 remaining required keys, followed by 772 alternative-kernel driver units. Initial resumed units have succeeded. The full A5000 cache build remains a separate, ongoing task.

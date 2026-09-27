# A6000 L384 training cache omissions

Recorded 2026-09-10. Inspection only; no tuning or performance change.

The native L384 benchmark left six cells unmeasured: MiniWorld AdaLN,
ConditionedTransition, DiT, SWA Attention, and both PyTorch / MiniWorld SWA DiT.
All are training at augmentation 48 with CUDA Graph OFF. Inference uses augmentation
5 and Graph ON. The benchmark stopped at the first missing key, so its two distinct
forward errors were not a complete list of missing forward/backward keys.

## What the follow-up found

The existing native benchmark target functions built the six exact input workloads,
then ran forward + backward on FakeTensors using the existing `derive` Triton launch
recorder. All six completed without errors: 24 distinct autotune keys,
20 with stored entries and **4 missing**. All four missing keys
are also absent from the sm86 rows of `registry_kernel.csv`.

| Kernel | Shape key | Actual flattened input | Affected cells |
|---|---:|---|---|
| `layernorm_fwd_saveact_strided_triton` | `274879480602` | `[18432, 384]` | MiniWorld AdaLN, ConditionedTransition, DiT |
| `layernorm_bwd_atomic_strided_triton` | `274879480602` | `[18432, 384]` | Same three cells |
| `rmsnorm_fwd_triton` | `9895604781850` | `[589824, 32]` | MiniWorld SWA Attention; PyTorch and MiniWorld SWA DiT |
| `rmsnorm_bwd_triton` | `9895604781850` | `[589824, 32]` | Same three cells |

All four use dtype label `bfloat16+float32`. The RMSNorm keys also contain
`HAS_WEIGHT=0`; the LayerNorm keys have no such field. These are four distinct
`(op, dtype, bucket)` entries, not four new operation implementations.

- LayerNorm conditioning: `[48, 1, 384, 384]` flattens to `M=48*384=18432`,
  `N=384`. `both_key` floors the row count to 16384, then packs `N=384`.
- SWA q/k normalization: `[48, 3072, 4, 32]` flattens to
  `M=48*3072*4=589824`, `N=32`. The row bucket is 589824.
- Both backward keys are reached in the completed FakeTensor trace; they were
  hidden behind the corresponding forward miss in the native benchmark.
- The SWA DiT PyTorch baseline shares the SWA attention core, including its
  MiniWorld RMSNorm and gate kernels. Its cache miss is therefore expected.

## Why the build plan missed them

`derive.record()` calls `case.inputs(1, ...)`; its unit/registry representation has
no augmentation field. `builder.cases()` supplies ordinary batch-1 tensors for
AdaLN and ConditionedTransition, and explicitly fixes augmentation to 2 in the
augmented-attention case. The SWA case also uses batch 1. This does not model the
benchmark's training augmentation 48 and the SWA q/k head axis folded into rows.

The mode-dependent augmentation policy was corrected in the benchmark runner and
YAMLs, but the build-case input declaration was not brought into agreement. Finishing
every key in the old derived plan therefore cannot prove coverage of this workload.
The prior L128 LayerNorm repair built row bucket 4096 (actual 6144 rows), which does
not cover this L384 row bucket 16384. This is an input-coverage omission; the
inspection found absent entries, not evidence of damaged compiled kernels.

There is a wider declaration gap: 8 traced keys are absent from the
current sm86 plan, of which 4 already have stored entries
from other work. Cache availability alone can conceal a plan omission.

Before a later repair, make module/build inputs carry the same mode-dependent
augmentation and effective shapes, derive/deduplicate the needed keys, and check the
full forward/backward key set before any expensive build. Rebuild only genuinely
missing entries, then numerically validate them and repeat the six native compiled
benchmark cells. This record does not implement those changes.

## Evidence and limits

[Raw launch inventory](a6000-l384-cache-missing.json) includes each of the six targets,
all keys, and plan/cache membership. Job `1672504` ran on gpu04 / A6000.
The diagnostic used native benchmark setup, BF16, L384, A48, atom length 3072,
depth 1 and mask probability 0.125. Compilation was disabled and the wrapper mode
was `disable` to expose launch bodies to the recorder; calibration was disabled.
No kernel was tuned and the before/after JSON cache hashes were identical.

This is launch-key coverage under FakeTensor execution, **not** a new compiled
benchmark, a numeric check, a binary-cache audit, or proof about untested shapes and
settings. `present` means a stored entry exists, not that this inspection revalidated
its numerical accuracy or timing. The original compiled benchmark remains the
evidence for the two first-forward misses.

Host-local artifacts outside the repository:
`../team-gm/mw_l384_missing_audit.py` (diagnostic);
`../team-gm/mw-l384-modules/missing-audit.{json,out}`;
`../team-gm/mw-l384-modules/results.{md,json}` (original native measurements).
The native benchmark source snapshot was
`4fa017e5636bb2a393ebb1dfc3a7231de48bfe1644c76b35bb767d375bf2e8b3`.

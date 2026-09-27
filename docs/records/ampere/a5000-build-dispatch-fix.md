# A5000 build dispatch and resume correction

Date: 2026-09-12. Baseline: `fb6777417c0a34e383e0984978fec89169c18e28`.
This record describes the build correction. Full A5000 cache qualification is pending
the jobs below; this file does not assert completion of the GPU build.

## Cause and correction

The verified derivation disables LayerNorm and bias-only-attention dispatch
calibration. Build children previously left both selectors in automatic mode.
A stored per-GPU choice could therefore execute atomic LayerNorm instead of the
planned persistent backward, or split gate projection instead of the planned
fused projection. Successful module execution did not imply that its planned
keys had been measured. The A5000 recovery run reproduced this discrepancy in
Transition and triangle-attention logs.

Build children now set `layernorm_dispatch="off"` and
`biasonly_dispatch="off"`, matching derivation. Explicit backend pins still take
precedence. Production callers keep automatic dispatch. This does not change
the kernel bodies, config grids, or measurement timing regime.

The build also previously pruned its compiled cache immediately after merging
partial measurements, before checking required-key coverage and failed units.
Cleanup now occurs only after those checks pass. Failed builds retain the
compiled artifacts needed for resume. Partial measured shards continue to merge.

Related fixes are recorded in [the memory-access investigation](a5000-memory-access-fix.md):
the A5000 LayerNormLinear bad schedule is excluded, fatal warmup failures stop
the poisoned CUDA process, and partial shards are not accepted as completed units.

## Verification

- Focused child-dispatch and CLI-completion tests: 30 passed.
- Full `tests/builder` and `tests/autotune`: 1,461 passed, 1 skipped.
- Scoped Ruff and ty: passed. `git diff --check`: passed.
- Fresh sm86 derivation: 5,258 invocations, zero errors, 1,008 required keys
  across 53 kernels. Published as the package's canonical plan.
- Dispatch source identity: `0e050abb4594ebec9d2955b01cc97eee79454bacdf46cf9cb1755b910068454d`.

The derivation used an independent A6000 allocation only to provide the CUDA
device context required by fake dispatch. It executes no tuning kernels. All
A5000 measurements and qualification run on gpu02 A5000 allocations.

## Remaining scheduled work

- Job **1680274** continues the A5000 cache build. Its module pass finished with
  16 successful units and four expected OOM units; the 772-input alternative
  driver pass is running.
- Job **1680369**, dependent on completion of 1680274, runs the corrected build
  to fill dispatch-related gaps. It audits each alternative driver's named-target
  measurements, replays 22 module forward/backward cases with automatic runtime
  dispatch, and runs the existing GPU suite plus FP32 numerical/determinism tests.
- Only successful completion of those checks writes
  `docs/records/a5000-production-audit.{json,md}`. Unexpected failures stop the
  job and retain their logs under `.scratch/a5000-finish/`.

Atom augmented-attention training at A48 requires at least 24 GiB of mandatory
backward workspace at L4096 and 96 GiB at L8192. Six split/reduce keys blocked by
that workspace remain explicit capacity exclusions on the 24 GB A5000. The
standard CLI's strict full-coverage exit code is not changed to conceal them.
No smaller-augmentation measurements are substituted for these cases.

The four historical Xid31 faults remain unreproduced, with their original cause
unconfirmed; see [the targeted follow-up](a5000-mmu-followup.md).

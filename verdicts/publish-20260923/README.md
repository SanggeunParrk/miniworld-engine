# Post-2.0.0 integration checkpoint — 2026-09-23

This checkpoint publishes the tested module wiring and follow-up work after tag v2.0.0. The existing tag is unchanged; these are main-branch additions under Unreleased.

## Included

- Packaged native H100 bidirectional TriMul training and inference dispatch.
- D128 outgoing/incoming native training, including residual and dropout.
- D128 parameter preparation/layout improvements and optimizer-state migration helper.
- D64/256/384/512 bidirectional preparation reuse and compiled x_n metadata correction.
- Residual/dropout epilogues in OPM/PWA, Token DiT inference wiring and Transition compile guards.
- Selected native sources/includes, configuration selections, architecture-specific derived plans and attribution.
- Historical benchmark inputs/results and the HTML comparison, preserving workload/baseline distinctions.

## Limits

Wide TriMul kernels are connected but further performance tuning remains deferred. Single-direction native training is D128-only. This checkpoint is not a claim of exhaustive tuning or a rebuild of every GPU cache. Installing into MiniWorld's training environment and publishing to a package index are separate operations.

See [dispatch contracts](../../docs/operations/h100-module-wiring.md) and [comparison dashboard](../version-compare-20260923/index.html).

## Verification

- Layout checks excluding the pending regenerated sweep page: 195 passed, 6 skipped.
- Registry, launch binding, compile contracts and parameter layout: 1150 passed, 12 skipped.
- A wheel build succeeded; all 65 packaged TriMul Python/CUDA/include/config source files were present.
- Regenerated sweep-page checks: 3 passed.
- H100 GPU integration: 54 passed (job 16294; 867.14 seconds, including fresh JIT compilation).
- SM86: 8506 invocations, zero errors; SM90: 9106 invocations, zero errors.
- Final outcomes and wheel digest: [validation.json](validation.json).
- Revision-plan archives stay in Git, outside the wheel under existing packaging policy; wheel installs derive architecture plans in their user cache.

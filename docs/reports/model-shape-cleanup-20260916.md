# Model-derived build shapes — 2026-09-16

Scope: FoldForge AF3 default, Protenix v1 and v2, OpenDDE default, installed ESMFold2 checkpoint configuration, and MiniWorld debug/small/medium/large dimensions. Evidence: [constructor census](../checkpoint-shapes-20260915.json), connected module source and MiniWorld model configs. This is shape coverage, not a new performance or numerical-accuracy claim.

## Changes

- Default module registry: 141 rows → 107 rows. Dimensions are paired contracts (input/output/conditioning width, head count, expansion), never a Cartesian union of every width in a model.
- Removed unconfigured pair512 TriMul, bidirectional pair256/512, synthetic triangle head combinations, square AdaLN384/384, and unrelated diagnostic module probes. The kernels themselves and saved tuning measurements remain available.
- Kept real pair256 (Protenix v2 / ESMFold2), pair384 (OpenDDE), and projected single attention384 with both 8 and 16 heads.
- LayerNorm16 belongs to local atom pairs `(B, ceil(A/32), 32, 128, 16)`, not global token pairs or token singles. Fourier LayerNorm256 belongs to one noise embedding per diffusion sample; genuine pair256 norms remain separately declared. ESMFold2's biased Fourier norm is included.
- OpenDDE MSA uses 128 input channels and 8 heads × 8 projected channels. Protenix v2 has the same projected width with pair256; ESMFold2 retains its distinct 128-channel projection.
- `build all` no longer appends a driver pass for kernels no configured model reaches. Explicit `--per-op` diagnostics remain available; they are not the default model build.
- HTML requires a source-verified dispatch plan. It no longer silently substitutes a cross-product driver ladder when the plan is stale. Displayed paired kernel dimensions are decoded from the recorded cache keys.
- Actual build units preserve registry row identity, including augmentation. A dimensions-only lookup previously mapped both A=1 and A=5/48 norm rows onto the last row and silently omitted A=1 workloads. Real and fake unit identities are now checked for equality and uniqueness.
- Dispatch evidence fingerprints include `checkpoint_cases.py`, so a changed leaf layout invalidates an old plan.

## Preserved

- Token lengths: 128, 256, 384, 512, 640, 768.
- Atom lengths: 1024 through 8192, step 1024. Runtime length bucketing/clamping unchanged.
- Existing training/inference modes, precision and dispatch alternatives for retained shapes.
- Every tile/warp/stage candidate CSV, including cross-GPU candidates and existing A6000 winners. No schedule was removed based on A6000 measurements.
- Saved cache entries, shards, checkpoint files, and existing unrelated worktree changes.

The Protenix v1 asymmetric template TriMul64/128 remains a reference fallback; the fused TriMul accepts equal input/hidden widths. This cleanup does not advertise that fallback as a newly supported fused kernel.

## Validation

The isolated review copy passed 49 model contract, build scheduling and plan refresh tests on a cssb3 allocated CPU node, and the same 49 tests on the allocated gpu04 node. Full SM86 fake-tensor dispatch: 8,242 invocations, zero errors, 49 kernels and 1,331 keys. The previous registry declared 10,780 invocations; the reduction is 23.5%. Final code passed 195 regression tests (67 model/build/resume contracts and 128 cache/grid/registry/page checks), with zero failures. Ruff passed for every modified Python file.

`layernorm_bwd_atomic_triton` requires 87 actual keys (125,280 key × grid candidates before cache reuse), replacing the previous HTML assumption of 184 Cartesian driver cases (264,960 candidates). These are different counting bases, not a measured speedup. `transition_fwd_b2b_triton` is unreachable in this SM86 model plan and contributes no default build keys; its candidate CSV is preserved for explicit diagnostics and other dispatch policies.

Stream labels on a derived key identify its caller workload. An MSA module can launch a pair norm at D=256/384; these values are not MSA activation channel widths. No kernel tuning or latency benchmarking is part of this validation.

[Machine-readable changes](model-shape-cleanup-20260916.json) retain removed/reclassified rows and their reasons.

# Consolidated development main (2026-09-27)

All local branches and all fetched origin branches are ancestors of the consolidated
main. Main is the development entry point. Merged branch refs were subsequently
deleted except the two MPNN refs; historical worktree files remain detached.
This is the 2.0.0 consolidation snapshot; see ../releases/2.1.0.md for the new policy.

Included implementations: current upstream main, D64 input-buffer barrier, Triton
Norm dispatch/tuning, opt-in native Norm, TriMul wide-forward sources and Transition
experiments, TokenDiT BF16 SM90 training core, latest DiT overlap experiments,
TriangleAttention checkpoint18246 runtime, and the current MPNN branch.

Merge decisions:

- Preserve the current runtime/cache/compile policy rather than restoring the older
  runtime surrounding the research attention checkout. Its superseded files are in
  `experiments/legacy_h100_runtime/`, with `MERGE.json` provenance (archived 2026-09-27).
- Keep both DiT experiment runners: `experiments/token_dit_fused/tdit/runner.py`
  is the latest overlap runner; `runner_multistream.py` preserves the earlier variant.
  Shared scratch ownership is not assumed interchangeable between them.
- Keep the D64 barrier, explicit backend selection and FakeTensor guard. Old tuning
  files live in `experiments/transition_fused/records/cache_snapshot_20260927/`.
- Union the H100 and MPNN kernel registries and MSA/edge build sides. Keep the newer
  cache-grid compatibility check and MPNN's validation-before-pruning behavior.
- Superseded cache snapshots, wheel-path fixes and the pre-rebase MPNN branch are
  recorded in `experiments/legacy_branches/` (archived 2026-09-27, see
  [experiments/README.md](../../experiments/README.md#archived)); their history is merged without
  rolling current APIs back to obsolete implementations.

This integration is not a new GPU qualification. No GPU, benchmark, installation or
native compilation was run. TriangleAttention's checkpoint18246 snapshot remains
byte-identical for all 48 recorded files. Eleven of the twelve Norm manifest files
remain identical; MPNN adds an optional `row_bucket` argument to
`kernels/layernorm/compile_native.py`, preserving the default caller expression.
The previous manifest is retained as historical evidence, not restamped.

Native attention `.so` artifacts in the source tree are tied to the recorded Python,
PyTorch and SM90 ABI. Ordinary wheels ship the rebuild sources, not those binaries.
Use a compatible source checkout or rebuild on the target machine, then verify the
actual dispatch, F+B, changed-input graph replay, sanitizers and paired timings.

CPU validation and remaining failures are recorded in
`main-consolidation-cpu-20260927.json`. In particular, source-derived sweep plans and
Ampere LNLinear tuning caches have become stale, and research helpers do not yet
satisfy every existing compile/style contract. Do not rewrite identity hashes to
pretend those artifacts were regenerated. GPU-dependent derivation/qualification is
left for a machine with an available device.

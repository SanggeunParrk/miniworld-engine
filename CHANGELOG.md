# Changelog

All notable changes to the **public contract** (`miniworld_kernels.kernels`) are
recorded here. This project is consumed by team-gm as a pinned submodule, so
surface changes are semver-relevant. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

The public surface is frozen and enforced by `tests/test_public_api.py`.

## [Unreleased]

### Changed
- **Packaging**: runtime `[project.dependencies]` slimmed to the kernel core
  (`torch`, `triton`, `einops`, `jaxtyping`, `numpy`). Benchmark harness,
  comparison baselines, the CuTeDSL backend, and dev tooling moved to
  `[project.optional-dependencies]` extras (`bench`, `baselines`, `cute`,
  `dev`). A parent consuming `miniworld_kernels.kernels` no longer inherits
  lightning / hydra / cuequivariance / scipy / cutlass.

### Fixed
- Repo-wide int64 promotion of Triton `program_id` / loop-index pointer offsets
  (raw-indexing kernels), fixing an illegal-memory-access at large `L`
  (`augmented_attention` atom training, L≥768). `make_block_ptr` kernels keep
  int32 block offsets (Triton constraint); CUTE int32 (tmem addr, RNG seed)
  left as-is.

### Added
- Public API contract test (`tests/test_public_api.py`): freezes the
  `kernels` surface and asserts the package import is side-effect-free.

## [0.1.0]
- Initial consolidation of AF3-style op kernels (triangle multiplication,
  transition, triangle/bias/augmented attention, layernorm, adaLN) with
  Triton / CuTeDSL / CUDA backends, out of team-gm + FlashAttentionBias.

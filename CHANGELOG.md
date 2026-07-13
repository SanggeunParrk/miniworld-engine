# Changelog

Notable changes to the public API (`miniworld_kernels.kernels`) are recorded
here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

The public surface is enforced by `tests/test_public_api.py`.

## [Unreleased]

### Added
- **`miniworld_kernels.ops` — the whole-op consumer contract.** Complete,
  autograd-transparent model-layer ops (weights as arguments, backend dispatch +
  fwd/bwd inside), consumed as a single call: `triangle_multiplicative_update`,
  `triangle_attention`, `transition`, `conditioned_transition`,
  `augmented_attention_pair_bias`, `layer_norm_linear`, `layer_norm`. Lazy,
  side-effect-free import; pinned by `_OPS_CONTRACT` in `tests/test_public_api.py`.
  Verified fwd+bwd vs the pytorch/cuequiv reference on B200 (≥0.9998).

### Changed
- **API framing**: `ops` is now the supported **consumer** surface; `kernels` is
  reframed as the **internal primitive** surface (per-GEMM/LN/gate/attention units)
  out of which the ops are built — still pinned (`_CONTRACT`) for internal stability
  but not intended for model code. `triangle_multiplicative_update` moved from
  `kernels` to `ops` accordingly.
- **Packaging**: runtime `[project.dependencies]` slimmed to the kernel core
  (`torch`, `triton`, `einops`, `jaxtyping`, `numpy`). Benchmark harness,
  comparison baselines, the CuTeDSL backend, and dev tooling moved to
  `[project.optional-dependencies]` extras (`bench`, `baselines`, `cute`,
  `dev`). Installing the package for `miniworld_kernels.kernels` no longer pulls
  lightning / hydra / cuequivariance / scipy / cutlass.

- Moved 21 unreferenced dev/probe/experiment kernel files (perf probes,
  autotune/tile sweeps, micro-benchmarks, superseded variants) out of the
  importable `src/` package into `research/` (convention: `src/` ships only
  the canonical path). Verified 0 importers + contract & numerical suites green.

### Fixed
- Repo-wide int64 promotion of Triton `program_id` / loop-index pointer offsets
  (raw-indexing kernels), fixing an illegal-memory-access at large `L`
  (`augmented_attention` atom training, L≥768). `make_block_ptr` kernels keep
  int32 block offsets (Triton constraint); CUTE int32 (tmem addr, RNG seed)
  left as-is.

### Added
- Public API contract test (`tests/test_public_api.py`): freezes the
  `kernels` surface and asserts the package import is side-effect-free.
- Numerical correctness suite (`tests/test_numerical.py`): each op's fused
  MINIWORLD backend vs the PyTorch reference (forward + input gradient),
  GPU-gated, asserting the fused path is actually engaged (no silent
  dtype-degrade). Promotes the benchmark cosine checks into an enforced
  correctness gate.
- CI (`.github/workflows/ci.yml`): ruff + pyright + CPU contract/guard tests
  on every push/PR (nested cutlass submodule skipped, package installed
  `--no-deps`). GPU numerical suite runs via `pixi run test-gpu` on an
  allocated node.

## [0.1.0]
- Initial consolidation of AF3-style op kernels (triangle multiplication,
  transition, triangle/bias/augmented attention, layernorm, adaLN) with
  Triton / CuTeDSL / CUDA backends.

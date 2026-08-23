# Changelog

Notable changes to the public API (`miniworld_engine.kernels`) are recorded
here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

The public surface is enforced by `tests/test_public_api.py`.

## [Unreleased]

### Added
- **`miniworld-engine audit`** — verifies the build system and, new, that every
  DECLARED `(op, dtype, shape-bucket)` is present in the shipped autotune cache.
  Declared means `registry.csv` crossed with each kernel's `level` and `dtypes`,
  so a hole is reported against the contract rather than against whatever the
  last build happened to measure. `build` already told users to run this; the
  subcommand did not exist and `build/audit.py` crashed on import.
- **`settings.autotune_miss_cap`** (default 24) — how many configs an autotune
  cache MISS may search before falling back to a heuristic subset.
- **`miniworld_engine.ops` — the whole-op consumer contract.** Complete,
  autograd-transparent model-layer ops (weights as arguments, backend dispatch +
  fwd/bwd inside), consumed as a single call: `triangle_multiplicative_update`,
  `triangle_attention`, `transition`, `conditioned_transition`,
  `augmented_attention_pair_bias`, `layer_norm_linear`, `layer_norm`. Lazy,
  side-effect-free import; pinned by `_OPS_CONTRACT` in `tests/test_public_api.py`.
  Verified fwd+bwd vs the pytorch/cuequiv reference on B200 (≥0.9998).
- Public API contract test (`tests/test_public_api.py`): freezes the
  `kernels` surface and asserts the package import is side-effect-free.
- Numerical correctness suite (`tests/test_numerical.py`): each op's fused
  MINIWORLD backend vs the PyTorch reference (forward + input gradient),
  GPU-gated, asserting the fused path is actually engaged (no silent
  dtype-degrade). Promotes the benchmark cosine checks into an enforced
  correctness gate.
- CI (`.github/workflows/ci.yml`): ruff + ty + the whole CPU suite on every
  push/PR (nested cutlass submodule skipped; CPU torch wheel installed so the
  type gate can see the stubs it is checking against). `ty`, not pyright, which
  cannot parse jaxtyping shape strings; the step gates, with no `|| true`. GPU
  numerical suite runs via `pixi run test-gpu` on an allocated node.

### Changed
- **`ty` is a gate, not a report.** CI ran `ty check src tests || true` against a
  job that installed the package with `--no-deps`, so torch and triton were
  absent, every `import torch` was an unresolved import, and every torch
  attribute was `Unknown` — the step could not see the types it was checking. It
  now installs the CPU torch wheel plus `[dev]`, checks `src tests benchmarks`,
  and has no `|| true`. The `dev` extra names `ty` instead of `pyright`, which
  cannot parse jaxtyping shape strings and reported 144 parse errors and no real
  findings.
- **`pixi run test` had been dead since the checkout was renamed**
  (miniworld-kernels → miniworld-engine): the `pytest` launcher script carries an
  absolute shebang. The tasks use `python -m pytest`, and `pixi run ci` runs the
  type gate like the CI job does.
- Removed `autotune.elem_bucket_of`, a factory for the per-kernel `bucket_of`
  objects deleted in fcd3c7a. No caller, and the `_miniworld_keys` attribute it
  documented as "lets build.audit introspect the extractor" had no reader.
- **An autotune cache miss no longer sweeps the full grid.** Triton kernels now
  narrow a miss to `autotune_miss_cap` configs centred on `num_warps` in {4, 8}
  and `num_stages` in {2, 3, 4}; the shipped grid is 205,266 configs, so the old
  behaviour put a tuning sweep inside the first production forward on any GPU
  without a cache. A build (`run_autotune=True`) still gets the whole space.
  The warning text now names the fallback that actually happened instead of
  always claiming "the full autotune grid".
- **Autotune cache entries are keyed on the toolchain and the kernel source**,
  not only on the config grid: new `env_identity` (triton/torch/CUDA/ptxas) and
  `op_identity` (kernel source + `key=[...]`) fields. Entries written before
  these existed still read, so committed caches do not become permanent misses.
- **API framing**: `ops` is now the supported **consumer** surface; `kernels` is
  reframed as the **internal primitive** surface (per-GEMM/LN/gate/attention units)
  out of which the ops are built — still pinned (`_CONTRACT`) for internal stability
  but not intended for model code. `triangle_multiplicative_update` moved from
  `kernels` to `ops` accordingly.
- **Packaging**: runtime `[project.dependencies]` slimmed to the kernel core
  (`torch`, `triton`, `einops`, `jaxtyping`, `numpy`). Benchmark harness,
  comparison baselines, the CuTeDSL backend, and dev tooling moved to
  `[project.optional-dependencies]` extras (`bench`, `baselines`, `cute`,
  `dev`). Installing the package for `miniworld_engine.kernels` no longer pulls
  lightning / hydra / cuequivariance / scipy / cutlass.

- Moved 21 unreferenced dev/probe/experiment kernel files (perf probes,
  autotune/tile sweeps, micro-benchmarks, superseded variants) out of the
  importable `src/` package into `research/` (convention: `src/` ships only
  the canonical path). Verified 0 importers + contract & numerical suites green.

### Fixed
- **The library needed an environment variable to work at all.** With
  `MINIWORLD_CONFIG_DIR` unset, every op registered an empty config list, triton
  substituted its own `Config({})`, and the first launch of every triton kernel
  died with `TypeError: dynamic_func() missing 2 required positional arguments`.
  Every sbatch script and bench entry point in the repo exports the variable, so
  the failure only ever showed up outside the scaffolding — measured on an A6000
  with it unset, `miniworld` was a `status=failed` row next to a healthy
  `pytorch` one; it is now 1.795 ms against pytorch's 7.084 ms.
  `autotune.configs.default_config_dir()` selects `grid` when nothing is set.
- **The `pre_hook` timer never ran.** `_install_launch_probes` assigned
  `Autotuner.pre_hook`, but triton sets that attribute per instance in
  `__init__` and the class carries no default, so the `hasattr` guard never
  opened and an instance would have shadowed the class attribute anyway. The
  build report printed `pre_hook 0 x -> 0s` for the whole life of the feature.
- **`bench_triangle_attention` crashed on the path that guards it.** Its
  unsupported-`old_triton` branch returned `BenchResult(status=..., error=...)`;
  `BenchResult` is a NamedTuple with neither field, so the guard raised
  `TypeError` instead of the NaN row every other bench returns.
- **A `str` appended to quack's `EXTRA_SOURCE_DIRS: list[Path]`.** The
  membership guard therefore never matched, and every import of
  `kernels/_quack_compat.py` added another copy of the same directory to the
  list the cute JIT cache key hashes.
- **`modules.primitives` imported scipy at module scope** for one weight
  initializer. scipy is in the `baselines` extra, not the core, so a core-only
  install could not import `miniworld_engine.modules.primitives` — or anything
  built on it.
- **`layernorm_linear_pytorch` declared `ln_bias: torch.Tensor`** while both of
  its callers pass `None` for a LayerNorm without beta, which `F.layer_norm`
  accepts.
- **`audit` failed the one kernel whose autotune key correctly carries no
  shape.** `transition_fold_triton` reads only the weights, so `N` and `K` are
  its whole shape and one cache bucket is right; the builder already knew and
  drove it at one length, while the audit reported `key is pinned` against a
  correct build. Both now answer through `builder._keys_on_shape`.
- **`kernels.cuda_transition` has never worked and now says so.** Its body
  deferred to `transition.cuda.cuda_transition`, a symbol with no history in
  this repo; the module binds only `cuda_transition_b2b` and
  `cuda_transition_expand_gate`. `Transition(implementation="cuda")` resolves to
  `KernelBackend.CUDA` and calls it in `forward`, so that path raised
  `ImportError` from four frames down. It now raises `NotImplementedError`
  naming what does exist. The name stays (the surface is frozen) but the
  CUDA backend for `Transition` should be considered unavailable.
- **`miniworld-engine build all` works as documented.** Its config-set argument
  defaulted to the string `"default"` and `configs/default` has never existed,
  so the command failed at argument parsing; `bench_module` hardcoded the same
  string. Both now default to the full search grid.
- **The per-op sweep never merged its shards.** `build all` ran to completion,
  wrote its shards, and left `data/` untouched — the build reported success and
  shipped nothing. It also refused to merge at all if any single unit failed,
  discarding every good measurement in the run.
- **`build all` never drove fp32.** The work list ignored `registry.csv`'s
  `dtypes` column and emitted bfloat16 for every unit, so the fp32 half of 66
  kernels was never tuned — and the coverage check counted `(op, bucket)` with
  no dtype axis, so it reported full coverage over a half-built cache.
- **`level=both` kernels built pair activations at atom shapes.** 18 kernels
  asked for `M = L*L` where production hands `M = A` — 67,108,864 rows at
  L=8192 — which is the whole explanation for the sweep's CUDA OOMs and one
  int32 offset overflow, and it poisoned those kernels' atom-bucket entries.
- **Shard writes are atomic.** A bare `write_text` let two workers interleave on
  one shard and produce unparseable JSON, which the merge then dropped silently,
  losing a whole unit's measurements with nothing said.
- Repo-wide int64 promotion of Triton `program_id` / loop-index pointer offsets
  (raw-indexing kernels), fixing an illegal-memory-access at large `L`
  (`augmented_attention` atom training, L≥768). `make_block_ptr` kernels keep
  int32 block offsets (Triton constraint); CUTE int32 (tmem addr, RNG seed)
  left as-is.

## [0.1.0]
- Initial consolidation of AF3-style op kernels (triangle multiplication,
  transition, triangle/bias/augmented attention, layernorm, adaLN) with
  Triton / CuTeDSL / CUDA backends.

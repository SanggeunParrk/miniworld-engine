# Changelog

Notable changes to the public API (`miniworld_engine.kernels`) are recorded
here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

The public surface is enforced by `tests/test_public_api.py`.

## [Unreleased]

### Added
- **`miniworld-engine dev audit`** — verifies the build system and, new, that every
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
- **One vocabulary for benchmark targets, and a level to hold it.** `bench.py`'s
  targets lived in one flat dict, which forced the kernel-level ones to abbreviate
  around the module-level ones: `tri_attn`, `bias_attn`, `aug_attn`, `ln_mask`,
  `gate_bwd`, `gemm_epil`. `bench_kernel triangle_attention` — the family's own name —
  came back "unknown target". `BenchConfig.kernel` is now `target` + `level`
  (`kernel` | `module`), the two levels are separate namespaces, and every target is
  spelled the way the engine spells it: a kernel target names its family in
  `kernels/registry.csv`, a module target names the module it constructs. So
  `triangle_attention` is now a legal name at both levels and means the right thing at
  each. Renamed: kernel `tri_attn`→`triangle_attention`, `bias_attn`→
  `bias_only_attention`, `aug_attn`→`augmented_attention`, `ln_mask`→`fused_ln_mask`,
  `gate_bwd`→`gemm_gate_bwd`, `gemm_epil[_bwd]`→`gemm_epilogue[_bwd]`,
  `dual_gemm_epil[_bwd]`→`dual_gemm_epilogue[_bwd]`, `cond_transition_tail`→
  `conditioned_transition_tail`; module `bias_only_attention`→`attention_pair_bias`
  (it benches `AttentionPairBias`, and the old name belongs to the kernel family);
  build cases `*_bidir`→`*_bidirectional`, `tm1_triton`/`tm2_triton`→`tm1`/`tm2`;
  implementation labels `triton_tri_attn*`/`triton_bias_attn`/`triton_aug_attn`/
  `aug_attn_memory_efficient` spelled out. The 120 committed result tables that carried
  an old name in their `run_name`/`target`/`implementation` columns were rewritten in
  place; **no measured value changed** (checked cell by cell), and the 40 plots whose
  drawn title named the old target were re-rendered from those same tables.
  `tests/test_bench_target_vocabulary.py` now holds the four name spaces —
  bench.py's tables, the CLI's, `builder.CASE_NAMES`, and the directory tree — to
  each other.
- **Each bench target loads its own config.** `@hydra.main(config_path=...)` was the
  constant `../modules/triangle_multiplication/configs`, so every run — kernel, module,
  atom — loaded that one file and the other 25 `configs/bench.yaml` were read by
  nothing. They disagreed with what ran: `augmented_attention_atom` declares a 128–384
  ladder and was swept at 384–1024, while its own committed tables show 128/256/384.
  The path is now computed from the `target=`/`level=` overrides before hydra starts,
  every one of the 26 targets owns a `configs/bench.yaml` (the 17 kernel targets' are
  copies of the base they already loaded, so nothing they measure changed), and a
  target with no config is an error instead of a silent fall back to another target's
  ladders.
- **`bench_module all` means all of them.** The "all" group read a table that
  `triangle_multiplication_bidirectional` had never been added to, so it ran eight of
  the nine module targets. The bench-args table and the build-case table — keyed by the
  same names, maintained apart — are now one `MODULE_TARGETS`.
- **Coverage no longer guesses a target's directory.** `_report_coverage` rebuilt the
  path from `target in KERNEL_BUILD_CASES`, which was wrong for
  `augmented_attention_token`/`_atom`: they shared one directory named after neither, so
  the lookup missed and both targets' kernels were reported as never launched. Each
  target now owns exactly one directory and the path is derived from `level`.
- **`miniworld-engine build <typo>` fails immediately.** It used to resolve the config
  set and import every kernel — minutes of triton compilation — before saying "unknown
  case". `builder.CASE_NAMES` is a declared tuple (`test_case_names_are_declared` pins it
  to `cases()`), and the per-op name space is read straight from `registry.csv`, so both
  are checked before the first import.
- **Every model-level op is a folder.** `modules/__init__.py` has always opened with the
  rule; `attention_pair_bias.py`, `msa_pair_weighted_averaging.py` and
  `swa_atom_attention.py` were flat files. They are now packages like the other eight,
  and `modules/ops.py` — one level below `miniworld_engine.ops`, the public whole-op
  contract, and meaning the opposite thing — is `modules/functional.py`.
  `tests/test_module_layout.py` holds the rule and the four shared modules that are
  legitimately flat.
- **The linter now checks what the code was already written against.** `select` was
  `["E", "F", "W"]` while the source carried ~700 `# noqa:` comments naming PLC0415,
  BLE001, ANN001, SLF001, S603, ARG005 and two dozen other rules that were not
  enabled — so none of them suppressed anything, and nothing said so. The set is now
  a deliberate one (import sorting, bugbear, pyupgrade, simplification, perf, pytest
  and logging idioms, RUF including RUF100) with every excluded family named and
  justified in `pyproject.toml`, and it runs clean. 264 dead directives were turned
  back into plain comments, keeping their reasons — ruff's own RUF100 fix deletes the
  whole comment, and the reason is the part worth having. Relative imports are banned
  outright (`ban-relative-imports = "all"`): 137 of them became absolute, which is the
  form that broke this session when a flat module became a package and `from .ops
  import sigmoid_gate` silently pointed somewhere else. `[tool.ruff] src = ["src"]`
  was missing, so ruff's own fix for those rewrote them to `src.miniworld_engine.*`.
  Two real defects surfaced: an `assert sig is None or True` that could never fail,
  and eleven late-bound loop variables captured by autotune lambdas in
  `autotune/build.py` (harmless today because each lambda is called in its own
  iteration, one refactor away from tuning every bucket against the last shape).
- **`tools/` is in both gates.** It was tracked code that `ruff check` and `ty check`
  never looked at, in the pixi tasks and in CI alike.
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
- **A `level=both` kernel keyed its cache on LENGTH, so its two sides shared a
  bucket.** A pair activation `(B, L, L, D)` at L=1024 and an atom activation
  `(B, A, D)` at A=1024 both have `shape[-2] == 1024`, so both landed in
  `shape_key=1024` — while the first launches 1,048,576 rows and the second
  launches 1,024. Visible in the shipped A6000 cache as two adjacent lines of one
  op: `shape_key=384 → 0.6103 ms` (pair, 147,456 rows) next to
  `shape_key=1024 → 0.0215 ms` (atom, 1,024 rows). Since the driver was corrected
  to build atom activations above L=512, the sweep measured only the atom side
  there and the module bench, which runs the pair side, was served those configs:
  transition on an A6000 at L=1024 in its committed baseline's exact configuration
  went from 5.498 ms to 9.504 ms while the pytorch reference reproduced within 3%.
  `both_key` now takes the ROW count (`shape_key.BOTH_ROWS`), a both-level kernel
  is two work lists with an explicit `side`, and `cache.KEY_SCHEME` invalidates the
  entries the change re-based — level-aware, so the 68 token/atom kernels whose
  buckets did not move keep theirs.
- **The `compiled` column of every benchmark CSV recorded the request, not what
  ran.** Four of the eight module benches guard `model.compile()` with
  `and conf.cudagraph == "disabled"`, so `compile=true cudagraph=manual` runs eager
  for those; `actual_compiled_flag` knew about `transition` by name and missed the
  other three. 330 committed tables say `compiled=True, cudagraph=manual`, and for
  triangle_multiplication, triangle_attention and bias_only_attention that is a
  measurement of eager code.
- **42 kernel-bench rows could not run at all** — the harness pre-flattened
  activations that seven entry points read their cache key off, so `triton_tm1`,
  `triton_tm2`, `triton_atomic/partial/persistent`, `triton_cond_transition` and
  `layernorm_linear_triton` came back `failed: shape ... is already flattened`.
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

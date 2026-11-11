# Requirements

`src/miniworld_engine/requirements/` states what the engine must serve. The rules are in
[requirements/README.md](../src/miniworld_engine/requirements/README.md); this page summarises them and shows who reads the tables.

## requirements_module.csv

One row per distinct (module class, stream, dims, options):

| column | meaning |
|---|---|
| `module` | a `BackendModule` class of `miniworld_engine.modules` (found by name, subpackages included) |
| `model` | the selector keys that build this class (`TokenDiT(model=...)`, `AtomDiT(model=...)`, `TriangleAttention(model=...)`, ...) |
| `stream` | the activation layout of a row-wise op: `token_pair` `[S, L, L, w]`, `msa_token` / a depth `[S, D, L, w]`, else `[S, L, w]`; the DiT blocks' multi-fold layout `token_folds` / `atom_folds` `[S, F, L, w]` with F = the `depths` value (docs/modules.md, multi-fold contract) |
| `lengths`, `depths` | `|`-separated token / atom lengths and MSA depths (the fold count F on a `token_folds` / `atom_folds` row) |
| `dims`, `options` | constructor kwargs `a=1;b=x`; `options` only where they change the math (for `TokenDiT*` / `AtomDiT*` only values differing from the class default); `a=False|True` means one variant per value |
| `dtypes` | `float32` and/or `master` (fp32 parameters, bf16 activations, forward under bf16 autocast). No row says `bfloat16`. The trunk-side classes (TriangleMultiplication*, TriangleAttention*, OuterProductMean, MSAPairWeightedAveraging, the Bidirectional* classes, PairformerBlock, the trunk Transitions) are `master` only; the five diffusion-head conditioning Transition rows (n = 2: pair 128 / 256 / 512, single 384 / 768) are `float32|master` |
| `modes` | `eval` and/or `train` |
| `eval_samples`, `train_samples` | the sample (batch) counts per mode |
| `source` | who builds the row: FoldForge / MiniWorld models, or `engine-own` |

Rows with equal (module, stream, dims, options) are merged: union of lengths, dtypes, modes and selector keys; all sources listed.

## requirements_kernel.csv

`name,family,modules,role,version`: every kernel a module backend launches.

- `name`: the kernel; a suffix `_sm90` / `_sm90a` (H100) or `_sm80` (A100) marks a card-specific build, no suffix a Triton or card-independent kernel.
- `family`: the directory under `kernels/` (also the launch-config tables it reads, `configs/<card>.json`).
- `modules`: `|`-separated module classes whose backend launches it.
- `role`: `forward` (both paths), `forward_infer` (no-grad only), `forward_train` (writes what only a backward reads), `backward`.
- `version`: human-managed. Bump it when the kernel's algorithm changes a result (not for a refactor, a comment or a rebuild). The
  judge then owes every case of its modules in the modes its role serves.

A kernel has no range or tolerance of its own: both come from its module rows.

## Derive rule

| role | module rows | modes |
|---|---|---|
| `forward` | every row of its classes | the rows' modes |
| `forward_infer` | rows with `eval` | eval |
| `forward_train` | rows with `train` | train |
| `backward` | rows with `train` | train |

The kernel's lengths, dims and dtypes are the union over those rows.

## Who reads the tables

| reader | uses |
|---|---|
| `measure/config.py` | `cases(cfg)`: rows -> cases (row, constructor variant, length, depth, samples, dtype, mode); `versions(module, mode, gpu)`: the class version, its `backend_versions`, the TF32 rule, the versions of the kernels listed for the class whose role serves the mode (a kernel with a card suffix `_smNN` only on that card), the card's table versions of their families |
| `kernels/_prebuild.py` | `family_rows(family)`: the rows a family's kernels must serve, by the derive rule; each family's `prebuild(target, rows)` expands its build variants from them |
| `kernels/_pretune.py` | the speed config's cases: the launches whose configs are tuned |
| `judge` | the canonical configs' cases: which cases must have a token |

## Accuracy rule (summary)

For every row and every (dtype, mode) an engine backend claims: `tol = max(err_pytorch, fixed[dtype])`; GREEN `<= tol`,
YELLOW `<= 5 tol`, RED beyond (NaN / inf is RED). Details in [measure.md](measure.md). The rule binds only the engine's backends
(`cuda`, `triton`); `anthropic` / `cueq` are measured the same way but their light is informational. Every backend, the reference
included, runs with TF32 allowed for float32 matmuls (`measure/config.py` `TF32_BACKENDS`); the gold is float64.

An engine backend must cover every row / dtype / mode it is the backend for; a gap is a `_supports_<backend>` refusal and is
visible as `unsupported` in the records.

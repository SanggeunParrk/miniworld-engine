# One rule for what must be redone: per-graph identities for the native caches and the derivation (2026-11-01)

Branch `feat/code-graph-triton`, on top of `0c67cea70` (uncommitted). Code: `src/miniworld_engine/autotune/identity.py` (new), `native.py`,
`plan.py`, `derive.py`, `builder.py`, `cache_status.py`, `cli.py`, `build/{repo_graph,code_graph_extra,kernel_bundle}.py`. Explanation of
the rule: `docs/kernels/prebuilt-kernels.md`.

## What changed

Two hand-made identities forced repeated work for kernels that had not changed:

| before | cost of any edit under `kernels/` |
|---|---|
| `native.source_identity()`: one hash of every `kernels/**/*.py` and `*.cu` + 4 autotune files; stored in each `*_cuda` cache as `op_identity` | a restamp of all 7 native caches |
| `plan.source_identity()`: one hash of `modules/`, `kernels/`, `integrations/`, registry CSVs, 5 autotune files; stored in `units.json`; also the directory name of every plan (`plans/<arch>/<hash>/`, the user-cache plan, the builder's resume generation) | a full `dev derive` (1252 units, a GPU), the sweep page, and `plan.load()` refusing the evidence |

Now one rule: an artifact stores the digest of the part of the graph it depends on and is stale iff that digest differs from the current one.

* **Native cache**: `native.op_identity(op)` = digest of the op's registry row, the closure of its files and the safety set. The closure
  starts at the row's file and driver and every python file that names the kernel file, follows python imports (function-level too),
  `#include` edges and the paths / `miniworld_engine.x.y` names written in string literals (not comments, not docstrings), and stays
  inside `kernels/`. `.py` by bytes, `.cu` / `.cuh` after the `_codegraph` normalization. `cache_status.scan`, `derive.coverage`,
  `native.choose_config` (the stored `op_identity`, the history key, the measurement record) and `cache.measurement_mismatch` (through
  its callers) compare the per-op value. `native.source_identity()` stays as a salt for `native_compile`'s scratch directory only.
  Native closure sizes: 99 files (layernorm), 109-111 (transition), 114 (trimul) -- of which 20 are the safety set -- against the 727
  files of the old hash; the families in a closure are layernorm, layernorm_linear, transition, trimul_inproj (the drivers import each
  other), none of adaln, augmented_attention, triangle_attention, rope, swa_dit, mpnn, ...
* **Derivation**: `registry_kernel.units.json` is schema 2. Per unit: keys (as before), launch counts (so the CSV's `units` column is
  what it was), and the set of files it depends on, each with a token (bytes + syntax tree). Evidence while a unit is recorded:
  `sys.monitoring` `PY_START` / `PY_RESUME` (every thread, so the autograd backward threads too; Python >= 3.12), a wrapped
  `builtins.__import__` / `importlib.import_module` (executed lazy imports, string-dispatched modules), the audit hook `open` (CSV / JSON
  read), the launched kernels' registry file / JIT helpers / driver, the module-level import closure of everything that ran (constants
  flow through imports without a call), and the safety set. `dev derive` reuses every unit whose stored files all match (`--prior`,
  default the file next to `--out`) and records the rest; the entries are rebuilt from per-unit keys, counts and unit fields, the same
  function for a recorded and a reused unit. `plan.load()` is valid iff complete, error free, registry identity and arch match and no
  stored unit has a changed or vanished file (or a safety file it does not list). File table stored once (`file_index`,
  `file_tokens`), units hold indices (`unit_files`): 292 files, 88-163 files per unit (median 103), 1.34 MB. The plans are no longer a
  directory per source hash (`plans/<arch>/registry_kernel.*`, user cache `plans/<arch>/registry_kernel.csv`); the builder's resume
  generation is per unit (`builder._unit_scope`: the unit's op closure, or the files the verified derivation recorded for the module unit).
* **Safety set** (every identity): `settings.py`, `modules/dispatch.py`, `autotune/{cache,shape_key,policy,builder,derive,checkpoint_cases,
  module_registry,native,fused_config,hopper_cuda_config,native_history}.py`; a unit also `registry.csv`, `registry_module.csv`,
  `build/gpu_to_kernels/*.csv`. Deviation from the brief, on purpose: a native op takes its own `registry.csv` row (as the graph already does:
  `registry-row:<op>`), not the whole file, and not `registry_module.csv` / `gpu_to_kernels` -- they select which units get built, never what a
  measured candidate is, and a new kernel row should not restamp seven caches.
* **Tools**: `dev restamp-native [--op OP] [--dry-run|--check]` (rewrites the stored identity, no measurement, replaces
  `fix1030/restamp.sh`); `dev derive --prior / --no-reuse`; `dev code-graph --diff BASE --new-only`, `--tests [cpu|gpu|all]` (one path per line,
  reached through lazy imports too: `pytest $(... --tests cpu)`). The derive writes its evidence to `<evidence>.new` and publishes it after a
  clean run (`.failed.json` on error): a failed run no longer overwrites the committed evidence with an incomplete file.

## Migration (once, no re-measuring)

1. Native caches: `dev restamp-native` rewrote `op_identity` (every occurrence of the old string in the file) of the 7 committed H100
   `*_cuda` caches (the identity moved three times while this change was being written because the safety set holds files that were
   being edited; the committed values are the last). `git diff --stat` shows equal insertions and deletions. Nothing was measured.
2. `dev derive --arch sm86 --workers 8` (job 40319, 8 workers, 1 GPU): 1252 units recorded in 59 s, schema 2. The registry CSV
   re-derived from scratch is **byte-identical to HEAD** (`git diff` empty): no kernel key moved.
3. `autotune/plans/<arch>/<hash>/` (two historical source hashes per arch, schema 1, excluded from the wheel) removed from the working
   tree (history is in git); `plans/sm86/registry_kernel.*` is the new plan (a copy of the registry files, identical bytes). **`plans/sm90`
   was not regenerated**: `dev derive --arch sm90` on this H100 fails from unit 69 on (augmented_attention: "data is not allocated yet") and waits on a prebuilt-build
   lock (jobs 40509, 40510, cancelled) -- the card's real capability meets the patched one in `install_native_recorders`. It needs a
   non-Hopper card (sm86 patches to sm90 there); until then `plan.ensure("sm90")` derives into the user cache.

## Measured reuse

Double run (job 40319): second `dev derive --arch sm86 --workers 8`: **1252 reused, 0 recorded, 2 s**; `registry_kernel.csv` and
`registry_kernel.units.json` byte-identical to the first run (md5 equal). (The units are written in label order; a parallel run files
them in completion order, so before sorting the two files differed.)

One-file edits against a tree snapshot taken before them (jobs 40523 and 40919; `dev code-graph --diff <tree> --check --new-only`,
`dev restamp-native --check`, `dev derive` into a scratch directory with `--prior` = the committed evidence; each file restored by hand
afterwards, `git status` clean for them):

| edit | `--new-only` reports | native caches owed a restamp | `dev derive`: units reused / recorded | CSV |
|---|---|---|---|---|
| comment in `kernels/rope/triton/main.py` | cosmetic, 0 pending | 0 of 7 | 1252 / 0 | identical |
| code in `kernels/rope/triton/main.py` (a module-level assignment) | 1 pending: 16 of 1252 units | 0 of 7 | 1236 / 16 | identical |
| code in `kernels/trimul_inproj/cuda/h100_inference.py` | 1 native op stale | 1 of 7 (`trimul_fwd_sm90_cuda`) | 1252 / 0 | identical |
| comment in `transition_b2b_kernel.cu` | cosmetic, 0 pending | 0 of 7 | 1252 / 0 | identical |
| code in `transition_b2b_kernel.cu` (a `__device__` global) | 1 native op stale | 1 of 7 (`transition_fwd_b2b_sm90_cuda`) | 1252 / 0 (no sm86 unit launches an sm90 op) | identical |

Before this change each of the five edits owed all 7 restamps and a full derive of 1252 units (the comments too: both hashes were over raw
bytes).

## Tests (every CPU run on a node-local copy, parallel per directory; `dev code-graph --diff main` as its own job before each submit)

* Main's baseline (`git archive main`, job 40315): 7 failures -- `test_a_manifest_says_when_and_against_what_it_was_produced`,
  `test_no_target_directory_is_left_without_a_config`, `test_every_target_owns_exactly_one_results_directory[kernel|module]`,
  `test_every_message_we_own_is_still_emitted`, `test_every_boolean_key_flag_is_tuned_on_both_sides`,
  `test_no_committed_cache_is_stale_on_a_gpu_we_can_rebuild`.
* This tree (job 40922; autotune, builder, compile, kernels, layout, mpnn, registry): the **same 7** and nothing else
  (autotune 1590 passed, builder 323, compile 691, kernels 155, layout 224, mpnn 54, registry 1445). The two git-dependent files
  (`test_repo_graph.py`, `test_kernel_code_graph.py`) run in the checkout (job 40923): 46 passed.
* New: `tests/cpu/autotune/test_incremental_derive.py` (trace of a unit incl. another thread, an executed import, the recorded set holds
  every executed file, full run == incremental run on a two-module fixture including the registry `units` counts and byte-equal
  evidence, a comment edit / unrelated file reuses everything, a new safety file stales everything, an edit during the run is refused,
  `dev derive` twice rewrites identical files); updated: `test_plan_refresh`, `test_takeover_integrity`, `test_build_resume_provenance`,
  `test_native_*`, `test_repo_graph`, `test_kernel_code_graph`, `test_native_identity_matches_the_wheel`, `test_v210_native`,
  `test_engine_native_*` (the old assertions about the global identities now assert the per-graph rule).
* ruff on every touched file (job 40924): only `test_v210_native.py:107` and `:120`, both present on main.
* GPU sanity (job 40921, 1 H100, warm cache): `tests/gpu/sm90/test_h100_trimul_batch_gpu.py` + `test_h100_weight_pack_gpu.py`: 47 passed. The
  committed native caches compare equal to the current per-op identities and `measurement_mismatch` is empty for all 7 (job: `native_cmp`).

## Where the rule can still under-approximate

1. **String dispatch the closure cannot read**: a module named by an f-string or built with `getattr`, and `kernels/__init__.py`'s lazy
   table (relative names). A native op's closure is static; a derivation unit is saved by the executed-import wrapper, a native op is not.
2. **Native closure stops at `kernels/`**: `autotune/capture.py` (timing), `native_compile.py`, `configs.py` are outside it and outside the
   safety set (as in the old identity); a change to how a candidate is timed does not stale a native cache (the cache's `measurement`
   record carries the regime and bench settings). `integrations/`, `modules/` are not followed from a native op.
3. **Constants through a hub**: the module-level import closure covers a constant imported at module level; a constant reached only through
   a hub's function-level import, in a module with no executed function, is covered only if that import statement ran in the unit.
4. **Files not seen**: data files are seen by the audit hook on `open`, not by `listdir` / `glob`; a new file in a directory a unit globs
   (other than `gpu_to_kernels`, which is in the safety set) is not seen. `sys.monitoring` needs Python 3.12: `dev derive` refuses on 3.10 / 3.11
   (so `plan.ensure` on such an interpreter fails with "kernel plan derivation failed" where it used to derive).
5. **Toolchain and environment are not in the evidence**: torch / triton / CUDA versions, environment variables and `settings` values a
   unit reads are not hashed (the native caches keep `env_identity`; the derivation never had one). A triton upgrade can change a
   bucket key without staling a unit.
6. **Token scheme across Pythons**: bytes equal is unchanged on any Python; a comment-only edit counts as unchanged only under the Python
   that made the token (`ast-py3.12-cg1-v1`), else it is stale (conservative). Native identities hash `.py` by bytes: a comment edit in a
   closure `.py` restamps that op.
7. **A source edited while `dev derive` runs**: caught by the digest snapshot of every indexed file and by the per-unit tokens, but a
   file outside the index (data) only by its token at publication time; an edit between an import and its use in a long-lived worker
   shows as a changed file only if it lies in the unit's set.
8. **First unit per worker**: records the imports that warmed the process (over-approximation); later units rely on the module-level
   closure for what an earlier unit imported (see 3).
9. **`registry_module.csv` / `registry.csv` / `gpu_to_kernels` are whole-file members of every unit's set**: adding a row restamps nothing
   native but re-records every unit (sound, coarse: ~1 min with 8 workers).
10. `plans/sm90` not regenerated (see Migration 3).

# Release actions

Kernel development, build and release are eight `miniworld-engine` commands. Every one is **code-graph native**: it runs the
repository graph first (`dev code-graph --diff`, the same engine), derives from it exactly what is owed, prints that scope
**before** doing anything, does only that, and does nothing (exit 0, no GPU, no cache touched) when the scope is empty. There is no
hook and no allow-list: the guard is the action design. A full rebuild, a whole-directory test run or a re-benchmark of untouched ops
is not something an action can be asked to do by accident.

Rules shared by every action:

* **Base: where the action left off.** Each run starts by snapshotting the checkout (a git tree object of tracked and untracked files,
  written with a temporary index: no commit, no ref, the checkout untouched) and, when it finishes, stores that snapshot and the items it did
  not finish (failed, not run) as the action's state. The next run diffs the tree against that snapshot (the code graph, symbol-level) and
  owes what the diff reaches plus the unfinished items. Before the first run, or when `git gc` pruned the snapshot, the base is the last
  uploaded release (`git describe --tags --abbrev=0 --match 'v*'`); `--base <rev>` overrides it. `bundle` and `verify` do not use any base
  for registry rows: they compare the bundle and each row record with the tree itself.
* **Cards**: the accessible card comes from the machine (`MINIWORLD_ACCESSIBLE_CARDS`, else the visible GPU, else sm90). Work the
  change owes on another card is printed as `UNVERIFIABLE-HERE (owed on <card>)`, never done and never claimed.
* **The internals refuse outside an action.** `dev bundle-kernels`, `dev bundle-verify`, `dev build-wheel`, `dev derive`, `bench_kernel`,
  `bench_module` run only under an action (`MINIWORLD_ENGINE_ACTION`, set by the action for itself and the processes it starts), and pytest
  refuses a directory (or no path: the whole suite) outside an action and CI; naming test files is allowed. A deliberate direct run says why:
  `MINIWORLD_ENGINE_INTERNAL='<why>'` (printed).
* **No scope arguments.** There is no `--only`, path, family or `--coverage` on any action. The only way to widen is
  `--all --reason "<text>"`, which prints the reason and stores it in the record. `--all` without a reason, or a reason without `--all`,
  is refused.
* **Already done is not redone.** A test file or bench module is owed only when the diff since the last run reaches it or that run left
  it unfinished; a registry row's state is its row record; an owed bundle row is also kept in a ledger with a digest of its inputs. The same
  edits are not run twice.
* **Records** (JSON: tree hash, base, scope, result, versions, time) are written under `<settings.engine_jit_root>/records/<action>/`
  (`MINIWORLD_ENGINE_RECORDS` overrides the root), outside the checkout. The *tree hash* is the content of everything that ships
  (`src/`, `pyproject.toml`; `_prebuilt/` is not in git). Downstream actions read records; a missing or stale one is an error that names
  the action to run.
* Run them through Slurm like everything else (docs/gpus/h100/h100.md); `status` needs a CPU allocation only.

| Action | Scope source (the code graph) | Refuses / does nothing when | Record it writes |
|---|---|---|---|
| `status` | all of the below, side by side | never refuses; changes nothing | none |
| `test` | `dev code-graph --diff <base> --tests` for the accessible card: the CPU files together in one pytest process, the GPU files as a work queue (one file per free GPU, one log line per file as it finishes) | no file named, or every named file already passed at identical inputs: `nothing to do`. GPU files with no GPU in the allocation are reported not run | `test/`: result, failed and not-run files; ledger: file -> input digest |
| `bundle` | **the bundle against the tree, not against the base.** Every artifact of the installed bundle carries its key material (the source files and their digests): an artifact whose sources now differ is *stale*. The row record (`tests/cpu/builder/data/registry_row_artifacts_<arch>.json`: per row of `registry_module.csv`, keyed `module/stream/dims/A=<eval>\|<train>`, what it declared, the artifacts it was served from) maps artifacts to rows. A row is owed when it is new or edited since its record, when one of its artifacts is stale or not in the bundle, or when the last `verify` found it missing a key. Only the owed rows run their coverage calls (an unchanged key is a prebuilt hit, so only owed keys compile), plus the `registry.csv` kernel rows the graph reached. Keep semantics: a name the previous release has is never dropped | empty scope: `nothing to do`. `--prune-reviewed <file>` is the only way to remove artifacts, and every name must already be in `tests/cpu/builder/data/prebuilt_names_allowed_drops.txt`. An artifact still stale after the run makes the result `fail`; a stale artifact no recorded row produces is listed as a gap (declare its row), never widened | `bundle/`: result, bundle digest (manifest hashes), artifact names, names dropped vs the previous release, items not ok, still stale, gaps |
| `verify` | **each row against its own clean record, not against the base**: a row runs when it is new or edited, when the last verify found it missing a key, when an artifact it is served from was rebuilt (`served`: the digest of those artifacts when it ran clean), or when a file its derivation units ran (`autotune/plans/<card>/registry_kernel.units.json`) changed since (`inputs`); plus the `registry.csv` kernel rows the graph reached. Run with an empty cache, `MINIWORLD_ENGINE_JIT_FALLBACK=0`, no nvcc (`dev bundle-verify` underneath, on the checkout's `src/`). A clean row's record is replaced, an unclean row's is dropped (so `bundle` owes it), a row the registry no longer has is dropped. Always also a whole-bundle **registry-row audit** (no GPU): every row has a clean record of what it declares now and every artifact of it is in each CUDA build | the audit reports a row without a clean record or with a missing artifact: `fail`, even with an empty GPU scope. Empty GPU scope: no GPU used | `verify/`: result, audit counts, bundle digest, rows found missing keys; the row record in the checkout |
| `derive` | the derivation units whose recorded files no longer match (`autotune/plan.py` `stale_units`); `dev derive` reuses every other unit | no stale unit: `nothing to do` | `derive/`: result, stale units left |
| `bench` | benchmark modules (`benchmarks/modules/<op>/`) the graph reaches for the card, minus those already measured at identical inputs; runs `bench_module` with the module compiled and CUDA-graphed, inference and training. There is no eager mode | none reached: `nothing to do`; a reached directory that is not a bench target is listed as a gap | `bench/`: result, failed ops |
| `wheel` | no graph: the records. `--split` and `--strict` are build options, not scope | no passing `bundle` and `verify` record for this exact tree hash, version and arch; the bundle changed since `verify`; the bundle lacks a name the previous release carried (and the allow-list does not explain it) | `wheel/`: files with sha256, bundle digest, names, the two records it rests on |
| `release` | no graph: the records | no passing `wheel` record for this tree and version, a wheel not built from the verified bundle (digest), a built file changed or gone since its record, a dropped name without an allow-list line, no committed `run_all` manifest for the version from a clean tree (`test_the_released_version_has_run_on_a_card`). It uploads **only** with `--confirm`; the maintainer uploads, an agent never does | `release/` on upload |

```
miniworld-engine status                       # what each action would do now; nothing is run
miniworld-engine test
miniworld-engine bundle
miniworld-engine verify
miniworld-engine derive
miniworld-engine bench
miniworld-engine test --all --reason "torch upgrade"    # a deliberate full run, recorded
miniworld-engine wheel --split
miniworld-engine release                      # every check, no upload
miniworld-engine release --confirm            # maintainer only
```

## Typical order

`status` -> edit -> `test` (only what the edit reaches) -> `derive` if it owes units -> `bundle` -> `verify` -> `bench` (only reached
modules) -> `wheel` -> `release`. Repeating an action on the same edits prints `nothing to do`.

## The low-level commands

`dev bundle-kernels`, `dev bundle-verify`, `dev build-wheel`, `dev derive`, `build`, `bench_kernel`, `bench_module` and the rest stay, as
the internals the actions call and for debugging. They take paths, filters and `--coverage`, so they are not how a release is made: use
the actions, whose scope the caller cannot widen. (`dev bundle-kernels` gained a `scoped:<file>` coverage driver and `dev bundle-verify` a
`--kernel` filter for the actions to use.)

## Known limits

* `verify` runs on the checkout's `src/` (the bundle in `_prebuilt/`), not on an unzipped wheel; the wheel-level check
  (`dev bundle-verify --wheels`) remains a manual step of the release.
* Several Python ABIs: run `wheel` once per interpreter; `release` collects every passing wheel record of the tree.
* A derivation unit that no coverage case corresponds to is reported as a gap, never widened to its module.

## Failures, retries, the path of a row

* **A failure is not rerun on the same inputs.** An item that failed (a test file, a bundle or verify row, a bench module) is kept in
  the action's state as *failing*: the next run reports it (`still failing, not rerun`) and the result stays `fail`, but it runs again
  only when a change reaches it. After an environment fix the graph cannot see (a tool missing from `PATH`, a driver), `--retry-failing`
  reruns exactly the recorded failures, nothing else. An item that no longer exists (a removed test file or registry row) leaves the list.
* **`bundle` refuses before building when `ninja` is not on `PATH`** (every owed key compiles through `torch.utils.cpp_extension`).
* **The path of a row.** The whole-bundle audit of `verify` and `status` lists the rows that ran clean but loaded no bundle artifact on
  this card: they ran on no engine kernel (cuBLAS, an external library, or the PyTorch statements). A registry row should run on CUDA
  here, so the list is a set of holes to look at, printed every time.
* **Precisions.** A registry dtype is a torch dtype or `master`: fp32 parameters fed bf16 activations (a trainer without autocast keeps
  fp32 master weights; other kernel builds, e.g. TriMul backward with `-DMASTER_FP32=1`). `autotune.module_registry.torch_dtypes`
  turns either into (parameter dtype, activation dtype) for the expansion, the coverage and the derivation.
* **`derive`** runs the units on FakeTensors. A hand-CUDA launch with no fake contract (`kernels/_compile.opaque(fake=...)`) fails its own
  unit with the extension and function named instead of launching (a launch on FakeTensors was a sticky illegal address that failed
  every later unit of the worker). After a successful derivation the sweep page is regenerated.

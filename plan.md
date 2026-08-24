# plan

Work list derived from `docs/library-standards.md`. Every item names the criterion it closes, the
gap as measured, the action, and **done-when** — a check that fails today and passes after.

Ordering is by *what a consumer feels first*, not by effort. P0 items are regressions and unproven
claims; P1–P4 are contract and correctness holes; P5+ is hygiene.

Status: `todo` / `doing` / `done` / `deferred (reason)`.

| id | criterion | title | status |
|---|---|---|---|
| P0a | D1 | plot-style entries orphaned by the label rename | **done** |
| P0b | E1 | prove `build all` end to end | todo |
| P1 | A3 | ship `py.typed` | **done** |
| P2 | B2 | per-kernel numerical tolerance | todo |
| P3 | B4 | ragged/fp32 shape modes become a gate | todo |
| P4 | D5 | the `configs` shadowing landmine | **done** |
| P5 | F3 | delete the orphan pilot builder | **done** |
| P6 | A4 | deprecation policy with a mechanism | todo |
| P7 | A5 | hardware support matrix, checked | todo |
| P8 | B5 | determinism statement + test | todo |
| P9 | C2 | quoted numbers traceable to a table | todo |
| P10 | D4 | end the `configs/grid` duplication | todo (premise corrected) |
| P11 | F4 | stale reference docs | **done** |
| P12 | F5 | `todo.md` is not repository furniture | **done** (2 comment refs pending) |
| P13 | F6 | CONTRIBUTING | **done** |
| P14 | A4 | the declared Python floor is untested | todo |

---

## P0a — plot-style entries orphaned by the label rename  (D1)

**Gap, measured.** Renaming the implementation labels (`triton_tri_attn` ->
`triton_triangle_attention` and four siblings) orphaned exactly 5 entries in
`src/miniworld_engine/viz/style.py`, whose keys are the label with separators stripped. One of the
five is worse than cosmetic: `triton_triangle_attention_miniworld` no longer matches its alias, so
`canonical()` falls through to the `"miniworld"` substring heuristic and it becomes *the same
identity* as the real `miniworld` series — one colour, one legend entry, in exactly the figures that
compare those two. Three further labels (`adaln_lnfold`, `augmented_attention_memory_efficient`,
`triton_triangle_attention_atomic`) were never catalogued at all, before the rename or after.

This is a regression I introduced. It is P0 for that reason.

**Action.** Re-key the 5, add the 3, and add `tests/test_plot_style_catalogue.py` asserting three
things over the union of (labels `bench.py` can produce, labels in the committed tables):
every label resolves through `_ALIASES`; no two labels collapse to one identity; no
`_KERNEL_VARIANTS` key is unreachable. The third is the one that would have caught this the moment
it happened.

**Done when.** `pytest tests/test_plot_style_catalogue.py` passes and fails if any of the 5 keys is
reverted.

**Done.** 5 re-keyed, 3 gaps filled, 49 cases green. Verified the collision is gone
(`triton_triangle_attention_miniworld` -> its own identity, not `miniworld`) and that reverting one
key fails 2 checks including `test_no_style_entry_is_unreachable`, which is the one that catches
this class at the moment it happens.

---

## P0b — prove `build all` end to end  (E1)

**Gap.** `build all` is structurally one command (validate -> config set -> 922 `(op, dtype,
bucket)` units -> run across GPUs -> merge into `data/`), and the CPU suite covers its
decomposition, claim/resume and merge policy. But **no run since the harness refactor has executed
it.** The refactor moved all 103 driver functions into `drivers/<family>.py`, and `build all
--per-op` calls exactly those. Import and `getattr` are verified (including under
`MINIWORLD_SHAPE_MODE=ragged`, `MINIWORLD_DRIVER_DTYPE=fp32` and the atom side, which are
import-time constant evaluations); *launching* is not. A driver whose shape block ended up bound to
the wrong host module imports fine and runs the wrong shapes.

The checker side is already covered: `tests/test_numerical.py` walks the registry's `check` column,
so the moved `checks/<family>.py` are exercised by the GPU suite.

**Action.** One GPU job, bench excluded:
1. `python -m miniworld_engine.autotune.run_all` — every declared driver launches; the report is
   `ok` / `failed` / `untested` against the registry as denominator.
2. A bounded real `build` — decompose, run, merge — over a handful of ops, with `data/` copied
   aside and restored, so no tracked cache changes. The full 922-unit sweep is hours and would
   rewrite the shipped A6000 cache; the point here is the chain, not the tuning.
3. `miniworld-engine dev audit --shards <that shard dir>` — coverage with real reachability
   evidence, which is the one thing a CPU test cannot have.

**Done when.** `run_all` reports 0 `failed`; the bounded build reports `N ok, 0 empty, 0 failed`
and its merge writes the entries; `dev audit` exits 0 against those shards. Numbers recorded here.

---

## P1 — ship `py.typed`  (A3)

**Gap.** The package is annotated, `ty` gates at zero over `src tests benchmarks tools`, and
consumers see `Any` for every symbol, because there is no PEP 561 marker.
`[tool.setuptools.package-data]` is thorough about `.json` / `.cu` / `.csv` and does not mention it.

**Action.** Add `src/miniworld_engine/py.typed`; add it to `package-data`; add a test asserting
both (the file exists *and* the glob covers it — shipping the marker and not the file, or the file
and not the glob, are the same failure).

**Done when.** a built wheel contains `miniworld_engine/py.typed`.

**Done, verified in a real wheel** (`pip wheel --no-deps`, 547 entries): `py.typed` present, plus
the 186 tuned caches, the packaged config set, `registry.csv`, the CUDA sources and the build
matrix. `tests/test_typed_marker.py` asserts both halves — the file exists AND `package-data` lists
it, because either alone ships nothing.

**A gap this turned up:** the test first used `tomllib`, and `ty` rejected it — `tomllib` is 3.11+
while this package declares `requires-python = ">=3.10"`. The test now reads the file as text. But
nothing exercises the declared floor: CI runs 3.12 only, so `>=3.10` is an unverified claim.
-> new item P14.

---

## P2 — per-kernel numerical tolerance  (B2)

**Gap.** `autotune/run_all.check_one` applies one band to all 99 checkers: `max|a-e| / max|e| <
5e-2`. That is the weakest kernel's band applied to a transpose, a mask fold and a gate multiply,
several of which should be bit-exact. A reduction-order regression costing 1e-3 is invisible.

**Action.** Add a declared tolerance to each registry row (a `rtol` column; blank = the default
band, which stays documented as bf16's ~3-decimal band). `check_one` compares against that row's
value. Then tighten: for each kernel, measure its actual observed `rel` on a known-good build and
set the band a decade above it, not at 5e-2. Kernels that are exact get `0`.

Two sub-steps, because the second needs GPU evidence:
- P2a: mechanism (column, reader, test that every row's band is respected and that a blank means
  the documented default).
- P2b: calibration (run all checkers, record observed `rel`, set per-row bands). Needs a GPU run.

**Done when.** P2a: a test that `check_one` reads the row's band, with a synthetic row proving a
tighter band actually fails. P2b: no row left at the default band unless its measured `rel`
justifies it, recorded in the commit.

---

## P3 — ragged/fp32 shape modes become a gate  (B4)

**Gap.** The mechanism that makes boundary masks execute (`MINIWORLD_SHAPE_MODE=ragged`) exists and
is the reason a whole class of tail-tile bug is findable. Nothing runs it automatically, so it
protects nothing — it is a tool someone has to remember.

**Action.** Add a GPU-suite stage (and a `pixi` task) that runs the numerical suite under
`MINIWORLD_SHAPE_MODE=ragged` and under `MINIWORLD_DRIVER_DTYPE=fp32`, and record the runtime. If
running all three modes is too slow for every GPU run, make ragged the default for the numerical
suite and aligned the opt-in — aligned shapes are the weaker test.

**Done when.** `pixi run test-gpu` covers at least the ragged mode, and a deliberately broken tail
mask fails it. Runtime for each mode recorded here.

---

## P4 — the `configs` shadowing landmine  (D5)

**Gap.** `autotune/configs.py` (the config-CSV reader) and `autotune/configs/` (the shipped default
config set) coexist. The module wins **only** because the directory has no `__init__.py` — verified
empirically. Adding one, which is the reflex when making a shipped asset importable, silently
replaces the reader with a namespace package.

**Action.** A test asserting `autotune/configs/` has no `__init__.py`, with the reason and the
experiment in the docstring. (Renaming either side is the alternative; it churns `package-data`,
the README and every `MINIWORLD_CONFIG_DIR` reference for a hazard a two-line test removes.)

**Done when.** The test exists and fails if an `__init__.py` appears.

**Done** — `tests/test_default_config_set.py::test_the_packaged_config_dir_is_not_a_package`, with
the two-line experiment that proves the precedence recorded in its docstring.

---

## P5 — delete the orphan pilot builder  (F3)

**Gap.** `src/miniworld_engine/autotune/build.py` stores under op names `transition_split_fwd` and
`trimul_bidir_front`. Neither is in `registry.csv`; `docs/kernels/rename-map.tsv` records both as
renamed (`transition_expand_swiglu_triton`, `trimul_gemm_gate_mmajor_triton`). The cache reader
looks up the name the kernel registers, so every entry this script has written since that rename is
unreadable. Nothing imports it. Its `main()` is a second, undocumented build entry point one letter
from the real one: `python -m miniworld_engine.autotune.build --op ...` against
`miniworld-engine build`. Both ops are fully covered now — driver, checker and shipped cache under
their current names.

**Action.** `git rm` it; update the sentence in `capture.py`'s docstring that calls these the
"hand-built pilot caches" (the cross-check it describes is how `builder.py` was validated, so the
history stays, in past tense).

**Done when.** The file is gone, the suite is green, and `dev audit` reports no new holes.

**Done.** `git rm`'d. Two places *documented* it as a way to build a cache —
`docs/operations/dispatch-cache.md` ("Explicit builder (per-kernel, for the pilot kernels)") and
`autotune/data/.gitkeep` ("populate on-target with ... then copy the generated cache here and
commit"). So a reader following the docs used a dead path that wrote entries nothing could read, and
then hand-copied them. Both now point at `miniworld-engine build all` / `build <op> --per-op`, and
`.gitkeep` says the build merges for you rather than telling you to copy files. `dev audit` still
to run — folded into P0b.

---

## P6 — deprecation policy with a mechanism  (A4)

**Gap.** CHANGELOG claims SemVer; `version = "0.1.0"`. `_CONTRACT` freezes the surface, so a
removal fails the test — which is right — but there is no *procedure* for a removal, so in practice
nothing is ever removed.

**Action.** A short policy section in CHANGELOG or CONTRIBUTING: a name is deprecated in release N
(kept, emitting `DeprecationWarning`, listed in CHANGELOG under Deprecated), and removed no earlier
than N+2. Add a `_DEPRECATED` set next to `_CONTRACT` and a test that every name in it warns on
access and is still importable.

**Done when.** The policy is written and the test passes with at least one entry — or, if nothing is
currently deprecated, with a synthetic case proving the mechanism.

---

## P7 — hardware support matrix, checked  (A5)

**Gap.** Arch requirements are asserts inside individual checkers ("SM90 (H100) only"). There is no
statement of which archs the library supports, which backends need which, or what a consumer on an
unsupported card gets. `registry.csv` has a `backend` column and the cute/cuda paths have implicit
arch floors.

**Action.** Derive the matrix from the registry plus the declared arch floors, render it into the
README, and add a test that the rendered table matches what the registry says — so it cannot drift.
State the fallback rule: every op has a triton path, and the dispatch falls back to it.

**Done when.** README carries the matrix; the test fails if a kernel's arch requirement changes
without the table changing.

---

## P8 — determinism statement + test  (B5)

**Gap.** Nothing says whether two runs agree bitwise. With an autotuner selecting per GPU and per
cache state, the honest answer is nuanced, which is exactly why it must be written.

**Action.** State it: within one process and one cache state, a given op is deterministic; across
cache states it may select a different config and results may differ in the last bits; reductions
are not order-stable across configs. Add a GPU test that two calls under one cache state are
bitwise equal.

**Done when.** The statement is in the README and the test passes.

---

## P9 — quoted numbers traceable to a table  (C2)

**Gap.** `benchmarks/RESULTS.md`, `docs/` and the CHANGELOG quote latencies and speedups. Nothing
ties a quoted number to the CSV row that produced it, so a doc can outlive its measurement.

**Action.** A test that every `N.NN ms`-shaped figure in `benchmarks/RESULTS.md` appears as a value
in some committed table (within float tolerance), or is explicitly marked as coming from a run that
was not committed. Extend to `docs/` only if the first is cheap.

**Done when.** The test passes, and every claim that cannot be traced is either sourced or marked.

---

## P10 — end the `configs/grid` duplication  (D4)

**Gap.** `configs/grid` exists at the repo root and inside the package. Deliberate, documented, and
`tests/test_default_config_set.py` asserts byte-identity — but `autotune/configs/README.md` says the
root copy goes away "once no running job depends on it".

**Measured — the premise was wrong.** `autotune/configs/README.md` said the root copy "goes away
once no running job depends on it", implying a leftover waiting on a sweep. It is not a leftover.
No tracked file points `MINIWORLD_CONFIG_DIR` at it, but `cli.resolve_config_dir` maps a short
config-set name to `repo / "configs" / <name>` — the **repo root** — so `build all`, both benches
and every accuracy run in a source checkout resolve `grid` there. The packaged copy is reached by a
different path entirely (`configs.default_config_dir()`, for wheel installs, where no repo root
exists). Two copies, two consumers, and deleting the root one breaks the CLI.

So this is not "delete the leftover"; it is "give the resolver one place to look".

**Action.** Add the packaged set as a fallback candidate in `resolve_config_dir`, so a short name
resolves to `repo/configs/<name>` when present and to the packaged `autotune/configs/<name>`
otherwise. Then the root `configs/grid` can go while the nine A-B sets stay where they are, and
`tests/test_default_config_set.py`'s byte-identity assertion is replaced by the stronger one: there
is only one copy. Correct the README sentence either way.

**Done when.** `configs/grid` exists in exactly one place, `miniworld-engine build all --resume`
still resolves its config set in a source checkout, and a test covers the fallback (short name with
no root copy -> packaged set).

**Blocked on:** nothing, but it touches `src/` and `tests/`, so not while a GPU job is reading this
worktree.

---

## P11 — stale reference docs  (F4)

**Gap, measured.** 86 mentions of pre-rename op names across 32 doc files. Of those, `naming-audit.md`
(27) and `naming.md` (5) have the old names as their *subject* and must not change;
`docs/kernel-optimization/**/v*.md` are dated per-version logs and are records. The genuinely stale
one is `docs/kernels/l2-swizzle.md` — an undated *reference* table ("the 18 kernels that do not use
GROUP_M") naming 21 ops by pre-rename names, every one of which `rename-map.tsv` maps to a name that
exists in the registry. `tm1.md`, `tiling-audit.md` and `triangle-multiplication-module.md` were
checked and are false positives: they name the `fused_ln_mask` *family*, which still exists.

Also: `autotune/configs/README.md` lists `devices` among the A-B config sets. `configs/devices/`
is not a config set — it is the tracked per-GPU kernel manifest read by `autotune/devices.py`.

**Action.** Apply `rename-map.tsv` to `l2-swizzle.md`'s op names, verifying each target against
`registry.csv`, and check each cited file path against the registry's own `file` column by hand
(a moved kernel is a different claim from a renamed one). Fix the `devices` sentence. Add a dated
header to the docs that are records, so the distinction is visible.

**Done when.** No doc that presents itself as current reference names an op absent from the
registry.

**Done.** 21 op names in `l2-swizzle.md` mapped through `rename-map.tsv`; every cited file path was
checked against the registry's `file` column and all 21 agree, so no kernel had moved — only been
renamed. `autotune/configs/README.md` no longer lists `devices` as a config set and says what it is
instead. `naming-audit.md`, `naming.md` and `tiling-audit.md` now open by saying they are records
and where the current names live; `l2-swizzle.md` says it is current reference and what makes it
so. Re-measured: zero current-reference docs name a pre-rename op.

---

## P12 — `todo.md` is not repository furniture  (F5)

**Gap.** 14 KB of dated engineering findings at the repo root of a library other repos consume.
Some items are already done.

**Action.** No triage was needed: the file marks its own items, `- [x]` / `- [ ]`, 13 done and 14
open, and **all 14 open ones are cute / CUTLASS work on sm90 or sm100** — several blocked on
hardware this cluster does not have ("sm100 (B200) kernels — deferred (no B200 to verify)"), and the
whole area is explicitly set aside for now. So the file is a record with a live tail, not a mixed
bag needing my judgement.

Moved whole, nothing removed, to
`docs/kernel-optimization/cute-autotune-and-config-pinning.md`, with a header saying where it came
from, what the 14 open items are, and that the section dates are the findings' dates rather than the
move's. This item is the tracker for that tail; it does not restate it.

**Done when.** The repo root holds no working notebook and nothing was thrown away. **Done** — with
one loose end: `kernels/layernorm_linear/cute/__init__.py:44` and `.../cute/_tuned.py:26` cite
`todo.md` by name. Both are one-line comment fixes, held back only because a GPU job is reading this
worktree; they land with the other `src/` work rather than as a dangling reference.

**The tail, for reference (not duplicated — see the record):** 14 open items across cute autotune
wiring (`resolve_config` for the remaining cute GEMMs, a capture driver for `sweep_and_cache`),
correctness-pinned tile/cluster constants on sm90 and sm100, one known-broken path
(`layernorm_linear_cute_fused` M2), and the `mPostAct` / `mAuxOut` epilogue-naming drift.

---

## P13 — CONTRIBUTING  (F6)

**Gap.** How to run the gates, what a change must include, and what the review bar is exist —
scattered across the README, `pyproject.toml` comments and commit messages.

**Action.** One `CONTRIBUTING.md`: the three gates and the one command that runs them; a change
includes a test that fails without it; a CHANGELOG entry for anything a consumer can see; the
vendored-code boundary; the deprecation procedure from P6; the standard in
`docs/library-standards.md` as the bar.

**Done when.** A newcomer can get to a green `pixi run ci` from that file alone.

**Done.** `CONTRIBUTING.md`: the gates and the one command that runs them, what a change must
include (a test that reproduces the defect, a CHANGELOG entry for anything observable, a reason on
every suppression), a table of where each kind of thing goes and which test enforces its shape, the
naming rule, the vendored-code boundary, the benchmark-provenance rule, and the cache/pre-commit
hook. Every `pixi run <task>` it names is a declared task. The deprecation section states the
mechanism that exists (a failing `_CONTRACT`) and calls the missing warning period a gap rather
than describing P6 as if it were done.

---

## P14 — the declared Python floor is untested  (A4)

**Gap, found while doing P1.** `requires-python = ">=3.10"` and ruff's `target-version = "py310"`,
but the pixi env is 3.12 and CI installs 3.12 only. Nothing has ever run this package on 3.10, so
the floor is a claim, not a fact — and it is the kind that breaks silently: my own new test reached
for `tomllib` (3.11+) and only `ty`, resolving against the declared floor, objected.

**Action.** Either add a 3.10 job to the CI matrix (cheap: the CPU suite is ~60 s and needs no
GPU), or raise the floor to what is actually tested and say so. A floor nobody exercises should not
be advertised.

**Done when.** The `requires-python` value is exercised by a CI job, or it equals the version CI
runs.

# plan

Work list derived from `docs/library-standards.md`. Every item names the criterion it closes, the
gap as measured, the action, and **done-when** — a check that fails today and passes after.

Ordering is by *what a consumer feels first*, not by effort. P0 items are regressions and unproven
claims; P1–P4 are contract and correctness holes; P5+ is hygiene.

Status: `todo` / `doing` / `done` / `deferred (reason)`.

| id | criterion | title | status |
|---|---|---|---|
| P0a | D1 | plot-style entries orphaned by the label rename | **done** |
| P0b | E1 | prove `build all` end to end | **done** — found 2 regressions + P15-P18 |
| P1 | A3 | ship `py.typed` | **done** |
| P2 | B2 | per-kernel numerical tolerance | **P2a done**, P2b needs a GPU |
| P3 | B4 | ragged/fp32 shape modes become a gate | **measured** — 3 candidate bugs found |
| P4 | D5 | the `configs` shadowing landmine | **done** |
| P5 | F3 | delete the orphan pilot builder | **done** |
| P6 | A4 | deprecation policy with a mechanism | **done** |
| P7 | A5 | hardware support matrix, checked | **done** |
| P8 | B5 | determinism statement + test | **test written**, statement pending its first run |
| P9 | C2 | quoted numbers traceable to a table | **done** (scope corrected) |
| P10 | D4 | end the `configs/grid` duplication | **done** |
| P11 | F4 | stale reference docs | **done** |
| P12 | F5 | `todo.md` is not repository furniture | **done** |
| P13 | F6 | CONTRIBUTING | **done** |
| P14 | A4 | the declared Python floor is untested | **done** |
| P15 | E2 | a stale JIT build lock hangs the GPU suite forever | **done** (both halves) |
| P16 | B1 | `["ALL"]` lint exemption hid a guaranteed NameError | **done** |
| P17 | E2 | `run_all` reported "wrong card" as "wrong answer" | **done** |
| P18 | B1 | opcheck asserted a contract the design declines | **done** |

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

**Done. The chain works, and the run found more than it was looking for.**

`run_all`, all 103 declared drivers, 77 s: `driven 103, ok 92, failed 11, no driver 0`. The 11 were
3 x the `.cpp` path bug, 2 x the undefined `L` (one bug, two kernels -- `pre_contig`'s DRIVER runs
the whole Function including backward), and 6 arch-gated cute kernels that are not failures at all,
which became P17.

The build chain, four ops chosen one per shape-block group so every cross-family import the harness
refactor introduced was exercised:

    adaln_fwd_triton                        6 ok, 0 empty, 0 failed   103 s
    bias_only_attention_fwd_triton          4 ok, 0 empty, 0 failed    74 s
    layernorm_fwd_rowscale_triton           4 ok, 0 empty, 0 failed    73 s
    gated_projection_gate_inplace_flat...   4 ok, 0 empty, 0 failed    93 s
    18 shards written; data/ changed for all four ops   <- the merge wrote
    dev audit: reach 4 OK  <- exactly the four built, read from the shard evidence
    data/ restored; `git status -- data` empty

So: decompose -> run -> **merge** -> audit, with the shipped cache untouched.

**Sizing, not the chain, was what failed first.** `build <op> --per-op` with the default `grid` set
blew a 600 s cap twice. `grid` is a grid SPEC (5 axes, cartesian) and per-op tuning benches every
config in it -- cli.py's own comment says a single 15,552-config op costs 244 GPU-h. `blk16` is
materialised with exactly one config per op, so each unit measures once and the plumbing is what
gets exercised. Tuning was never what needed proving.

**Two unlooked-for confirmations:**

* the STALE guard works. The audit read entries the merge had just written from a `blk16` build,
  against the default `grid` config space, and the reader correctly reported
  "STALE (kernel config grid changed) ... falling back to the full grid" for each. That is exactly
  the failure mode `default_config_dir`'s docstring warns about, caught by the guard rather than by
  a silent wrong answer.
* `dev audit`'s own `import` check is **permanently red on any non-Hopper card**:
  `import: 0 OK, 2 not OK`, `transition.cuda -> RuntimeError: Error building extension
  'transition_b2b_cuda'`. That is the eager import-time build P15 deferred, and it is now not a
  theoretical concern -- it makes `dev audit` exit 1 on an A6000 for a reason that has nothing to do
  with the cache. -> raises the priority of P15's lazy half.

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

**P2a done.** `registry.csv` gains an `rtol` column (blank = the documented default, never
"unchecked"); `check_one(check, rtol=None)` compares against it and names the band in the detail
string, so a failure says what it was measured against; `declared_rtol(row)` raises on a malformed
or negative value rather than falling back to the loose default -- a typo that widens a kernel's
tolerance is the thing this column exists to prevent. `tests/test_declared_tolerance.py` covers it
with synthetic checkers, so the mechanism is verified without a device.

**And it found a real bug, which is the argument for the whole item.** The old code read:

    worst = max(worst, rel)
    ok = worst < 5e-2 and worst == worst      # `worst == worst` "rejects NaN"

`max()` returns its first argument when the comparison is false, and `nan > 0.0` is false, so
`max(0.0, nan)` is `0.0`. The NaN was discarded before the guard ever saw it: **a kernel writing
NaN scored 0.0 and passed every band**, including a declared 0. Non-finiteness is now checked per
pair, at the point it is computed, and reported as `NON-FINITE`. Note the consequence for the
validation run in flight: it is testing `7af55ce`, before this fix, so it cannot fail a NaN-writing
kernel. `test_numerical` must be re-run after this merges.

**P2b** stays open: calibrating each row's band needs the measured `rel` from a good GPU run.

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

**First run ever. It found three failures, which is the point of the mode.** Under
`MINIWORLD_SHAPE_MODE=ragged` the first three kernels in collection order fail:

    adaln_bwd_dlnw_triton
    adaln_bwd_dw_triton
    adaln_bwd_dx_dbias_triton

All three are adaln backward weight-gradient kernels, and all three pass at aligned extents. That
is the boundary-mask signature this mode exists to expose: every default driver extent is a multiple
of 128, so no tail tile had ever been executed. Their drivers use `ragged()` rather than
`aligned_only(label, n, why)`, so the author's own declaration says they are meant to handle a
partial tile. Candidate bugs, not confirmed: fixing them needs a device to iterate on, and I have
not read the kernels yet. **Do not treat these as verified defects until someone has.**

**On making it a gate — the cost is NOT what I first assumed.** The stage hit its 900 s cap having
done 12 of 100. My first guess was a cache miss; the measurement says otherwise:

* `token_key` FLOORS to the tuned ladder, so a ragged extent lands in an existing bucket, not a
  new one: 512 -> 512 but 509 -> 384, 128 and 125 -> 128, 384 -> 384 but 381 -> 256.
* the stage's output contains **zero** "no tuned autotune cache" / STALE warnings.

So the cache hits. The most likely cost is a cold **triton** recompile: triton specialises on
divisibility-by-16 of its integer arguments, and 509 is not 512, so every kernel JITs a second
specialisation the aligned runs never produced. Evidence is consistent (aligned stage A did 100
cases in 776 s with a warm triton cache; ragged did 12 in 900 s) but this is not proven -- I have
not instrumented the compile. If it is right, the cost is paid ONCE per triton cache and the mode is
gateable; if it is not, this needs a different answer.

**Next step, in order:** (1) confirm the cold-compile theory by running the ragged stage twice in
one job and comparing wall times; (2) read the three adaln kernels and decide whether they are tail
bugs or drivers that should have said `aligned_only`; (3) only then make it a gate.

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

**Done, and with a real entry rather than a synthetic one.** `kernels.cuda_transition` was the
obvious first case: a public, frozen name that has never had an implementation (it deferred to a
`transition/cuda` symbol git has no record of) and raises `NotImplementedError` when called. It
could not simply be deleted — the frozen surface is what stopped that — so it is now the thing the
mechanism was missing for.

`kernels._DEPRECATED` maps name -> why-and-what-instead; `_warn_deprecated()` emits with
`stacklevel=3` so the warning points at the caller's line. Enforced by four tests: every
deprecated name is still in `__all__` (deprecated is not removed), each message names a
replacement, using the name warns, and the name is still reachable. A fifth covers the lazy path
with a synthetic entry, so the file keeps meaning something when `_DEPRECATED` is empty again.

**One design point worth recording**, because the first version of the test was wrong: I asserted
that *access* warns, and `cuda_transition` failed it — it is a plain module-level function, so
`__getattr__` never runs. Bending the mechanism to warn on access would have been worse than the
test: `hasattr`, `dir()` and a re-export all touch an attribute without using it. The rule is "a
deprecated name warns when it is USED", and use has two shapes here — resolution for the ~16 lazy
names, the call for the 3 module-level functions. The test accepts either and requires at least
one, which is what makes a name that warns on neither fail.

Also: a deprecated lazy name is deliberately not cached into `globals()`. The cache is what makes
`__getattr__` run once per process, and a warning only the first caller sees is not a warning.

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

**Done.** `registry.csv` gains an `arch` column -- a declared MINIMUM per kernel, not a derived
one, because the derivation (an `sm100` in a filename) is a heuristic that breaks the moment
someone names a file differently. Populated from what the tree already states: 94 at sm80 (88
unmarked Triton kernels, which have committed result tables on A100/A5000/A6000/H100/B200, plus the
6 hand-CUDA ones whose `kernels/<family>/cuda/setup.py` declares `-gencode` for
compute_80/86/89/90 explicitly), 2 cute at sm90, 4 cute + 3 Triton at sm100.

README renders it inside `BEGIN/END GENERATED` markers and `tests/test_hardware_support.py`
regenerates and compares, so the table cannot drift from the column. Four more checks keep the
column honest rather than decorative: every kernel declares one of the three levels; a name
containing `sm100`/`sm90` may not claim a lower arch (the name is evidence, not the source of
truth); no CuTeDSL kernel may claim the Triton floor; and -- the one that backs the README's actual
promise -- **every family with an above-floor kernel also has one at the floor**, so an
unsupported card loses performance and not function. Verified non-vacuous: 9 kernels across 5
families are above the floor, and all 5 families have an sm80 fallback.

One thing the table deliberately excludes, with the reason in the README: `transition_b2b_cuda` is
compiled for `sm_90a` and fails to build on sm_86, but it is not a registry kernel -- the
`Transition` module builds it on demand -- so it is named in prose instead of implied to be one of
the 103.

---

## P8 — determinism statement + test  (B5)

**Gap.** Nothing says whether two runs agree bitwise. With an autotuner selecting per GPU and per
cache state, the honest answer is nuanced, which is exactly why it must be written.

**Action.** State it: within one process and one cache state, a given op is deterministic; across
cache states it may select a different config and results may differ in the last bits; reductions
are not order-stable across configs. Add a GPU test that two calls under one cache state are
bitwise equal.

**Done when.** The statement is in the README and the test passes.

**Test written, statement deliberately withheld.** `tests/test_determinism_gpu.py` asserts the half
that IS a promise: two calls in one process, one cache state, identical inputs -> **bitwise** equal
(`torch.equal`, not `allclose` — a tolerance here would hide the exact thing being asked). Ten
kernels, one per family with a checker, chosen for launch-path variety (an atomic accumulation, a
split reduction, a persistent grid, a fused epilogue) rather than for coverage; declared in `SAMPLE`
with a guard test that fails if a name in it stops having a checker, so a rename cannot quietly
shrink the file to nothing.

The README statement is NOT written yet, on purpose. Writing "this is deterministic" before the
test has ever run is precisely the unverified claim `docs/library-standards.md` is about — and B5's
answer has two halves, only one of which is a promise:

* within one process and one cache state: bitwise identical (what the test asserts);
* across cache states — a rebuild, another GPU, another config set — a different config may win, and
  a different tile shape is a different reduction order, so the last bits may move. Not a bug; that
  is what tuning is, and it is the half that produces "your library is non-deterministic" reports.

The statement lands when the test has passed once. The test rides the next GPU job.

---

## P9 — quoted numbers traceable to a table  (C2)

**Gap.** `benchmarks/RESULTS.md`, `docs/` and the CHANGELOG quote latencies and speedups. Nothing
ties a quoted number to the CSV row that produced it, so a doc can outlive its measurement.

**Action.** A test that every `N.NN ms`-shaped figure in `benchmarks/RESULTS.md` appears as a value
in some committed table (within float tolerance), or is explicitly marked as coming from a run that
was not committed. Extend to `docs/` only if the first is cheap.

**Done when.** The test passes, and every claim that cannot be traced is either sourced or marked.

**Scope corrected first.** The item assumed `benchmarks/RESULTS.md` quotes numbers. It quotes
**zero** — measured. The real distribution: 1121 figures across `docs/`, but the overwhelming
majority are in `docs/kernel-optimization/**/v*.md`, which are dated per-version logs ("on this
date, v3 measured 1.75 ms"). A record does not go stale; a reference does. The live surface is 24
figures: README 1, CHANGELOG 6, `docs/benchmarking-cautions.md` 17.

**And the worst offender was the doc about not mis-measuring.** `benchmarking-cautions.md` carried
17 latencies and named a device **once**, in a file whose stated purpose is "read this before
trusting any trimul number".

**Done, enforced at file level rather than per claim, and that is a deliberate weakening.**
Per-claim attribution is not retroactively recoverable: back-filling would mean guessing which card
a 2026-07 trimul run used, which is worse than saying it is unknown. So the rule is that a live doc
making performance claims must state the hardware behind them *somewhere* — satisfied by a
provenance paragraph, including one that admits which numbers are unattributed and why.

`benchmarking-cautions.md` now opens with exactly that: the `compile_wrap` table is named as
**A6000 (sm86), L=384, d_pair=128, bf16, reproduced 3x**; the rest is marked as evidence for the
*effect* rather than figures to compare against, kept because the lesson survives the missing
metadata and flagged rather than back-filled. `tests/test_performance_claims.py` holds the seven
declared live docs to it, verified to bite (stripping the device names fails it with the 17 figures
named), plus a vacuity guard and a check that the excluded record tree still exists — if it
vanishes, the record/reference reasoning no longer applies and someone should notice.

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

**Done.** `resolve_config_dir` now tries, in order: an explicit path, `repo/configs/<name>`, then
the packaged `autotune/configs/<name>`. The root `configs/grid` (91 CSVs, verified byte-identical
to the packaged copy first) is deleted, and both readers -- the CLI's short name and
`configs.default_config_dir()` -- resolve to the one remaining copy. `default_config_dir` loses its
repo-root branch and the "neither the packaged nor the repo-root" warning loses half its text.

The byte-identity test is replaced by the stronger `test_grid_exists_in_exactly_one_place`, plus
two that pin the resolution order: a short name with no repo-root copy finds the packaged set, and
a set present in both resolves to the repo's (the A-B sets live only there, and that is where an
experiment edits one).

**What actually caused the duplication**, since the README blamed a leftover: the generator writes
the root copy (`gen_shards.py --out configs/grid`) and the packaged one was a manual copy of it.
One writer, two destinations. `gen_shards.py`'s usage line and the two `tools/kernel-audit`
launchers that exported `MINIWORLD_CONFIG_DIR=$R/configs/grid` now point at the package.

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

**Done when.** The repo root holds no working notebook and nothing was thrown away. **Done** — the
two `src/` comments that cited `todo.md` by name (`kernels/layernorm_linear/cute/__init__.py:44`,
`.../cute/_tuned.py:26`) now point at the moved record; `git grep todo.md -- src` is empty.

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

**Measured, and a correction to my own first reading.** A static probe over all 312 `.py` files
under `src/tests/benchmarks/tools` (`ast.parse(feature_version=(3,10))` plus a newer-stdlib name
check) found **0 syntax failures** and exactly one 3.11+ stdlib use:
`modules/dispatch.py` importing `enum.StrEnum`. I first read that as "the `>=3.10` claim is false".
It is not — the import sits inside `if sys.version_info >= (3, 11):` with an `else` that defines
`class _StrEnum(str, Enum)`, and a comment explaining why the guard is on `version_info` rather
than `try/except` (a checker can follow the first). So the code does support 3.10.

The real gap was narrower, and the code says it itself: that `else` branch is marked
`# pragma: no cover -- exercised only on 3.10`, and nothing has ever run 3.10. The static half of
the floor was already covered (ruff's `target-version` and ty's `python-version` both derive from
`requires-python`, which is how `ty` caught my `tomllib` import in P1); the runtime half was not.

**Done.** A second CI job, `floor`: installs on 3.10, asserts the package imports, asserts that the
**3.10 branch is the one that ran** (no `enum.StrEnum`, `_StrEnum` in the MRO, and
`KernelBackend.TRITON == "triton"` so the fallback behaves like a StrEnum and not just inherits
from one), then runs the CPU suite. Lint and types are deliberately not repeated — a job that
silently ran 3.12 would pass every other step and prove nothing, which is what the MRO assertion is
for.

**Also a note on how this was found**, because it nearly went the other way: the first probe run
scanned 98 files and reported zero findings. The shell's working directory had drifted to a
different repository (`practice/team-gm`), so the measurement was of the wrong tree. Caught by the
file count not matching `git ls-files`. Every subsequent command pins the directory.

---

## P15 — a stale JIT build lock hangs the GPU suite forever  (E2)

**Found by losing three runs to it**, ~5 GPU-hours: a 45-minute job I killed, a 4-hour job that hit
its time limit, and a third that stalled at test 42 of 100. All three looked identical to "slow".

`kernels/layernorm/cuda/__init__.py` calls `torch.utils.cpp_extension.load(...)` **at module import
time**, building `layer_norm_cuda_kernel.cu` for `gencodes("80","90","100", ptx=("100",))`. `load()`
serialises concurrent builds with a `FileBaton` on `~/.cache/torch_extensions/<abi>/<ext>/lock`, and
`FileBaton.wait()` polls for that file to disappear **with no timeout and no message**.

A run at 09:44 rewrote `build.ninja`, created the lock, and died. Every run afterwards blocked on it
forever. Proof: the lock was 13 hours old with no process holding it, and deleting it moved the
stalled job from test 41 to 48 within 20 seconds.

Three separate faults, and the third is the one that matters:

1. **The lock is never reclaimed.** A killed build leaves a file that stops every future run on this
   machine, for every extension it had started.
2. **The wait is unbounded and silent.** No timeout, no "waiting for another build", nothing to
   distinguish it from a long compile — which is exactly why it survived three investigations.
3. **The build happens at import.** Importing `kernels.layernorm` compiles CUDA for four
   architectures, so any consumer that touches that family pays it, and `test_numerical` pays it in
   the middle of an unrelated kernel's test.

**Action.**
- Wrap the extension load so a wait longer than a threshold (say 120 s) raises, naming the lock file
  and saying to delete it. A message beats a hang, and the fix is one `rm`.
- Reclaim a lock whose mtime is older than the build could plausibly be, the way
  `build --reclaim` already does for the builder's own O_EXCL claims — that precedent is in this
  repo and this is the same problem.
- Make the load lazy so importing the family does not compile anything, and only the tests and
  dispatch paths that need the CUDA backend pay for it. This is also A2 (importing the package does
  nothing) — the criterion is met for the package's own import only because nothing imports
  `layernorm.cuda` eagerly today.

**Done when.** A stale lock produces an error naming it within ~2 minutes instead of a hang, and
`python -c "import miniworld_engine.kernels.layernorm"` runs no compiler.

**Verified on a device, and it took looking to see that it worked.** `dev audit`'s import check
still read `import: 0 OK, 2 not OK` after the lazy conversion — the same count as before. Same
number, different cause, and concluding "the fix failed" without reading the message would have
been wrong:

    before   transition.cuda          RuntimeError: Error building extension 'transition_b2b_cuda'
    after    .../cuda/setup.py x2     SystemExit: usage: cli.py [global_opts] cmd1 ...

The build-at-import defect is gone. What surfaced underneath is a second, unrelated one:
`kernels/<family>/cuda/setup.py` are standalone setuptools scripts that live INSIDE the importable
package and call `setup()` at module scope, so anything walking the tree imports them and runs
setuptools. Guarded with `if __name__ == "__main__":` — running `python setup.py build_ext` is
unchanged — and `tests/test_no_build_at_import.py` grew a check for it, verified to bite.

**Done — the hang half.** `kernels/_nvcc.load_extension()` wraps torch's `load`: a lock older than
`STALE_LOCK_SECONDS` (30 min — longer than any real build here) is reclaimed with a note, and a
wait on a FRESH lock is bounded by `LOCK_WAIT_SECONDS` (10 min) and then raises **naming the file
and giving the `rm`**. Correctness still comes from torch's own baton; this only decides how long
to tolerate it. All four call sites now go through it, and
`test_every_jit_load_goes_through_the_guard` fails if a new one imports torch's `load` directly —
a guard is worth nothing if a call site can bypass it.

`tests/test_jit_build_lock.py` (9 cases) covers the boundary in both directions. The one that
matters as much as the reclaim is `test_a_fresh_lock_is_left_alone`: reclaiming a lock a live build
owns would corrupt a concurrent compile, which is a worse failure than the one being fixed.

**The lazy-load half, now done too** — P0b's audit turned it from cleanup into a defect
(`dev audit` exits 1 on any non-Hopper card because `transition.cuda` builds an sm_90a extension at
import). Both packages are PEP 562 lazy: the build moved into a function, the names come through a
module-level `__getattr__`, and `ensure_cuda_home()` moved with the build because it mutates
`os.environ` and an import should not. `tests/test_no_build_at_import.py` reads the SOURCE for a
module-scope `load`/`load_extension`, so it runs anywhere, with or without CUDA, and does not
depend on catching a build in the act. Verified it bites by putting an eager call back.

**And the conversion had a bug that my own P16 guard caught before it shipped**, which is the
best evidence that guard was worth adding. A module-level `__getattr__` is consulted for
`module.attr` from OUTSIDE; a bare global lookup inside the same module is not. So
`return layer_norm_cuda.layer_norm_bwd(...)`, sitting in a function in that very module, would have
raised `NameError` at call time -- the identical class of bug as the `token_key(L)` one this run
found, and equally invisible to import checks. `ruff --isolated --select F821` flagged all six
sites; each function now goes through an `_ext()` accessor that builds-and-caches.

**Originally deferred, for the record:** Both `layernorm/cuda/__init__.py` and
`transition/cuda/__init__.py` still build at IMPORT time, and `transition/cuda` builds three
extensions eagerly, one of them `sm_90a`-only, so importing that module on an A6000 fails outright.
Making them lazy is right (it is also A2, and every consumer already imports them inside a function
body, so nothing needs the eager build) — but it changes import semantics for vendored CUDA paths
that only a GPU run can verify, and I have no green GPU run to verify against yet. Landing the hang
fix alone is the smaller, checkable change; the lazy conversion goes with the run that can confirm
it.

**Evidence from P0b, raising this from cleanup to a defect:** `dev audit` reports
`import: 0 OK, 2 not OK` on an A6000 -- `transition.cuda` fails with "Error building extension
'transition_b2b_cuda'" because that extension is sm_90a-only and the module builds it at import.
So the audit command exits 1 on every non-Hopper card, for a reason unrelated to what it audits.
The lazy conversion is what fixes it.

**Incidental find, not fixed:** `transition/cuda/__init__.py` passes
`-I/home/psk6950/mathdx_dl/extracted/...` — a personal absolute path baked into a build. It works
for one user on one machine. -> follow-up.

**Operational note for whoever hits this next:**
`rm ~/.cache/torch_extensions/py312_cu128/*/lock`

---

## P16 — the vendored-body lint exemption hid a guaranteed NameError  (B1, F2)

**Found by P0b's GPU run**, which is the argument for having insisted on it. Two regressions from
the harness refactor, both invisible to every CPU check:

1. `triangle_attention/triton/atomic.py`'s `backward` called `token_key(L)` where `L` existed
   only inside einops pattern STRINGS -- never bound. **Every backward of that kernel raised
   NameError.** The module imports, the op registers, `test_registry_complete` resolves the
   checker; the failure needs the kernel to run.
2. `drivers/transition.py` built the hand-CUDA extension from
   `Path(__file__).parent / "transition" / "cuda"`. That was right when the module was
   `kernels/drivers_trans.py`; after the move to `kernels/drivers/transition.py` it resolves one
   level deeper, to `kernels/drivers/transition/cuda/…`, a path that has never existed. Three
   kernels raised `FileNotFoundError` on their first run.

Both are the exact "import and getattr are verified, LAUNCHING is not" gap P0b was written for.

**Why nothing caught the first one:** `pyproject.toml` excludes the vendored kernel bodies with
`= ["ALL"]`, which is right for style (F2) -- but `["ALL"]` cannot be un-ignored per rule, so it
also switched off `F821 undefined-name`, which is not a style rule. It is a guaranteed runtime
`NameError`. Running `ruff --isolated --select F821` over the package found the bug plus four
false positives, all jaxtyping (`Float[torch.Tensor, "d"]` -- ruff reads the shape string as a
forward reference, the same incompatibility the global config already ignores `F722` for).

**Done.** Both bugs fixed. `tests/test_no_undefined_names.py` runs F821 over the whole package with
`--isolated`, deliberately bypassing the per-file-ignores that hid it, and declares the jaxtyping
false positive rather than ignoring it: a finding on a line that is not a jaxtyping annotation
fails. Verified it bites by injecting an undefined name into an `["ALL"]`-excluded file. A second
test fails if F821 ever reports nothing at all, so a broken invocation cannot look like success.

**Related gap, not yet fixed:** `run_all.check_one` catches the checker's exception and reports
only `type(exc).__name__: first line`, so the traceback is lost. Finding *where* `L` was undefined
took a static search instead of reading the failure. -> folded into P2b's run, which needs the
detail anyway.

---

## P17 — `run_all` reported "wrong card" as "wrong answer"  (E2)

**From P0b's stage D**, the first full `run_all` this repo has had: `declared 103, driven 103,
ok 92, failed 11, no driver 0` in 77 s. Of the 11, six were CuTeDSL kernels raising
`expects arch to be sm_90a, but got sm_86` — on an A6000, which is not a defect in them or in the
card. The report was red for hardware those kernels were never written for, and a report that is
always red is one nobody reads.

The repo already draws this distinction on the build side (`is_bad_unit`,
`tests/test_permanent_skip_classification.py`: "a unit that skipped a shape this card cannot hold
is not a bad unit"). `run_all` had no notion of it — and the predicate that *does* know, in
`tests/test_numerical.py`, was the reason: it lived in the test, so the module that produces the
verdict never got it. A D4 violation that cost exactly what D4 says it costs.

**Done, with the declaration doing the work rather than a string match.** Two mechanisms, in order:

1. `meets_arch(row, device)` reads P7's `arch` column and **skips before launching**, so a kernel
   this card cannot run costs no compile and cannot be reported as a failure.
2. `is_arch_gated(detail)` stays as a runtime backstop for a row whose declaration is missing or
   wrong — and a kernel that passes the declared gate then refuses on arch grounds is printed as a
   **registry error**, not absorbed into the skip count. That is the case where `arch` is wrong,
   and it should be loud.

The summary line now has three categories (`ok / failed / skipped (this card is smXX) / no driver`)
plus an accounting check that fails loudly if the four do not sum to `declared` — the previous line
computed "no driver" as `declared - driven`, which silently absorbed anything that fell through.

`tests/test_arch_gating.py` (25 cases) covers both directions, and the important one is the second:
a genuine bug must NOT be classified as an arch refusal. Both bugs this run found carry text a
sloppier predicate could have swallowed (`NameError: name 'L' is not defined`,
`FileNotFoundError: ... transition_cuda.cpp`). It also pins `_sm()`'s numeric ordering, because
`"sm100" < "sm86"` lexically — a string compare would have made every sm100 kernel look runnable
on an A6000.

**A refinement to P7 this exposed.** Of the 9 kernels declared above the floor, only 6 actually
refused. The other 3 are the sm100 *Triton* kernels, which ran fine on sm86: triton compiles for
whatever card it is on, so for a Triton kernel `arch` records the architecture it was WRITTEN for,
while for cute/cuda it is an enforced gate. Those are two different claims sharing one column. The
consumer-facing table is still right (an sm100 Triton kernel is not something to rely on at sm86),
but the column should say which kind it is. -> follow-up.

---

## P18 — opcheck asserted a contract the design deliberately declines  (B1)

**From P0b's stage F**, the first real run of `tests/test_op_contracts_gpu.py` (4 min, so opcheck
was never the time sink either -- all of that was P15's lock). It failed 20+ ops, every one with:

    test_aot_dispatch_dynamic failed with Trying to backward through
    miniworld_engine.<op>.default but no autograd formula was registered.

That is not a defect. `kernels/_compile.py` states the decision and the reason: `register_autograd`
is deliberately unused, because `setup_context` can only save the op's inputs and outputs, so every
intermediate a backward needs (LN stats, the normalised activation) would have to become a forward
return. Keeping `autograd.Function` keeps `save_for_backward` free. Every op is a launch wrapper
called from inside `Function.forward`; **differentiating one directly was never part of its
contract.**

So the test was asserting a property the design does not claim. Fixed by splitting, not by
deleting:

* `test_schema` + `test_faketensor` run against the arguments the op is really called with.
* `test_aot_dispatch_static` + `_dynamic` run with the arguments **detached** — which is the shape
  these ops are actually compiled in, so the compile-path coverage is kept rather than dropped.
* the gradients are checked where they exist, one level up: `test_numerical` compares each kernel's
  dq/dk/dv/dbias against a torch reference through the Function.

**And the exclusion is now asserted, not assumed.** `test_no_op_is_directly_differentiable` requires
backward through each op to RAISE. If it ever succeeds, someone added a formula and the split needs
revisiting.

Getting that test right took two tries, and the first way was wrong in an instructive way: I checked
`_dispatch_has_kernel_for_dispatch_key(name, "Autograd")`, which is **True for every custom_op** —
`custom_op` installs a not-implemented Autograd fallback, so the dispatcher cannot tell a real
formula from the fallback and the test would have failed for all 20+ ops. Verified the discriminator
on a probe pair instead (plain raises "no autograd formula"; `register_autograd` succeeds), and the
whole diagnosis on a minimal repro with no GPU: detached passes, `requires_grad` fails with exactly
stage F's message.

**Still to run:** the fix itself is unverified on a device. It is a GPU-marked file, so the CPU suite
cannot exercise it.
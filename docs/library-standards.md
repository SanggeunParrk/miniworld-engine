# What a tier-1 library owes its consumers

This file is the standard `miniworld-engine` holds itself to. It is not a wish list: every
criterion below states **the failure it prevents** and **how it is enforced mechanically**, because
a standard nobody can check is a preference. Where this repo does not yet meet one, the status line
says so, and `plan.md` carries the work.

Two rules govern the whole document.

**A standard is enforced or it is decoration.** "We keep names consistent" is not a standard;
`tests/test_bench_target_vocabulary.py` is. Every criterion here names the check that fails when it
is violated, or admits there isn't one.

**The check must fail for the right reason.** A test that passes because it stopped finding
anything is worse than no test. `tests/test_lazy_import_targets.py::test_there_are_lazy_wrappers_to_check`
exists for exactly this: the import-style change made its collector return zero cases, and without
that guard the file would have kept passing while checking nothing.

---

## A. Contract — what a consumer may rely on

### A1. The public surface is small, named, and frozen

A consumer cannot depend on what it cannot see, and cannot plan around what changes silently. The
library declares its public names in one place, and a change to that set is a deliberate, recorded
act rather than a side effect of a refactor.

*Prevents:* a rename that compiles locally and breaks every downstream import; a private helper
that becomes load-bearing for a consumer because nothing said it was private.

*Enforced by:* `tests/test_public_api.py` freezes `kernels.__all__` and `ops.__all__` against
`_CONTRACT` / `_OPS_CONTRACT`. Changing either fails the suite and the message says to update the
CHANGELOG.

*Status:* **met.**

### A2. Importing the package does nothing

An import must not compile a kernel, touch a GPU, read a cache, or spend seconds. Consumers import
libraries inside test collection, inside CLI startup, inside other libraries' imports.

*Prevents:* a consumer's unrelated test suite paying a 2-minute triton compile; an import that
fails on a machine with no CUDA.

*Enforced by:* `tests/test_public_api.py` asserts the import is side-effect-free; the whole CPU
suite runs with `CUDA_VISIBLE_DEVICES=""` in CI.

*Status:* **met.**

### A3. A typed library ships its types

Annotating the source is half the job. Without a PEP 561 marker, every consumer sees `Any` for
every symbol, and the library's own type gate protects only the library.

*Prevents:* a consumer whose type checker cannot see a single signature — so the effort spent
getting `ty` to zero buys them nothing.

*Enforced by:* a test asserting `src/miniworld_engine/py.typed` exists and is shipped by
`[tool.setuptools.package-data]`.

*Status:* **NOT met** — there is no `py.typed`. -> `plan.md` P1.

### A4. Version, and a documented path for removal

SemVer is a promise about what a version number means. A library that claims it must also say how
a name gets removed: deprecated in which release, warning in which, gone in which.

*Prevents:* a consumer pinned to `>=0.1` discovering a removal; or, worse, the library never
removing anything because there is no procedure.

*Enforced by:* CHANGELOG structure (present) plus a stated deprecation policy and a test that a
name marked deprecated actually emits a `DeprecationWarning`.

*Status:* **partially met** — SemVer claimed in CHANGELOG, `version = "0.1.0"`, no deprecation
policy and no mechanism. -> `plan.md` P6.

### A5. The supported hardware is stated, and unsupported hardware fails clearly

A kernel library is not portable by default. Which architectures are supported, which kernels need
which arch, and what happens on a card that has neither, is part of the contract.

*Prevents:* a consumer on an unsupported card getting a CUTLASS build error 40 frames deep instead
of "this backend needs SM90; the triton path covers your card".

*Enforced by:* a support matrix checked against the registry's own arch requirements, so it cannot
drift from the code.

*Status:* **NOT met** — arch gates live as asserts inside individual checkers (`"SM90 (H100)
only"`); no matrix anywhere. -> `plan.md` P7.

---

## B. Correctness — what "right" means, and how it is proven

### B1. Every kernel has a reference, and the reference is the definition

A fused kernel is only correct relative to something. That something is a plain torch expression a
reader can check against the algorithm, kept next to the kernel, and it is what "correct" means for
that family.

*Prevents:* the state this repo was in before `checks/` existed — 56 kernels reached by a driver
with no reference at all, where "ok" meant "did not raise".

*Enforced by:* `tests/test_kernel_layout.py` requires `reference.py` per family;
`tests/test_registry_complete.py::test_every_kernel_with_a_driver_declares_a_checker`;
`tests/test_numerical.py` runs all 99 declared checkers on GPU.

*Status:* **met** for existence and execution. See B2 for the band.

### B2. The tolerance is declared per kernel, not globally

One global band is the weakest kernel's band applied to all of them. A kernel that should be
bit-exact (a transpose, a mask fold, a gate multiply) passing at 5% relative error is a test that
cannot see a real regression.

*Prevents:* a numerics bug inside the band. A reduction-order change that costs 1e-3 is invisible
under 5e-2, and 1e-3 on a residual accumulated over 48 blocks is not invisible in the model.

*Enforced by:* a declared tolerance per registry row, with `check_one` comparing against that
row's band rather than a module constant. A kernel that wants a wider band has to say so, in the
file that declares it, where a reviewer sees it.

*Status:* **NOT met** — `run_all.check_one` applies one band, `rel < 5e-2`, to all 99.
-> `plan.md` P2.

### B3. Coverage is measured against a declaration, never against itself

The denominator must be something the repo states, not something derived from the run. Derive it
and every unreachable case drops out of numerator and denominator together, and coverage reads
100% forever.

*Prevents:* exactly that. It is why `_report_coverage` reads `registry.csv` and not the set of ops
that happened to fire.

*Enforced by:* `registry.csv` as the declared inventory; `tests/test_registry_complete.py`,
`tests/test_declared_dtype_coverage.py`, `tests/test_spread_shape_key.py`; `dev audit` for the
cache side.

*Status:* **met.**

### B4. The tests exercise the shapes that break kernels

Aligned, power-of-two shapes never execute a boundary mask. A suite built only from them cannot
observe a missing `mask=` on a tail tile.

*Prevents:* a tail-tile bug shipping. This repo already found it: every default driver extent was a
multiple of 128, so no kernel's boundary mask had ever run.

*Enforced by:* `MINIWORLD_SHAPE_MODE=ragged` on the driver extents, `MINIWORLD_DRIVER_DTYPE=fp32`,
and the atom/token side split — all import-time, all in `drivers/`.

*Status:* **met** as a mechanism, **NOT met** as a gate: nothing runs the ragged mode
automatically, so the mechanism protects nothing. -> `plan.md` P3.

### B5. Determinism is stated

A library that autotunes picks a different kernel config per GPU and per cache state. Whether two
runs of the same input on the same card give bitwise-identical output is a question consumers must
be able to answer without reading the source.

*Prevents:* a consumer chasing a "nondeterminism bug" that is the autotuner, or assuming
reproducibility the library never promised.

*Enforced by:* a stated policy plus a test that two calls under one cache state agree bitwise.

*Status:* **NOT met** — nothing states it. -> `plan.md` P8.

---

## C. Evidence — performance claims

### C1. No number without provenance

A benchmark number is a claim about a machine, a config set, a dtype, a compile mode and a
version. Detached from those it is folklore, and folklore is what makes a team re-run everything
before trusting anything.

*Prevents:* the failure this repo already had — 330 of 350 committed tables said
`compiled=True, cudagraph=manual` while four of eight module benches silently ran eager, so a third
of the table was eager code labelled compiled.

*Enforced by:* the long-form CSV is the artifact and carries device, torch/cuda version, mode,
`compiled`, `cudagraph`, `compile_wrap`, precision, dtypes and the execution path per row;
`tests/test_compiled_flag_is_what_ran.py` asserts the `compiled` column says what ran.

*Status:* **met** for the tables.

### C2. A number quoted in prose is traceable to its artifact

Docs and CHANGELOGs quote numbers. Each should name the artifact it came from, or be checkable
against it.

*Prevents:* a doc that outlives its measurement.

*Enforced by:* not yet. Candidate: a test that every `N.NN ms`-shaped claim in
`benchmarks/RESULTS.md` matches a value in a committed table.

*Status:* **NOT met.** -> `plan.md` P9.

### C3. The comparison is fair by construction, and the regime is named

A speedup against an unfair baseline is not a speedup. The baseline must be the same shapes, the
same dtype, the same compile regime — and which regime was used has to be on the artifact, because
a captured CUDA graph and a compiled module measure different things.

*Prevents:* the two failures already recorded here: a capture that benched the PyTorch reference
and reported it as ours, and a default `cudagraph=manual` read as a recommendation when the
measurement shows compile-only *winning* by 4.46x on the module with unfused work around its
kernels.

*Enforced by:* `cudagraph` / `compile` / `compile_wrap` are required config fields, recorded per
row; `bench.py` refuses a run that is neither compiled nor graphed.

*Status:* **met.**

---

## D. Coherence — one name, one shape, one source of truth

### D1. One name per thing, across every surface

Code, CLI, docs, config, data and directory names are one vocabulary. The same computation is not
`tri_attn` in one table and `triangle_attention` in another.

*Prevents:* `bench_kernel triangle_attention` returning "unknown target" for a kernel that exists;
a doc command nobody can run.

*Enforced by:* `tests/test_bench_target_vocabulary.py` ties four namespaces together (bench.py's
tables, the CLI's, `builder.CASE_NAMES`, the directory tree);
`tests/test_cli_documented_commands.py` parses every `miniworld-engine ...` line in the docs.

*Status:* **met.**

### D2. Two names may collide only where a level distinguishes them

Where one word legitimately names two things at different levels (the `triangle_attention` kernel
and the `triangle_attention` module), the fix is an explicit level, not mangling one of the names.

*Enforced by:* `bench.py`'s `level` field, and the same test as D1.

*Status:* **met.**

### D3. Every instance of a kind has the same shape

A new kernel family, a new module, a new bench target: there is exactly one template, and a reader
who has seen one has seen them all.

*Prevents:* each new instance copying whichever neighbour its author opened. This repo had one
family that was not a package at all and `interface.py` for four of thirteen.

*Enforced by:* `tests/test_kernel_layout.py`, `tests/test_module_layout.py`,
`tests/test_registry_complete.py::test_the_harness_is_one_module_per_family`,
`tests/test_bench_config_per_target.py`.

*Status:* **met.**

### D4. One writer per piece of state

Every value has exactly one place that produces it. Two hand-maintained tables keyed by the same
names will drift, and the drift will be silent.

*Prevents:* `bench_module all` running eight of nine targets because "all" read one of two tables;
`augmented_attention_token` and `_atom` sharing a directory named after neither.

*Enforced by:* the merged `MODULE_TARGETS`; `builder.CASE_NAMES` pinned to `cases()`;
`target_dir` derived from `level` rather than tabulated.

*Status:* **met**, with one scheduled exception: `configs/grid` exists at the repo root *and*
inside the package. It is deliberate, documented, and `tests/test_default_config_set.py` asserts
the two are byte-identical — but the duplication is meant to end. -> `plan.md` P10.

### D5. A name collision that Python resolves by accident is a defect

`autotune/configs.py` wins over `autotune/configs/` only because the directory has no
`__init__.py`. Adding one — the reflex when making a shipped asset importable — silently replaces
the module with a namespace package and breaks every import of the config reader.

*Enforced by:* not yet. -> `plan.md` P4.

---

## E. Operability

### E1. One command per task, and everything it depends on is an argument

A run's behaviour must not live in shell state that nothing records.

*Prevents:* the two runs this repo lost to it — a capture that benched the reference and reported
it as ours, and one that skipped every kernel on the losing side of a dispatch decision. Both
looked like successful runs.

*Enforced by:* `miniworld-engine` has three top-level commands; every switch is a flag; the config
set is an argument; `build` decomposes, runs and merges in one invocation.

*Status:* **met** structurally, **NOT verified** — no run since the harness refactor has proven
`build all` end to end. -> `plan.md` P5.

### E2. Failure is distinguishable from absence

"This card cannot hold this shape" is a permanent, correct answer. Counting it as a failure made a
resumed job report "0 ok, 9 failed" and refuse to merge.

*Enforced by:* `is_bad_unit`, `tests/test_permanent_skip_classification.py`.

*Status:* **met.**

### E3. A partial result is kept and its holes are named

One OOM must not discard 526 good measurements. Merge what succeeded, report the holes, and offer
`--strict` for CI.

*Enforced by:* `_merge_built_shards`, `tests/test_shard_merge.py`, `dev audit`.

*Status:* **met.**

### E4. Every error message names the fix

An error is a place a human is standing. It should say what to do, not just what happened.

*Prevents:* the class of message this repo replaced — `dynamic_func() missing 1 required
positional argument` for a config spec that lost an axis.

*Enforced by:* convention only, and deliberately so: a lint rule here would be noise. Reviewed by
hand.

*Status:* **partially met.**

### E5. The artifact records how it was made

A cache, a table, a figure: each says what produced it.

*Enforced by:* the `provenance` block in each `data/<op>/<gpu>.json` (build time, torch, triton);
the CSV's version columns; `tests/test_shipped_cache_wellformed.py`.

*Status:* **met.**

### E6. Cost is predictable before it is paid

`build all` is hours of GPU time. A user must be able to see the size of the work before starting
it, and a typo must not cost minutes.

*Enforced by:* `build` prints its unit count before running; `_reject_unknown_build_target`
validates both namespaces before the first import.

*Status:* **met.**

---

## F. Stewardship

### F1. The gates gate

Lint, types and tests run clean, block on failure, and have no `|| true`. A suppression carries a
reason and is itself checked.

*Prevents:* a green step that checks nothing — this repo's `ty` step once ran against an install
with no torch, so every attribute was `Unknown` and the step could not see what it was checking.

*Enforced by:* `pixi run ci` and `.github/workflows/ci.yml` run the same three gates over the same
paths; `RUF100` fails a `# noqa` that suppresses nothing; the rule set and every excluded family
are justified in `pyproject.toml`.

*Status:* **met.**

### F2. Vendored code has a stated boundary

Faithful ports are not held to local style, and that decision is written down where the exclusion
lives — not discovered by a reader wondering why one directory is different.

*Enforced by:* `[tool.ruff.lint.per-file-ignores]` and `[tool.ty.src].exclude` name the same set,
with the reason inline.

*Status:* **met.**

### F3. No orphan code

A module nobody imports, a second CLI one letter from the real one, a cache written under names
nothing reads: each is a trap for the next reader.

*Enforced by:* not automatically. `autotune/build.py` is the current instance.
-> `plan.md` P11.

### F4. Docs are either executable or dated

A reference doc must be current. A record of a past investigation must say it is one. The failure
mode is a reference doc that has quietly become a record.

*Prevents:* a reader following `benchmarks/kernels/layernorm_linear/artifacts` — a path that never
existed.

*Enforced by:* `tests/test_cli_documented_commands.py` for commands. Op names and paths in prose
are not checked.

*Status:* **partially met** — `docs/kernels/l2-swizzle.md` names 21 pre-rename ops.
-> `plan.md` P12.

### F5. Working notes are not repository furniture

A root `todo.md` of dated findings is a private notebook in a public hallway. Either the items are
live work — in which case they belong in a tracker or a plan — or they are history, in which case
they belong under `docs/`.

*Status:* **NOT met.** -> `plan.md` P13.

### F6. A consumer can contribute

The repo states how to run the gates, what a change must include (test, CHANGELOG entry), and what
the review bar is.

*Status:* **NOT met** — no CONTRIBUTING. The information exists, scattered across README and
pyproject comments. -> `plan.md` P14.

---

## What is deliberately *not* a criterion here

**100% line coverage.** The suite's job is to make the repo's *claims* checkable. A coverage number
is a proxy that rewards testing the easy half.

**A rule for every lint family.** `pyproject.toml` names the families this codebase declines and
why, with the measurement (709 `N` findings are math notation; 512 `PLC0415` are load-bearing lazy
imports). Turning them on to reach a number would mean 700 suppressions, which is the state this
repo just left.

**Uniform style inside vendored kernel bodies.** See F2.

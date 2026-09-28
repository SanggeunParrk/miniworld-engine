# Standards

Three documents in one: [project direction](#project-direction-building-on-anthropics-inference-work), [library standard](#what-a-tier-1-library-owes-its-consumers) (criteria A–F) and [product standard](#what-a-product-owes-someone-who-is-not-its-author) (criteria G–).

## Project direction: building on Anthropic's inference work

Adopted: 2026-09-19.

### Acknowledgment

We developed our own inference kernels for biomolecular models. Anthropic's
biomolecular modeling optimization release achieved substantially stronger
inference results than our effort. We recognize that achievement and are
choosing to build on it. The next phase of miniworld-engine carries that work
forward by integrating its inference implementations and extending them with
high-performance training support.

우리도 생체분자 모델의 추론 커널을 자체 개발해 왔다. 그러나 Anthropic이
공개한 추론 최적화의 성과가 우리의 자체 개발보다 훨씬 뛰어났음을 인정한다.
우리는 그 성과를 존중하며, 해당 개발을 계승한다. miniworld-engine의 다음
단계는 그 추론 구현을 통합하고, 이를 바탕으로 고성능 학습 지원을 확장하는 것이다.

Primary sources:

- [Anthropic: How Claude is uplifting biomolecular modeling](https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling)
- [Upstream code](https://github.com/anthropics/uplifting-biomolecular-modeling)
- [Shared kernels and FlashPairformer](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/main/common/opt_core/README.md)
- [Upstream attribution and licensing notices](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/main/NOTICE)

### What we retain and what changes

We retain miniworld-engine's operation interfaces, backend dispatch, configuration
search, tuning caches, correctness checks, and benchmark infrastructure. Existing
kernels and measurements remain available as baselines and, where useful,
fallbacks. Their previous fusion boundaries are no longer constraints on new work.

We will inventory kernels across the upstream project, including shared and
model-specific implementations, preserve their provenance, and integrate them
into the engine. FlashPairformer is part of this scope, not its entirety.
Inventory entries must identify variants, duplicate implementations, supported
shapes and dtypes, dependencies, and any existing backward support.

The development sequence is:

1. Pin the upstream revision and preserve source hashes, licenses, and notices.
2. Integrate and reproduce its inference kernels on H100 with matched inputs,
   numerical policies, masks, residual behavior, and timing methods. Record the
   implementation actually executed, including any fallback.
3. Profile with Nsight Compute. Assess the relevant compute, HBM, and L2 limits
   per kernel and shape, rather than using SM utilization alone. Deprioritize
   implementation tuning near the relevant roofline; separately assess designs
   that reduce the amount of work or memory traffic.
4. Build training implementations using the upstream algorithms and implementation
   ideas. Optimize backward together with the training forward's saved tensors
   and recomputation strategy, including dropout, masks, and residual gradients.
5. Validate outputs and all parameter/input gradients, then measure forward plus
   backward time, peak memory, and model training performance.

### Attribution and performance claims

Our intended contribution is **high-performance training support built on
Anthropic's inference optimizations**, together with their integration into
miniworld-engine. Upstream inference algorithms and implementations retain their
authors' credit. We will distinguish unchanged imports, modified upstream code,
and new training code, and preserve third-party attribution carried by upstream.

The acknowledgment above states our assessment and development decision. Each
quantitative performance claim still requires a reproducible comparison with
matched hardware, shapes, precision, and execution mode. Model-level inference
speedups do not establish kernel-level speedups, and inference results do not
establish training speedups.

At adoption of this direction, the full import, H100 profiling campaign, and
training extension are planned work. This document does not declare them
complete. Earlier development records describe the implementations and
measurements at their recorded dates; they remain historical evidence.

## What a tier-1 library owes its consumers

> **This is the inner layer.** These criteria ask whether the code is a good library.
> They stop at the edge of this machine. `docs/standards.md` covers the rest —
> portability, distribution, releases, verification at a distance, and the consumer —
> and is the document `archive/records-20260928:docs/records/product-plan.md` is derived from.

This file is the standard `miniworld-engine` holds itself to. It is not a wish list: every
criterion below states **the failure it prevents** and **how it is enforced mechanically**, because
a standard nobody can check is a preference. Where this repo does not yet meet one, the status line
says so, and `archive/records-20260928:docs/records/product-plan.md` carries the work.

Two rules govern the whole document.

**A standard is enforced or it is decoration.** "We keep names consistent" is not a standard;
`tests/layout/test_bench_target_vocabulary.py` is. Every criterion here names the check that fails when it
is violated, or admits there isn't one.

**The check must fail for the right reason.** A test that passes because it stopped finding
anything is worse than no test. `tests/builder/test_lazy_import_targets.py::test_there_are_lazy_wrappers_to_check`
exists for exactly this: the import-style change made its collector return zero cases, and without
that guard the file would have kept passing while checking nothing.

---

### A. Contract — what a consumer may rely on

#### A1. The public surface is small, named, and frozen

A consumer cannot depend on what it cannot see, and cannot plan around what changes silently. The
library declares its public names in one place, and a change to that set is a deliberate, recorded
act rather than a side effect of a refactor.

*Prevents:* a rename that compiles locally and breaks every downstream import; a private helper
that becomes load-bearing for a consumer because nothing said it was private.

*Enforced by:* `tests/compile/test_public_api.py` freezes `kernels.__all__` and `ops.__all__` against
`_CONTRACT` / `_OPS_CONTRACT`. Changing either fails the suite and the message says to update the
CHANGELOG.

*Status:* **met.**

#### A2. Importing the package does nothing

An import must not compile a kernel, touch a GPU, read a cache, or spend seconds. Consumers import
libraries inside test collection, inside CLI startup, inside other libraries' imports.

*Prevents:* a consumer's unrelated test suite paying a 2-minute triton compile; an import that
fails on a machine with no CUDA.

*Enforced by:* `tests/compile/test_public_api.py` asserts the import is side-effect-free; the whole CPU
suite runs with `CUDA_VISIBLE_DEVICES=""` in CI.

*Status:* **met.**

#### A3. A typed library ships its types

Annotating the source is half the job. Without a PEP 561 marker, every consumer sees `Any` for
every symbol, and the library's own type gate protects only the library.

*Prevents:* a consumer whose type checker cannot see a single signature — so the effort spent
getting `ty` to zero buys them nothing.

*Enforced by:* a test asserting `src/miniworld_engine/py.typed` exists and is shipped by
`[tool.setuptools.package-data]`.

*Status:* **NOT met** — there is no `py.typed`. -> `archive/records-20260928:docs/records/product-plan.md` P1.

#### A4. Version, and a documented path for removal

SemVer is a promise about what a version number means. A library that claims it must also say how
a name gets removed: deprecated in which release, warning in which, gone in which.

*Prevents:* a consumer pinned to `>=0.1` discovering a removal; or, worse, the library never
removing anything because there is no procedure.

*Enforced by:* CHANGELOG structure (present) plus a stated deprecation policy and a test that a
name marked deprecated actually emits a `DeprecationWarning`.

*Status:* **partially met** — SemVer claimed in CHANGELOG, `version = "0.1.0"`, no deprecation
policy and no mechanism. -> `archive/records-20260928:docs/records/product-plan.md` P6.

#### A5. The supported hardware is stated, and unsupported hardware fails clearly

A kernel library is not portable by default. Which architectures are supported, which kernels need
which arch, and what happens on a card that has neither, is part of the contract.

*Prevents:* a consumer on an unsupported card getting a CUTLASS build error 40 frames deep instead
of "this backend needs SM90; the triton path covers your card".

*Enforced by:* a support matrix checked against the registry's own arch requirements, so it cannot
drift from the code.

*Status:* **NOT met** — arch gates live as asserts inside individual checkers (`"SM90 (H100)
only"`); no matrix anywhere. -> `archive/records-20260928:docs/records/product-plan.md` P7.

---

### B. Correctness — what "right" means, and how it is proven

#### B1. Every kernel has a reference, and the reference is the definition

A fused kernel is only correct relative to something. That something is a plain torch expression a
reader can check against the algorithm, kept next to the kernel, and it is what "correct" means for
that family.

*Prevents:* the state this repo was in before `checks/` existed — 56 kernels reached by a driver
with no reference at all, where "ok" meant "did not raise".

*Enforced by:* `tests/layout/test_kernel_layout.py` requires `reference.py` per family;
`tests/registry/test_registry_complete.py::test_every_kernel_with_a_driver_declares_a_checker`;
`tests/numerics/test_numerical.py` runs all 99 declared checkers on GPU.

*Status:* **met** for existence and execution. See B2 for the band.

#### B2. The tolerance is declared per kernel, not globally

One global band is the weakest kernel's band applied to all of them. A kernel that should be
bit-exact (a transpose, a mask fold, a gate multiply) passing at 5% relative error is a test that
cannot see a real regression.

*Prevents:* a numerics bug inside the band. A reduction-order change that costs 1e-3 is invisible
under 5e-2, and 1e-3 on a residual accumulated over 48 blocks is not invisible in the model.

*Enforced by:* a declared tolerance per registry row, with `check_one` comparing against that
row's band rather than a module constant. A kernel that wants a wider band has to say so, in the
file that declares it, where a reviewer sees it.

*Status:* **NOT met** — `run_all.check_one` applies one band, `rel < 5e-2`, to all 99.
-> `archive/records-20260928:docs/records/product-plan.md` P2.

#### B3. Coverage is measured against a declaration, never against itself

The denominator must be something the repo states, not something derived from the run. Derive it
and every unreachable case drops out of numerator and denominator together, and coverage reads
100% forever.

*Prevents:* exactly that. It is why `_report_coverage` reads `registry.csv` and not the set of ops
that happened to fire.

*Enforced by:* `registry.csv` as the declared inventory; `tests/registry/test_registry_complete.py`,
`tests/registry/test_declared_dtype_coverage.py`, `tests/autotune/test_spread_shape_key.py`; `dev audit` for the
cache side.

*Status:* **met.**

#### B4. The tests exercise the shapes that break kernels

Aligned, power-of-two shapes never execute a boundary mask. A suite built only from them cannot
observe a missing `mask=` on a tail tile.

*Prevents:* a tail-tile bug shipping. This repo already found it: every default driver extent was a
multiple of 128, so no kernel's boundary mask had ever run.

*Enforced by:* `MINIWORLD_SHAPE_MODE=ragged` on the driver extents, `MINIWORLD_DRIVER_DTYPE=fp32`,
and the atom/token side split — all import-time, all in `drivers/`.

*Status:* **met** as a mechanism, **NOT met** as a gate: nothing runs the ragged mode
automatically, so the mechanism protects nothing. -> `archive/records-20260928:docs/records/product-plan.md` P3.

#### B5. Determinism is stated

A library that autotunes picks a different kernel config per GPU and per cache state. Whether two
runs of the same input on the same card give bitwise-identical output is a question consumers must
be able to answer without reading the source.

*Prevents:* a consumer chasing a "nondeterminism bug" that is the autotuner, or assuming
reproducibility the library never promised.

*Enforced by:* a stated policy plus a test that two calls under one cache state agree bitwise.

*Status:* **NOT met** — nothing states it. -> `archive/records-20260928:docs/records/product-plan.md` P8.

---

### C. Evidence — performance claims

#### C1. No number without provenance

A benchmark number is a claim about a machine, a config set, a dtype, a compile mode and a
version. Detached from those it is folklore, and folklore is what makes a team re-run everything
before trusting anything.

*Prevents:* the failure this repo already had — 330 of 350 committed tables said
`compiled=True, cudagraph=manual` while four of eight module benches silently ran eager, so a third
of the table was eager code labelled compiled.

*Enforced by:* the long-form CSV is the artifact and carries device, torch/cuda version, mode,
`compiled`, `cudagraph`, `compile_wrap`, precision, dtypes and the execution path per row;
`tests/compile/test_compiled_flag_is_what_ran.py` asserts the `compiled` column says what ran.

*Status:* **met** for the tables.

#### C2. A number quoted in prose is traceable to its artifact

Docs and CHANGELOGs quote numbers. Each should name the artifact it came from, or be checkable
against it.

*Prevents:* a doc that outlives its measurement.

*Enforced by:* not yet. Candidate: a test that every `N.NN ms`-shaped claim in
`benchmarks/results.md` matches a value in a committed table.

*Status:* **NOT met.** -> `archive/records-20260928:docs/records/product-plan.md` P9.

#### C3. The comparison is fair by construction, and the regime is named

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

### D. Coherence — one name, one shape, one source of truth

#### D1. One name per thing, across every surface

Code, CLI, docs, config, data and directory names are one vocabulary. The same computation is not
`tri_attn` in one table and `triangle_attention` in another.

*Prevents:* `bench_kernel triangle_attention` returning "unknown target" for a kernel that exists;
a doc command nobody can run.

*Enforced by:* `tests/layout/test_bench_target_vocabulary.py` ties four namespaces together (bench.py's
tables, the CLI's, `builder.CASE_NAMES`, the directory tree);
`tests/layout/test_cli_documented_commands.py` parses every `miniworld-engine ...` line in the docs.

*Status:* **met.**

#### D2. Two names may collide only where a level distinguishes them

Where one word legitimately names two things at different levels (the `triangle_attention` kernel
and the `triangle_attention` module), the fix is an explicit level, not mangling one of the names.

*Enforced by:* `bench.py`'s `level` field, and the same test as D1.

*Status:* **met.**

#### D3. Every instance of a kind has the same shape

A new kernel family, a new module, a new bench target: there is exactly one template, and a reader
who has seen one has seen them all.

*Prevents:* each new instance copying whichever neighbour its author opened. This repo had one
family that was not a package at all and `interface.py` for four of thirteen.

*Enforced by:* `tests/layout/test_kernel_layout.py`, `tests/layout/test_module_layout.py`,
`tests/registry/test_registry_complete.py::test_the_harness_is_one_module_per_family`,
`tests/layout/test_bench_config_per_target.py`.

*Status:* **met.**

#### D4. One writer per piece of state

Every value has exactly one place that produces it. Two hand-maintained tables keyed by the same
names will drift, and the drift will be silent.

*Prevents:* `bench_module all` running eight of nine targets because "all" read one of two tables;
`augmented_attention_token` and `_atom` sharing a directory named after neither.

*Enforced by:* the merged `MODULE_TARGETS`; `builder.CASE_NAMES` pinned to `cases()`;
`target_dir` derived from `level` rather than tabulated.

*Status:* **met.** The scheduled exception is closed: `configs/grid` was duplicated at the repo
root and inside the package, and every other config set lived only at the root — so a short name
resolved against a different root depending on the caller, and a wheel could reach only one of
them. All eleven sets are packaged now, `configs.config_set(name)` is the single resolver, and
`tests/autotune/test_default_config_set.py` asserts a repo-root directory cannot shadow the package.

#### D5. A name collision that Python resolves by accident is a defect

`autotune/configs.py` wins over `autotune/configs/` only because the directory has no
`__init__.py`. Adding one — the reflex when making a shipped asset importable — silently replaces
the module with a namespace package and breaks every import of the config reader.

*Enforced by:* not yet. -> `archive/records-20260928:docs/records/product-plan.md` P4.

---

### E. Operability

#### E1. One command per task, and everything it depends on is an argument

A run's behaviour must not live in shell state that nothing records.

*Prevents:* the two runs this repo lost to it — a capture that benched the reference and reported
it as ours, and one that skipped every kernel on the losing side of a dispatch decision. Both
looked like successful runs.

*Enforced by:* `miniworld-engine` has three top-level commands; every switch is a flag; the config
set is an argument; `build` decomposes, runs and merges in one invocation.

*Status:* **met** structurally, **NOT verified** — no run since the harness refactor has proven
`build all` end to end. -> `archive/records-20260928:docs/records/product-plan.md` P5.

#### E2. Failure is distinguishable from absence

"This card cannot hold this shape" is a permanent, correct answer. Counting it as a failure made a
resumed job report "0 ok, 9 failed" and refuse to merge.

*Enforced by:* `is_bad_unit`, `tests/registry/test_permanent_skip_classification.py`.

*Status:* **met.**

#### E3. A partial result is kept and its holes are named

One OOM must not discard 526 good measurements. Merge what succeeded, report the holes, and offer
`--strict` for CI.

*Enforced by:* `_merge_built_shards`, `tests/autotune/test_shard_merge.py`, `dev audit`.

*Status:* **met.**

#### E4. Every error message names the fix

An error is a place a human is standing. It should say what to do, not just what happened.

*Prevents:* the class of message this repo replaced — `dynamic_func() missing 1 required
positional argument` for a config spec that lost an axis.

*Enforced by:* convention only, and deliberately so: a lint rule here would be noise. Reviewed by
hand.

*Status:* **partially met.**

#### E5. The artifact records how it was made

A cache, a table, a figure: each says what produced it.

*Enforced by:* the `provenance` block in each `data/<op>/<gpu>.json` (build time, torch, triton);
the CSV's version columns; `tests/autotune/test_shipped_cache_wellformed.py`.

*Status:* **met.**

#### E6. Cost is predictable before it is paid

`build all` is hours of GPU time. A user must be able to see the size of the work before starting
it, and a typo must not cost minutes.

*Enforced by:* `build` prints its unit count before running; `_reject_unknown_build_target`
validates both namespaces before the first import.

*Status:* **met.**

---

### F. Stewardship

#### F1. The gates gate

Lint, types and tests run clean, block on failure, and have no `|| true`. A suppression carries a
reason and is itself checked.

*Prevents:* a green step that checks nothing — this repo's `ty` step once ran against an install
with no torch, so every attribute was `Unknown` and the step could not see what it was checking.

*Enforced by:* `pixi run ci` and `.github/workflows/ci.yml` run the same three gates over the same
paths; `RUF100` fails a `# noqa` that suppresses nothing; the rule set and every excluded family
are justified in `pyproject.toml`.

*Status:* **met.**

#### F2. Vendored code has a stated boundary

Faithful ports are not held to local style, and that decision is written down where the exclusion
lives — not discovered by a reader wondering why one directory is different.

*Enforced by:* `[tool.ruff.lint.per-file-ignores]` and `[tool.ty.src].exclude` name the same set,
with the reason inline.

*Status:* **met.**

#### F3. No orphan code

A module nobody imports, a second CLI one letter from the real one, a cache written under names
nothing reads: each is a trap for the next reader.

*Enforced by:* not automatically. `autotune/build.py` is the current instance.
-> `archive/records-20260928:docs/records/product-plan.md` P11.

#### F4. Docs are either executable or dated

A reference doc must be current. A record of a past investigation must say it is one. The failure
mode is a reference doc that has quietly become a record.

*Prevents:* a reader following `benchmarks/kernels/layernorm_linear/artifacts` — a path that never
existed.

*Enforced by:* `tests/layout/test_cli_documented_commands.py` for commands. Op names and paths in prose
are not checked.

*Status:* **partially met** — `docs/kernels/autotune-l2-swizzle.md` names 21 pre-rename ops.
-> `archive/records-20260928:docs/records/product-plan.md` P12.

#### F5. Working notes are not repository furniture

A root `todo.md` of dated findings is a private notebook in a public hallway. Either the items are
live work — in which case they belong in a tracker or a plan — or they are history, in which case
they belong under `docs/`.

*Status:* **NOT met.** -> `archive/records-20260928:docs/records/product-plan.md` P13.

#### F6. A consumer can contribute

The repo states how to run the gates, what a change must include (test, CHANGELOG entry), and what
the review bar is.

*Status:* **NOT met** — no CONTRIBUTING. The information exists, scattered across README and
pyproject comments. -> `archive/records-20260928:docs/records/product-plan.md` P14.

---

### What is deliberately *not* a criterion here

**100% line coverage.** The suite's job is to make the repo's *claims* checkable. A coverage number
is a proxy that rewards testing the easy half.

**A rule for every lint family.** `pyproject.toml` names the families this codebase declines and
why, with the measurement (709 `N` findings are math notation; 512 `PLC0415` are load-bearing lazy
imports). Turning them on to reach a number would mean 700 suppressions, which is the state this
repo just left.

**Uniform style inside vendored kernel bodies.** See F2.

## What a product owes someone who is not its author

`docs/standards.md` asked whether this code is a good library. Every one of its 30
criteria is about the code itself: is the surface frozen, is the tolerance declared, does the
name mean one thing. That was the right question and it is nearly answered.

It is also the wrong ceiling. A library can satisfy all of A–F and still be useless to
everyone but the person who wrote it, because none of those criteria ever leave the author's
machine. This file is the standard for the part that does.

The distinction is not theoretical. On **2026-08-25** an A100 run on a different cluster
burned ten hours and produced nothing. Nothing in the code was wrong. The commit that fixes
the unit count (527 → 859) had been sitting unpushed for 70 commits, and a 13-hour-old JIT
lock file made `FileBaton.wait()` poll forever with no message. Both are perfect scores under
A–F and total failures under this document.

Three rules govern it.

**A standard is enforced or it is decoration.** Inherited from the library standard and it
still binds: every criterion below names the check that fails when it is violated, or admits
there isn't one.

**The check must fail for the right reason.** Also inherited, and on **2026-08-25** it turned out
to be the most-violated rule here. Running the things this repository says it does -- a clean
clone, the GPU-marked suite, the build-system audit, the coverage replay -- produced nine defects
(`archive/records-20260928:docs/records/product-plan.md` §H), and seven were checks that could not fail, could not pass, or answered a question
nobody asked: a gate asserting per-clone git config that only the author's machine has; an audit
printing 139 findings about its own missing arguments and exiting 1 every run; a marker documented
as "needs a CUDA device" that failed instead of skipping without one; a `missing_pairs 0` that
could not see 363 real holes; a miss set that only grew, so a filled cache could never be observed
as filled. A green suite that stopped looking is worse than a red one, and a red one that is
always red is the same thing.

**The author is not the judge.** A criterion is met when a machine that is not this one, or a
person who is not the author, demonstrates it. "It works for me" is the null hypothesis this
document exists to reject. Where the only evidence is the author running something by hand,
the status below says *unverified*, not *met*.

The library criteria A–F are the foundation of this document, not a competitor to it. They are
not restated here; where a product criterion depends on one, it names it.

---

### G. Portability — it runs where it says it runs

#### G1. No path that exists on only one machine

Shipped source may not name a filesystem location that is not part of the package, the
toolchain, or something the user configured. A hardcoded home directory is a build that
cannot be reproduced by anyone.

*Prevents:* the exact failure this whole document is about — code that is correct, tested,
and unbuildable by its consumer.

*Enforced by:* nothing yet. Measured today: `src/miniworld_engine/kernels/transition/cuda/__init__.py`
carries **six** occurrences of `-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/...` across
three build functions. That kernel cannot compile on any machine but this one. A grep guard in
the suite is one test and does not exist.

*Status:* **met.** `tests/layout/test_no_machine_paths.py` scans `src/` and `tools/` -- the two
trees whose contents are executed -- and `_nvcc.mathdx_includes` resolves at run time, naming the
variable to set when it cannot. Verified to fail on the pre-fix file, all six lines.

#### G2. Every architecture the registry claims is either exercised or declared unexercised

`registry.csv` declares `arch` for 103 kernels: 94 `sm80`, 2 `sm90`, 7 `sm100`. A declaration
is a promise to a consumer choosing hardware. A promise nothing runs against is a guess.

*Prevents:* a consumer buying or booking time on hardware the library claims to support and
discovering at run time that the claim was aspirational.

*Enforced by:* `tests/registry/test_hardware_support.py` and `tests/registry/test_arch_gating.py` check that the
declaration is internally coherent and that unsupported hardware fails clearly — not that the
kernel was ever *run* on the arch it names. In practice everything has been verified on sm86
(A5000 / A6000) by hand. sm90 and sm100 kernels have never been executed by any automated
process.

*Status:* **partially met.** Coherence enforced, execution still unverified for sm90/sm100 --
`docs/gpus/supported.md` says which. The conflation is gone: `arch` is the enforced minimum and
`tuned_for` is what a kernel was written against, so the three triton kernels that lived inside
sm100-named cute modules are now launched and checked on sm86 (`driven` 94 -> 97, `skipped`
9 -> 6).

#### G3. The toolchain range is stated and its edges are tested

`torch>=2.8`, `triton` unpinned, `requires-python >=3.10`. An open upper bound is a claim that
every future release will work.

*Prevents:* a consumer on a different torch/triton/CUDA combination hitting a failure the
author never saw, with no way to know whether they are inside or outside the supported set.

*Enforced by:* CI's `floor` job tests the Python floor (3.10) and `checks` tests 3.12. Neither
varies torch, triton, or CUDA. There is no matrix.

*Status:* **partially met** — Python edges tested, the edges that actually break kernels are not.

#### G4. A clean clone builds with no personal environment

The build must not depend on an env var, a cache, or a directory that the author happens to
have. The other-cluster failure was a clean clone that could not do what this checkout does.

*Prevents:* a working repository that is only working here.

*Enforced by:* `benchmarks/reproducing-a-report.md` gives the recipe and it is run rather than described.
Two known causes were also removed: import-time nvcc builds (`test_no_build_at_import.py`) and
stale JIT locks (`test_jit_build_lock.py`).

*Status:* **met, unenforced.** Measured: clone from the remote (not from this checkout), every
cache pointed somewhere empty (`TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `MINIWORLD_CONFIG_DIR`
unset, `PYTHONPATH` unset), build a wheel, install it to an empty `--target`, import from there,
run the suite. Wheel, install and import all clean; version 1.0.0, the packaged config set and the
device manifests all present; **1225 passed, 7 skipped** from the clone. Unenforced because
nothing repeats it: no CI job can clone and build a kernel without a GPU (J1).

#### G5. The absence of a GPU is a supported state

A consumer runs unit tests, reads docs, and imports the package on a laptop.

*Prevents:* an import or a CLI invocation that dies on a machine with no CUDA.

*Enforced by:* the whole CPU suite (1191 tests) runs on `ubuntu-latest` with no GPU, and A2
forbids work at import. This one is genuinely covered.

*Status:* **met.**

---

### H. Distribution — a stranger can obtain and install it

#### H1. The artifact a consumer installs is built and inspected, not assumed

*Prevents:* a wheel that imports on the author's machine because `src/` is on the path, and
fails everywhere else because the data files were never packaged.

*Enforced by:* not automatically — but measured today, and it passes. `pip wheel` produces
`miniworld_engine-0.1.0-py3-none-any.whl`, 563 files: 256 `.py`, 186 autotune `.json`, 98
`.csv`, 14 `.cu`, 1 `.cuh`, `py.typed`, `registry.csv`, 91 config files. Installed to an
isolated `--target` and imported without `src/` on the path, `miniworld_engine`,
`miniworld_engine.kernels` and `miniworld_engine.autotune.cache` all import.

*Status:* **met.** A `wheel` CI job builds it, asserts each shipped asset count against the
tree (so adding a kernel cannot fail it for the wrong reason), asserts the lab notebook and the
A/B config sets are absent, and imports it from a `--target` install with `src/` off the path.

#### H2. Dependencies are a contract, not a snapshot of what was installed

*Prevents:* an install that resolves to a combination nobody has run.

*Enforced by:* `pyproject.toml` separates a lean runtime core (`torch`, `triton`, `einops`,
`jaxtyping`, `numpy`) from extras, with the reasoning written down. But `triton` has no floor
at all, and the pixi lock — the only fully-pinned artifact — is not what a `pip install`
consumer gets.

*Status:* **met for the floor that exists.** `triton>=3.3` is declared with the code evidence
behind it; `einops`/`jaxtyping`/`numpy` stay unbounded deliberately, because no version of this
repo has been exercised against an older release and a guessed floor reads like evidence.
`docs/gpus/supported.md` states what was actually run.

#### H3. Installation is documented for the case where the author is not present

*Prevents:* an installation that requires asking the author.

*Enforced by:* nothing. `README.md` has `## Toolchain` and pixi commands; there is no
installation section written for someone starting from an empty machine, and no page telling
them what to do when nvcc/mathdx/CUDA is missing.

*Status:* **met.** `## Quickstart` is four steps at the top of the README, three of them
GPU-free, executed by `tests/layout/test_quickstart_runs.py`; `docs/gpus/troubleshooting.md` covers
what goes wrong, tied to the message literals in `src/`.

#### H4. The name is one name, everywhere, including in the consumer

*Prevents:* the state this was written in — the package renamed from `miniworld-kernels` to
`miniworld-engine`, `import miniworld_kernels` raising `ModuleNotFoundError`, while the one real
consumer's submodule URL, dependency entry and directory were all still the old name.

*Enforced by:* D1 covers names *inside* the repo; nothing mechanical covers the name as a consumer
spells it, and nothing here can — it is a different repository.

*Status:* **met in the code, uncommitted in the consumer.** `team-gm` now has the submodule at
`libs/miniworld-engine` pinned to the `v1.0.0` tag, `miniworld-engine` in `pyproject.toml` and
`uv.lock`, and no `import miniworld_kernels` anywhere. One occurrence of the old string remains
**on purpose**: `ImplementationType.MINIWORLD_KERNELS = "miniworld_kernels"` is a config value that
four YAML files select by name, so renaming it is a config break needing its own deprecation — the
lesson of I4, applied rather than repeated.

The change is not committed there, and that is not mine to do: 11 of the 13 migrated files carry
the user's uncommitted work.

---

### I. Release — a version number means something

#### I1. The version moves when the package does

*Prevents:* two mutually incompatible packages answering to the same version string. Measured
today: version has been `0.1.0` across **191 commits and a package rename**. The consumer's
pinned checkout says `name = "miniworld-kernels", version = "0.1.0"`; main says
`name = "miniworld-engine", version = "0.1.0"`. A consumer cannot distinguish them by any
declared field.

*Enforced by:* nothing. No release has ever been tagged (`git tag` holds two `archive/*` tags
and no version).

*Status:* **met.** 1.0.0, tagged, with a `### Breaking` entry naming the rename.
`tests/registry/test_version_is_released.py` fails a bump with no changelog section, a changelog
whose only section is `[Unreleased]`, and an x.0.0 with no Breaking entry.

#### I2. Every release is a tag, and every tag is reachable

*Prevents:* "which commit was running when we measured that?" being unanswerable.

*Enforced by:* nothing.

*Status:* **met.** `v1.0.0` is an annotated tag on the released commit, and
`autotune/manifests/` records which commit the GPU evidence was produced at.

#### I3. The changelog describes released things

`docs/CHANGELOG.md` exists, is 264 lines, is written well, and is **entirely** under
`## [Unreleased]`. It documents a public-API contract enforced by `tests/compile/test_public_api.py`
(A1) — for a package that has never published a version.

*Prevents:* a consumer being unable to learn what changed between the version they have and
the version they want.

*Status:* **met.** `## [1.0.0] - 2026-08-25` with `[Unreleased]` above it, and the version and
the changelog cannot disagree without failing a test.

#### I4. A breaking change is announced before it lands, not discovered

The `miniworld-kernels` → `miniworld-engine` rename is a breaking change to every import in
every consumer. It shipped with no major-version bump, no deprecation shim, and no compat
alias.

*Prevents:* an upgrade that cannot be attempted incrementally.

*Enforced by:* A4 documents a removal path for *API names*. It says nothing about the
distribution or import name.

*Status:* **met.** The rename is a `### Breaking` entry saying what to change and why the
version is 1.0.0 rather than 0.2.0.

---

### J. Verification at a distance — CI proves what the README claims

#### J1. The GPU claims are backed by dated evidence, not by memory

Three CI jobs, all `runs-on: ubuntu-latest`. 1230 CPU tests run on every push; the **116
gpu-marked tests run zero times**, and the one step that mentions the GPU is `--collect-only`,
which proves they can be collected. Every claim about kernel correctness comes from a person
running `run_all` on this cluster.

A self-hosted GPU runner would close that and is **deliberately excluded** — see the section at
the end. So the criterion is not "CI executes them", which is unreachable here; it is that the
evidence exists, says when and against what it was produced, and that its absence is loud at the
moment it matters.

*Prevents:* a release going out that nothing has ever run on a card, and — the subtler one — a
manifest from six months and two rewrites ago being read as current.

*Enforced by:* `run_all` writes `autotune/manifests/<card>.csv` with a `#provenance` row (version,
commit, clean/dirty, date); `docs/gpus/supported.md` cites those manifests;
`tests/registry/test_a_release_has_been_run_on_a_card.py` fails a release whose version appears in
no manifest, or only in one produced from a dirty tree.

*Status:* **met, scoped.** And the scope has a cost that must not be misread: **a green CI does
not mean the kernels are verified.** It means nothing about them. Today's tolerance tightening
(95 bands, 5x narrower) and arch relaxation (3 kernels ungated) were both checked by hand on an
A6000; CI was green before and after either, and would have been green if either had been wrong.

#### J2. The shipped autotune cache is validated, not trusted

186 JSON files ship inside the wheel and directly determine which kernel config runs.

*Prevents:* a merged edit that corrupts or orphans tuned entries. Real today: the 512² bucket
change altered bucket indices; the check that nothing was orphaned was a one-off script run by
hand, not a test.

*Enforced by:* `tests/autotune/test_shipped_cache_wellformed.py` checks shape. `dev audit` checks
the build system and declared-vs-present coverage, and now RUNS in CI -- it could not before,
because it exited 1 on every default invocation with 139 findings about its own missing arguments
(88 reachability with no `--shards`, 51 coverage against the key `cpu`). `dev audit --replay`
answers the different question -- what a run of the module matrix asks for and does not get -- and
had no caller anywhere in the repo until now, while `cache.py` named it as "the only direct
measure of whether the cache covers a workload".

*Status:* **partially met, and the gap is now measured rather than assumed.** Two numbers, both
true, on the same shipped cache on an A6000:

- declared coverage: **91 OK, missing_pairs 0** -- every (op, dtype, shape bucket) `op_units`
  enumerates is present.
- the replay: **363 lookups the module matrix asks for and the cache does not serve, across 42 of
  91 ops** (`archive/records-20260928:docs/records/cache-coverage-replay-a6000.md`).

The cache key carries each kernel's constexprs and no declared work list enumerates them, so the
first number cannot see the second. `build all` with no flags now runs both work lists rather than
one, which is what closes the 363 -- for a cache that is rebuilt. This one is not: the 349 entries
the bench budget poisoned are still in it, and so are the 363 holes.

#### J3. A performance claim is re-measured, or it is dated

*Prevents:* a README number that was true on one card, one torch version, and one config set,
and is quoted forever.

*Enforced by:* `tests/numerics/test_performance_claims.py` traces prose numbers to artifacts (C1/C2).
It does not re-run them, and cannot without J1.

*Status:* **partially met** — provenance enforced, freshness not.

#### J4. The gates that exist run on every change

*Prevents:* a lint or type rule that is configured and never executed.

*Enforced by:* `.github/workflows/ci.yml` runs ruff, ty, the CPU suite and `dev audit` on push,
and the `pixi run ci` task mirrors it in the same order. Verified green today: **1241 passed, 11
skipped, 116 deselected**, and **1241 passed, 127 skipped** with no marker flag -- the second
number used to be eleven failures reading "Found no NVIDIA driver", because nothing ever ran the
`gpu` marker's own meaning.

The audit step is new and is what J2 records: 264 findings over 88 live autotuners that ran
nowhere automatic, and could not, because the command exited 1 on every invocation.

*Status:* **met**, for the CPU half.

---

### K. The consumer — a real integration, proven

#### K1. There is a consumer, it is current, and upgrading it is a routine act

*Prevents:* a library that improves in a direction nobody can follow. When this was written,
`team-gm` pinned `403d382` of 2026-07-27 — **191 commits and 29 days behind** — by the pre-rename
name, so advancing it broke every import. Every fix in those commits was invisible to the only
thing that uses this library.

*Enforced by:* nothing mechanical; a second repository cannot be gated from here. What replaced
"nobody has tried" is that it has now been done and what it costs is known.

*Status:* **done, uncommitted.** Pin advanced 191 commits to the `v1.0.0` TAG rather than a bare
SHA. The upgrade also surfaced the thing that made it non-routine, which was not the rename:
team-gm's environment held **torch 2.6.0** against this package's declared `torch>=2.8`, and six
modules failed with `infer_schema(func): Parameter input_shape has unsupported type list[int]`.
The floor caught a real incompatibility, which is what a floor is for. The environment is now on
`torch 2.11.0+cu128`, the same CUDA and triton line this package is developed against.

What is left is a commit in a repository whose working tree is not mine to commit.

#### K2. An end-to-end test proves the kernels are substitutable

The suite has 47 test files. Every one is a unit or contract test. **None** runs a model with
these kernels and compares its output to the same model without them.

*Prevents:* per-kernel correctness that does not add up to a correct model — a tolerance that
is fine in isolation and compounds across 48 layers, a dtype that silently promotes, a kernel
that is right on its own inputs and wrong on the ones the model actually produces.

*Enforced by:* `tests/numerics/test_stack_substitutability_gpu.py`. One Pairformer, built twice
from the same weights, `PYTORCH` against `MINIWORLD`: 1.30e-02 over four blocks against a 6e-02
budget. Three separate tests keep it from being vacuous — the stack must not be the identity,
dispatch must not have resolved to the reference, and a projection replaced with noise must break
the comparison (it moves it to 1.09e-01).

*Status:* **met.**

#### K3. The speedup is measured where the consumer will feel it

Per-kernel benchmarks exist in quantity and are well governed (C1–C3). A consumer does not buy
kernel microseconds; it buys step time.

*Prevents:* a 3× kernel that moves a training step by 2%.

*Enforced by:* the module-level bench harness exists (`benchmarks/modules/`), but no artifact
ties a module or step-level number to a released version.

*Status:* **partially met.**

#### K4. The consumer's failure is reproducible here

*Prevents:* today's ten-hour A100 loss, which took a session of analysis to attribute because
the failing environment could not be reproduced on this cluster.

*Enforced by:* `benchmarks/reproducing-a-report.md` -- isolate the four caches that carry state between
runs, ask the CPU-only question first, take a card last. Demonstrated on the report that motivated
it: `0854ac4^` gives 527 units, main gives 859, no GPU involved. The count is now pinned by
`tests/builder/test_build_matrix.py`.

*Status:* **met.**

---

### L. Documentation for someone who is not the author

#### L1. A stranger can get a first result from the README alone

*Prevents:* a library whose entry cost is a conversation with its author.

*Enforced by:* `tests/layout/test_quickstart_runs.py` executes the quickstart's `# cpu` blocks
as scripts and fails when one does not work, so the first page a newcomer runs cannot drift from
what the code does.

*Status:* **met.**

#### L2. Working notes are separated from consumer documentation

`docs/` held 124 tracked files, **101** of them per-round optimization logs, with their
profiler captures split into a third tree (`profiles/`). Now separated and then placed: `docs/`
is 26 pages written for a consumer, and each family's log lives with the kernel it is about, at
`src/miniworld_engine/kernels/<family>/notes/`.

*Prevents:* a consumer unable to find the four pages that concern them among a hundred that do
not.

*Enforced by:* `tests/layout/test_kernel_layout.py` allows `notes/` beside a family's backends and
forbids it being a package; `tests/layout/test_notes_stay_out_of_the_wheel.py` keeps it out of the
artifact. `kernels/NOTES.md` states what the tree is and that it is not maintained.

*Status:* **met, unenforced.**

#### L3. Every failure mode a consumer can hit has a page

*Prevents:* a stale JIT lock, a missing mathdx include, a cache miss, or an unsupported arch
each costing a consumer a day.

*Enforced by:* E4 requires error messages to name the fix, and `docs/gpus/troubleshooting.md` gives
each failure a section: what produces it and the command that ends it.
`tests/layout/test_troubleshooting_quotes_real_messages.py` fails when a quoted message stops
existing in `src/`, so a reworded message cannot leave a section describing something that no
longer happens.

*Status:* **met.**

#### L4. The supported set is a document, not a paragraph

*Prevents:* ambiguity about which card, driver, torch, and CUDA are inside the promise.

*Enforced by:* `docs/gpus/supported.md`, where every row cites the artifact behind it -- a device
manifest in `autotune/manifests/` or a CI job -- and the nine kernels declared for sm90/sm100 are
listed under "GPU that has NOT been run".

*Status:* **met**, in the only sense available: the page states what ran and marks the rest
untested, rather than implying a matrix that does not exist.

---

### M. Lifecycle — the project outlives one person's attention

#### M1. Work in progress is visible to anyone who looks

*Prevents:* 70 commits accumulating locally while a second machine runs a month-old tree —
directly, the ten hours lost today.

*Enforced by:* `.githooks/post-commit` reports what the remote does not have -- count, age of the
oldest, first five subjects -- after every commit. Not a gate: blocking a commit for being
unpushed is nonsense and blocking a push is backwards. It makes the invisible state visible at the
moment you would otherwise stop looking, which is what `git status` does not do.

*Status:* **met.** Verified against a constructed remote rather than by reading the hook.

#### M2. A second person can make a change

*Prevents:* a bus factor of one.

*Enforced by:* `docs/kernels/contributing.md` exists and F6 is met for the mechanics — clone, gates, how to
run the suite. Untested by any second person.

*Status:* **partially met.** `docs/kernels/contributing.md` covers the mechanics and the quickstart is now
executed rather than described. Still untested by any second person.

#### M3. Nothing is retained that nobody can explain

*Prevents:* archives, branches, and result directories accumulating until nobody dares delete
them.

*Enforced by:* F3 (no orphan code). Done today, by hand: 7 stale worktrees removed, 3 merged
branches deleted, 1 redundant `archive/` tag deleted, 14 retired-name benchmark directories
moved to `/public_data02/psk6950/mwe-attic/` with provenance. Remote is now `main` + `mpnn` +
2 archive tags.

*Status:* **met today, unenforced.**

---

### What is deliberately *not* a criterion

**A public release.** Nothing here requires PyPI, a docs site, or external users. The standard
is that a *named* consumer can install, upgrade, and verify — that consumer is `team-gm`.

**A full hardware matrix in CI.** sm100 CI is not a reasonable ask. The criterion (G2, J1) is
that the claim matches the evidence: run what can be run, and mark the rest unverified rather
than implying it was tested.

**A self-hosted GPU runner.** It would let CI execute the 116 gpu-marked tests, and it is out of
scope by decision: it needs a registered runner token, a daemon resident on a cluster GPU node,
and that node's capacity held for CI rather than for work. The consequence is accepted and stated
rather than worked around -- J1 is met by dated evidence, and a green CI says nothing about the
kernels. Anything that reintroduces "CI is the gate for kernel correctness" is reintroducing a
claim this repository cannot support.

**Backwards compatibility with the pre-rename package.** The rename was correct. What is
required (I4) is that the break be versioned and announced, not that it be undone.

**Documentation of internals.** L2 asks that the lab notebook be separated from consumer docs,
not shrunk. The optimization logs stay.

---

### The single acceptance test

Every criterion above is a component of one sentence:

> On a machine that is not this one, a person who is not the author checks out a tagged
> version, installs it, runs the suite, upgrades `team-gm` to that tag, and gets the same
> numerical result and a measured step-time improvement — using only what is written down.

Today that sentence fails at the first clause. `archive/records-20260928:docs/records/product-plan.md` is the ordered work to make it true.

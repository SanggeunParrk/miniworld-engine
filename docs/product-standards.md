# What a product owes someone who is not its author

`docs/library-standards.md` asked whether this code is a good library. Every one of its 30
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

**The check must fail for the right reason.** Also inherited. A green suite that stopped
looking is worse than a red one.

**The author is not the judge.** A criterion is met when a machine that is not this one, or a
person who is not the author, demonstrates it. "It works for me" is the null hypothesis this
document exists to reject. Where the only evidence is the author running something by hand,
the status below says *unverified*, not *met*.

The library criteria A–F are the foundation of this document, not a competitor to it. They are
not restated here; where a product criterion depends on one, it names it.

---

## G. Portability — it runs where it says it runs

### G1. No path that exists on only one machine

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

### G2. Every architecture the registry claims is either exercised or declared unexercised

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
`docs/supported.md` says which. The conflation is gone: `arch` is the enforced minimum and
`tuned_for` is what a kernel was written against, so the three triton kernels that lived inside
sm100-named cute modules are now launched and checked on sm86 (`driven` 94 -> 97, `skipped`
9 -> 6).

### G3. The toolchain range is stated and its edges are tested

`torch>=2.8`, `triton` unpinned, `requires-python >=3.10`. An open upper bound is a claim that
every future release will work.

*Prevents:* a consumer on a different torch/triton/CUDA combination hitting a failure the
author never saw, with no way to know whether they are inside or outside the supported set.

*Enforced by:* CI's `floor` job tests the Python floor (3.10) and `checks` tests 3.12. Neither
varies torch, triton, or CUDA. There is no matrix.

*Status:* **partially met** — Python edges tested, the edges that actually break kernels are not.

### G4. A clean clone builds with no personal environment

The build must not depend on an env var, a cache, or a directory that the author happens to
have. The other-cluster failure was a clean clone that could not do what this checkout does.

*Prevents:* a working repository that is only working here.

*Enforced by:* `docs/reproducing-a-report.md` gives the recipe and it is run rather than described.
Two known causes were also removed: import-time nvcc builds (`test_no_build_at_import.py`) and
stale JIT locks (`test_jit_build_lock.py`).

*Status:* **met, unenforced.** Measured: clone from the remote (not from this checkout), every
cache pointed somewhere empty (`TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `MINIWORLD_CONFIG_DIR`
unset, `PYTHONPATH` unset), build a wheel, install it to an empty `--target`, import from there,
run the suite. Wheel, install and import all clean; version 1.0.0, the packaged config set and the
device manifests all present; **1225 passed, 7 skipped** from the clone. Unenforced because
nothing repeats it: no CI job can clone and build a kernel without a GPU (J1).

### G5. The absence of a GPU is a supported state

A consumer runs unit tests, reads docs, and imports the package on a laptop.

*Prevents:* an import or a CLI invocation that dies on a machine with no CUDA.

*Enforced by:* the whole CPU suite (1191 tests) runs on `ubuntu-latest` with no GPU, and A2
forbids work at import. This one is genuinely covered.

*Status:* **met.**

---

## H. Distribution — a stranger can obtain and install it

### H1. The artifact a consumer installs is built and inspected, not assumed

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

### H2. Dependencies are a contract, not a snapshot of what was installed

*Prevents:* an install that resolves to a combination nobody has run.

*Enforced by:* `pyproject.toml` separates a lean runtime core (`torch`, `triton`, `einops`,
`jaxtyping`, `numpy`) from extras, with the reasoning written down. But `triton` has no floor
at all, and the pixi lock — the only fully-pinned artifact — is not what a `pip install`
consumer gets.

*Status:* **met for the floor that exists.** `triton>=3.3` is declared with the code evidence
behind it; `einops`/`jaxtyping`/`numpy` stay unbounded deliberately, because no version of this
repo has been exercised against an older release and a guessed floor reads like evidence.
`docs/supported.md` states what was actually run.

### H3. Installation is documented for the case where the author is not present

*Prevents:* an installation that requires asking the author.

*Enforced by:* nothing. `README.md` has `## Toolchain` and pixi commands; there is no
installation section written for someone starting from an empty machine, and no page telling
them what to do when nvcc/mathdx/CUDA is missing.

*Status:* **met.** `## Quickstart` is four steps at the top of the README, three of them
GPU-free, executed by `tests/layout/test_quickstart_runs.py`; `docs/troubleshooting.md` covers
what goes wrong, tied to the message literals in `src/`.

### H4. The name is one name, everywhere, including in the consumer

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

## I. Release — a version number means something

### I1. The version moves when the package does

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

### I2. Every release is a tag, and every tag is reachable

*Prevents:* "which commit was running when we measured that?" being unanswerable.

*Enforced by:* nothing.

*Status:* **met.** `v1.0.0` is an annotated tag on the released commit, and
`autotune/manifests/` records which commit the GPU evidence was produced at.

### I3. The changelog describes released things

`CHANGELOG.md` exists, is 264 lines, is written well, and is **entirely** under
`## [Unreleased]`. It documents a public-API contract enforced by `tests/compile/test_public_api.py`
(A1) — for a package that has never published a version.

*Prevents:* a consumer being unable to learn what changed between the version they have and
the version they want.

*Status:* **met.** `## [1.0.0] - 2026-08-25` with `[Unreleased]` above it, and the version and
the changelog cannot disagree without failing a test.

### I4. A breaking change is announced before it lands, not discovered

The `miniworld-kernels` → `miniworld-engine` rename is a breaking change to every import in
every consumer. It shipped with no major-version bump, no deprecation shim, and no compat
alias.

*Prevents:* an upgrade that cannot be attempted incrementally.

*Enforced by:* A4 documents a removal path for *API names*. It says nothing about the
distribution or import name.

*Status:* **met.** The rename is a `### Breaking` entry saying what to change and why the
version is 1.0.0 rather than 0.2.0.

---

## J. Verification at a distance — CI proves what the README claims

### J1. The GPU claims are backed by dated evidence, not by memory

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
commit, clean/dirty, date); `docs/supported.md` cites those manifests;
`tests/registry/test_a_release_has_been_run_on_a_card.py` fails a release whose version appears in
no manifest, or only in one produced from a dirty tree.

*Status:* **met, scoped.** And the scope has a cost that must not be misread: **a green CI does
not mean the kernels are verified.** It means nothing about them. Today's tolerance tightening
(95 bands, 5x narrower) and arch relaxation (3 kernels ungated) were both checked by hand on an
A6000; CI was green before and after either, and would have been green if either had been wrong.

### J2. The shipped autotune cache is validated, not trusted

186 JSON files ship inside the wheel and directly determine which kernel config runs.

*Prevents:* a merged edit that corrupts or orphans tuned entries. Real today: the 512² bucket
change altered bucket indices; the check that nothing was orphaned was a one-off script run by
hand, not a test.

*Enforced by:* `tests/autotune/test_shipped_cache_wellformed.py` checks shape; `dev audit` checks
declared-vs-present reachability, but is a CLI command nobody runs automatically.

*Status:* **partially met.** `tests/autotune/test_shipped_cache_wellformed.py` checks shape and
`dev audit` checks declared-vs-present reachability, but the audit is a command nobody runs
automatically. The 349 entries the bench budget poisoned are still in the shipped cache.

### J3. A performance claim is re-measured, or it is dated

*Prevents:* a README number that was true on one card, one torch version, and one config set,
and is quoted forever.

*Enforced by:* `tests/numerics/test_performance_claims.py` traces prose numbers to artifacts (C1/C2).
It does not re-run them, and cannot without J1.

*Status:* **partially met** — provenance enforced, freshness not.

### J4. The gates that exist run on every change

*Prevents:* a lint or type rule that is configured and never executed.

*Enforced by:* `.github/workflows/ci.yml` runs ruff, ty, and the CPU suite on push, and the
`pixi run ci` task mirrors it in the same order. Verified green today: **1191 passed, 7
skipped, 111 deselected**.

*Status:* **met**, for the CPU half.

---

## K. The consumer — a real integration, proven

### K1. There is a consumer, it is current, and upgrading it is a routine act

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

### K2. An end-to-end test proves the kernels are substitutable

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

### K3. The speedup is measured where the consumer will feel it

Per-kernel benchmarks exist in quantity and are well governed (C1–C3). A consumer does not buy
kernel microseconds; it buys step time.

*Prevents:* a 3× kernel that moves a training step by 2%.

*Enforced by:* the module-level bench harness exists (`benchmarks/modules/`), but no artifact
ties a module or step-level number to a released version.

*Status:* **partially met.**

### K4. The consumer's failure is reproducible here

*Prevents:* today's ten-hour A100 loss, which took a session of analysis to attribute because
the failing environment could not be reproduced on this cluster.

*Enforced by:* `docs/reproducing-a-report.md` -- isolate the four caches that carry state between
runs, ask the CPU-only question first, take a card last. Demonstrated on the report that motivated
it: `0854ac4^` gives 527 units, main gives 859, no GPU involved. The count is now pinned by
`tests/builder/test_build_matrix.py`.

*Status:* **met.**

---

## L. Documentation for someone who is not the author

### L1. A stranger can get a first result from the README alone

*Prevents:* a library whose entry cost is a conversation with its author.

*Enforced by:* `tests/layout/test_quickstart_runs.py` executes the quickstart's `# cpu` blocks
as scripts and fails when one does not work, so the first page a newcomer runs cannot drift from
what the code does.

*Status:* **met.**

### L2. Working notes are separated from consumer documentation

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

### L3. Every failure mode a consumer can hit has a page

*Prevents:* a stale JIT lock, a missing mathdx include, a cache miss, or an unsupported arch
each costing a consumer a day.

*Enforced by:* E4 requires error messages to name the fix, and `docs/troubleshooting.md` gives
each failure a section: what produces it and the command that ends it.
`tests/layout/test_troubleshooting_quotes_real_messages.py` fails when a quoted message stops
existing in `src/`, so a reworded message cannot leave a section describing something that no
longer happens.

*Status:* **met.**

### L4. The supported set is a document, not a paragraph

*Prevents:* ambiguity about which card, driver, torch, and CUDA are inside the promise.

*Enforced by:* `docs/supported.md`, where every row cites the artifact behind it -- a device
manifest in `autotune/manifests/` or a CI job -- and the nine kernels declared for sm90/sm100 are
listed under "GPU that has NOT been run".

*Status:* **met**, in the only sense available: the page states what ran and marks the rest
untested, rather than implying a matrix that does not exist.

---

## M. Lifecycle — the project outlives one person's attention

### M1. Work in progress is visible to anyone who looks

*Prevents:* 70 commits accumulating locally while a second machine runs a month-old tree —
directly, the ten hours lost today.

*Enforced by:* `.githooks/post-commit` reports what the remote does not have -- count, age of the
oldest, first five subjects -- after every commit. Not a gate: blocking a commit for being
unpushed is nonsense and blocking a push is backwards. It makes the invisible state visible at the
moment you would otherwise stop looking, which is what `git status` does not do.

*Status:* **met.** Verified against a constructed remote rather than by reading the hook.

### M2. A second person can make a change

*Prevents:* a bus factor of one.

*Enforced by:* `CONTRIBUTING.md` exists and F6 is met for the mechanics — clone, gates, how to
run the suite. Untested by any second person.

*Status:* **partially met.** `CONTRIBUTING.md` covers the mechanics and the quickstart is now
executed rather than described. Still untested by any second person.

### M3. Nothing is retained that nobody can explain

*Prevents:* archives, branches, and result directories accumulating until nobody dares delete
them.

*Enforced by:* F3 (no orphan code). Done today, by hand: 7 stale worktrees removed, 3 merged
branches deleted, 1 redundant `archive/` tag deleted, 14 retired-name benchmark directories
moved to `/public_data02/psk6950/mwe-attic/` with provenance. Remote is now `main` + `mpnn` +
2 archive tags.

*Status:* **met today, unenforced.**

---

## What is deliberately *not* a criterion

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

## The single acceptance test

Every criterion above is a component of one sentence:

> On a machine that is not this one, a person who is not the author checks out a tagged
> version, installs it, runs the suite, upgrades `team-gm` to that tag, and gets the same
> numerical result and a measured step-time improvement — using only what is written down.

Today that sentence fails at the first clause. `plan.md` is the ordered work to make it true.

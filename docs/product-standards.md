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

*Status:* **not met**, with a named defect.

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

*Status:* **partially met.** Coherence enforced, execution unverified. Compounded by the
`arch` column conflating an enforced gate (cute/cuda) with "written for" (triton) — that
conflation currently skips 3 sm100 *triton* kernels that would have run on sm86.

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

*Enforced by:* nothing. No CI job clones fresh and builds a kernel, because no CI job has a GPU
(J1). Today's fixes — lazy nvcc (`test_no_build_at_import.py`) and stale-lock recovery
(`test_jit_build_lock.py`) — remove two known causes but do not prove the general case.

*Status:* **not met.**

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

*Status:* **not met.**

### H4. The name is one name, everywhere, including in the consumer

*Prevents:* precisely the state measured today (K1): the package renamed from
`miniworld-kernels` to `miniworld-engine`, `import miniworld_kernels` now raising
`ModuleNotFoundError`, while the one real consumer's submodule URL, dependency entry, and
directory are all still the old name.

*Enforced by:* D1 covers names *inside* the repo. Nothing covers the name as the consumer
spells it.

*Status:* **not met.**

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

*Status:* **not met.** This is the most severe finding in the document.

### I2. Every release is a tag, and every tag is reachable

*Prevents:* "which commit was running when we measured that?" being unanswerable.

*Enforced by:* nothing.

*Status:* **not met.**

### I3. The changelog describes released things

`CHANGELOG.md` exists, is 264 lines, is written well, and is **entirely** under
`## [Unreleased]`. It documents a public-API contract enforced by `tests/compile/test_public_api.py`
(A1) — for a package that has never published a version.

*Prevents:* a consumer being unable to learn what changed between the version they have and
the version they want.

*Status:* **partially met** — the discipline exists, the release boundary does not.

### I4. A breaking change is announced before it lands, not discovered

The `miniworld-kernels` → `miniworld-engine` rename is a breaking change to every import in
every consumer. It shipped with no major-version bump, no deprecation shim, and no compat
alias.

*Prevents:* an upgrade that cannot be attempted incrementally.

*Enforced by:* A4 documents a removal path for *API names*. It says nothing about the
distribution or import name.

*Status:* **not met.**

---

## J. Verification at a distance — CI proves what the README claims

### J1. The claims that need a GPU are checked by something that has one

Two CI jobs, `checks` and `floor`, both `runs-on: ubuntu-latest`. The GPU step in `checks`
runs `pytest --collect-only -m gpu` — it verifies that GPU tests can be *collected*, not that
any of them pass. **103 kernels, 0 executed in CI.**

*Prevents:* the state this repo is in, where every correctness and performance claim rests on
the author having run something by hand on one of two Ampere cards.

*Enforced by:* nothing. `run_all` (`ok 94, failed 0, skipped 9`), the numerical suite
(`98 passed, 2 skipped`), and opcheck (`5 passed`) are all real results and all manual.

*Status:* **not met.** Everything else in section J is downstream of this.

### J2. The shipped autotune cache is validated, not trusted

186 JSON files ship inside the wheel and directly determine which kernel config runs.

*Prevents:* a merged edit that corrupts or orphans tuned entries. Real today: the 512² bucket
change altered bucket indices; the check that nothing was orphaned was a one-off script run by
hand, not a test.

*Enforced by:* `tests/autotune/test_shipped_cache_wellformed.py` checks shape; `dev audit` checks
declared-vs-present reachability, but is a CLI command nobody runs automatically.

*Status:* **partially met.**

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

Measured today. `team-gm` consumes this library as the submodule `libs/miniworld-kernels`,
pinned at `403d382`, dated **2026-07-27** — **191 commits and 29 days behind main**. Its
`pyproject.toml` depends on `miniworld-kernels` by path. The pin predates the rename, so
advancing it breaks every import in the consumer, which is presumably why it has never been
advanced.

*Prevents:* a library that improves in a direction nobody can follow. Every fix landed in
those 191 commits — the fp32 dtype fix, the JIT lock recovery, the lazy nvcc build, the
per-target bench configs — is invisible to the only thing that uses this library.

*Enforced by:* nothing. Nothing anywhere checks that the consumer's pin is advanceable.

*Status:* **not met.** With I1, the pair of findings that matter most.

### K2. An end-to-end test proves the kernels are substitutable

The suite has 47 test files. Every one is a unit or contract test. **None** runs a model with
these kernels and compares its output to the same model without them.

*Prevents:* per-kernel correctness that does not add up to a correct model — a tolerance that
is fine in isolation and compounds across 48 layers, a dtype that silently promotes, a kernel
that is right on its own inputs and wrong on the ones the model actually produces.

*Enforced by:* nothing.

*Status:* **not met.**

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

*Enforced by:* nothing.

*Status:* **not met.**

---

## L. Documentation for someone who is not the author

### L1. A stranger can get a first result from the README alone

*Prevents:* a library whose entry cost is a conversation with its author.

*Enforced by:* nothing. `README.md` has `Critical Safety`, `Layout`, `Benchmarking`,
`torch.compile`, `Supported hardware`, `Status`, `CLI`, `Toolchain` — organised around the
repository, not around a newcomer's first hour.

*Status:* **not met.**

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

*Enforced by:* E4 requires error messages to name the fix and today's lock error does exactly
that. There is no troubleshooting document.

*Status:* **partially met.**

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

*Enforced by:* nothing mechanical. The working rule adopted today is to push at the end of
each unit of work.

*Status:* **not met**, cause understood.

### M2. A second person can make a change

*Prevents:* a bus factor of one.

*Enforced by:* `CONTRIBUTING.md` exists and F6 is met for the mechanics — clone, gates, how to
run the suite. Untested by any second person.

*Status:* **partially met.**

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

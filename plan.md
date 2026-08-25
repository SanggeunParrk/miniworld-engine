# plan.md — the work to become a product

Derived from `docs/product-standards.md`. Every item states the criterion it answers, **the
gap as measured** (not as guessed), the action, and what makes it done. An item is done when
the check fails before the change and passes after it; if there is no check, the item's first
job is to build one.

Rewritten **2026-08-25**. The previous plan.md worked the library standard (`docs/library-standards.md`,
criteria A–F) and is 17/20 closed. That work was necessary and is not repeated here — it is
summarised in **§0** and its three open items are carried forward as **C1–C3** because product
criteria depend on them.

Ordering is by unblocking power, not by size. **A1 → A3** are the two findings that make
everything else moot, and the cheap fix that unblocks other machines.

---

## §0 — closed (library standard A–F)

| item | criterion | what it fixed |
|---|---|---|
| P0a | D1 | plot-style entries orphaned by the label rename |
| P0b | E1 | `build all` proven end to end on device |
| P1 | A3 | `py.typed` shipped (PEP 561) |
| P2a | B2 | `rtol` column added to `registry.csv`; NaN hole in `run_all` closed |
| P4 | D5 | `configs` shadowing landmine |
| P5 | F3 | orphan pilot builder deleted |
| P6 | A4 | deprecation policy with a mechanism |
| P7 | A5 | hardware-support matrix, checked |
| P9 | C2 | quoted numbers traceable to artifacts |
| P10 | D4 | `configs/grid` duplication ended |
| P11 | F4 | stale reference docs |
| P12 | F5 | `todo.md` removed from the repo |
| P13 | F6 | `CONTRIBUTING.md` |
| P14 | A4 | Python floor (3.10) actually tested in CI |
| P15 | E2 | stale JIT build lock no longer hangs forever |
| P16 | B1, F2 | vendored-body lint exemption hid a guaranteed `NameError` |
| P17 | E2 | `run_all` reported "wrong card" as "wrong answer" |
| P18 | B1 | opcheck asserted a contract the design declines |

---

## A. The two findings that make the rest moot

### A1 — the version number does not distinguish two incompatible packages  (I1, I2, I4)

*Gap, measured:* `version = "0.1.0"` has not moved across **191 commits including a package
rename**. `git tag` holds `archive/gate-fuse-v1` and `archive/ln-bwd-cuda-v1` and no version
tag. The consumer's pinned tree declares `name = "miniworld-kernels", version = "0.1.0"`; main
declares `name = "miniworld-engine", version = "0.1.0"`. `import miniworld_kernels` →
`ModuleNotFoundError`. Nothing in either package's metadata lets a consumer tell them apart.
`CHANGELOG.md` is 264 good lines, all under `[Unreleased]`.

*Action:* the rename is a breaking change, so the next version is **1.0.0**, not 0.2.0 — a
0.x bump would understate it. Move `[Unreleased]` to `## [1.0.0] - 2026-08-25` with an
explicit **Breaking** section naming `miniworld-kernels` → `miniworld-engine` and the import
path change. Tag `v1.0.0`. Add `tests/test_version_is_released.py`: the version in
`pyproject.toml` must have a matching `## [<version>]` heading in `CHANGELOG.md`, and
`[Unreleased]` must not be the only section.

*Done when:* the test fails on a version bump without a changelog entry, `git tag -l 'v*'`
lists `v1.0.0`, and `pip download miniworld-engine==1.0.0` from the tag resolves.

### A2 — the only consumer is 191 commits behind and cannot advance  (K1, H4)

*Gap, measured:* `team-gm` pins `libs/miniworld-kernels` at `403d382` (2026-07-27), **191
commits / 29 days** behind. Its `.gitmodules` URL, its `pyproject.toml` dependency name, and
its directory are all `miniworld-kernels`. Advancing the pin breaks every import. Every fix in
those 191 commits — the fp32 dtype fix that turns 527 units into 859, the JIT-lock recovery,
the lazy nvcc build, the per-target bench configs — is invisible to the one thing that uses
this library.

*Action:* land A1 first so the target is a tag, not a moving branch. Then, in `team-gm`:
rename the submodule path and URL, update the dependency name, and rewrite the imports. Do it
as one commit on a branch off `exp/miniworld-integrated`, gated on `team-gm`'s own suite.

*Done when:* `team-gm` builds and its tests pass against `miniworld-engine v1.0.0`, and the
submodule pin is a tag rather than a bare SHA.

### A3 — six hardcoded personal paths make one kernel unbuildable elsewhere  (G1)

*Gap, measured:* `src/miniworld_engine/kernels/transition/cuda/__init__.py` lines 22–23,
48–49, 74–75 — `-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/include` and
`.../external/cutlass/include`, in three separate build functions.

*Action:* resolve mathdx from, in order: an explicit setting, `MATHDX_HOME`/`NVIDIA_MATHDX_HOME`,
the installed `nvidia-mathdx` package, then a documented failure that names the variable to
set (E4). Add `tests/test_no_machine_paths.py`: no shipped source under `src/` may contain
`/home/<user>` or another absolute path outside the package or toolchain.

*Done when:* the new test fails on today's tree and passes after, and the kernel builds on a
machine with mathdx installed anywhere else.

---

## B. Prove it somewhere that is not this machine

### B1 — no GPU runs in CI  (J1)

*Gap, measured:* both CI jobs are `runs-on: ubuntu-latest`. The "gpu" step runs
`pytest --collect-only -m gpu` — it proves GPU tests can be *collected*. **103 kernels, 0
executed by CI.** `run_all` (`ok 94, failed 0, skipped 9`), the numerical suite (`98 passed,
2 skipped`) and opcheck (`5 passed`) are real and entirely manual.

*Action:* a self-hosted runner is not available, so make the gate Slurm-shaped instead: a
`scripts/ci-gpu.sbatch` that runs `pytest -m gpu`, `run_all`, and opcheck, writes a JSON
verdict with the commit SHA and the device, and a `tests/test_gpu_verdict_is_current.py` that
fails when the newest verdict does not match `HEAD` or is older than N days. That converts
"the author ran it" into a dated, checkable artifact.

*Done when:* the verdict file is a required part of a release (A1) and the test fails on a
stale one.

### B2 — no end-to-end test proves the kernels are substitutable  (K2)

*Gap, measured:* 47 test files, all unit or contract. None runs a model with these kernels and
compares against the same model without them.

*Action:* one test that builds a small `team-gm` block twice — reference ops and
miniworld-engine ops — feeds identical seeded input, and asserts agreement at the *declared*
per-kernel tolerances (B2/P2a) compounded across depth. Mark `gpu`; it belongs in the B1
verdict.

*Done when:* it fails if any single kernel is swapped for a deliberately wrong one.

### B3 — the wheel is never built by anything automatic  (H1)

*Gap, measured:* verified by hand today and it passes — 563 files, `py.typed`, 186 autotune
JSONs, `registry.csv`, 91 configs, imports clean from an isolated `--target` install. Nothing
re-checks it, so the next `package-data` edit can silently drop the shipped cache.

*Action:* a CI job that builds the wheel, installs it into a clean venv with `--no-deps`, and
imports `miniworld_engine.kernels` and `miniworld_engine.autotune.cache` with `src/` absent
from the path. Assert the autotune JSON count against `registry.csv` rather than a magic
number.

*Done when:* deleting a `package-data` glob turns CI red.

### B4 — the toolchain range is unpinned at the top and untested at the edges  (G3)

*Gap, measured:* `torch>=2.8`, `triton` with no floor at all, `requires-python >=3.10`. CI
varies Python (3.10, 3.12) and nothing else.

*Action:* give `triton` a floor matching the oldest version the kernels are known to compile
under. Add a second CPU job at the torch floor. State the tested combinations in the doc L4
asks for; do not claim untested ones.

*Done when:* `pyproject.toml` has no unbounded-below dependency and the supported-set page
lists only combinations something ran.

---

## C. Carried forward from the library standard

### C1 — the determinism test is wrong  (B5, P8)

*Gap, measured:* 8 failures in the last GPU run with differences up to `1.6e+04` — output-
scale, not reduction noise. The driver helpers use unseeded `torch.randn`, so the two calls
compare different inputs. **The test is wrong, not the kernels.**

*Action:* seed immediately before each checker call so both calls see identical input. Then
write the README determinism sentence P8 was blocked on.

*Done when:* the test passes for the right reason — verified by making one kernel
deliberately non-deterministic and watching it fail.

### C2 — tolerance is one global number  (B2, P2b)

*Gap, measured:* `DEFAULT_RTOL = 5e-2` for all 103 kernels. Stage C of the last GPU run
recorded per-kernel `rel`, so the data to calibrate exists.

*Action:* set each kernel's `rtol` from its measured `rel` with a stated margin; leave the
default only where there is no measurement, and say so in the row.

*Done when:* no kernel's declared `rtol` is more than the stated margin above its measured
`rel`, checked by `tests/test_declared_tolerance.py`.

### C3 — `arch` conflates an enforced gate with an intention  (G2, P7)

*Gap, measured:* `arch` is `sm80`×94, `sm90`×2, `sm100`×7. For cute/cuda it gates execution;
for triton it means "written for". The conflation currently skips **3 sm100 triton kernels
that would have run and been checked on sm86**.

*Action:* split into `arch` (minimum required, enforced) and `tuned_for` (informational).
Re-run `run_all` and confirm the 3 kernels move from skipped to checked.

*Done when:* the skip count drops by 3 and `tests/test_arch_gating.py` distinguishes the two.

---

## D. Make it usable by someone who is not the author

### D1 — the README is organised around the repository, not a newcomer  (L1, H3)

*Gap, measured:* sections are `Critical Safety`, `Layout`, `Benchmarking`, `torch.compile`,
`Supported hardware`, `Status`, `CLI`, `Toolchain`. There is no installation path for an empty
machine and no first-result walkthrough.

*Action:* a quickstart at the top — install, verify, run one kernel, read one number — whose
commands are copied verbatim into a test that runs them.

*Done when:* the quickstart commands are executed by CI, so they cannot rot.

### D2 — 110 of 130 doc files are a lab notebook  (L2)

*Gap, measured:* `docs/` has 124 tracked files; **101** are `docs/kernel-optimization/**/vN.md`
plus `.py`/`.txt`/ncu dumps.

*Action:* move them under `docs/notebook/` with an index explaining what they are and that
they are not maintained. Consumer-facing pages stay at `docs/`. Nothing is deleted (F3, L2).

*Done when:* `docs/` root holds only pages written for a consumer.

### D3 — no troubleshooting page  (L3)

*Gap, measured:* every failure mode that cost time this month — stale JIT lock, missing
mathdx include, autotune cache miss, unsupported arch, a `build all` that reports 527 units
instead of 859 — is undocumented. E4 makes the *messages* good; there is no page.

*Action:* one page, one section per failure, each with the exact message, the cause, and the
command that fixes it.

*Done when:* each section quotes a message string that exists in the source, checked by a test
so renamed messages cannot silently orphan a section.

### D4 — the supported set is a paragraph  (L4)

*Action:* a page listing tested (card, arch, torch, triton, CUDA, Python) combinations, each
citing the B1 verdict that tested it. Untested rows are marked untested.

*Done when:* it cites verdicts rather than asserting.

---

## E. Keep it from regressing

### E1 — work sits unpushed  (M1)

*Gap, measured:* 70 commits sat local while a second cluster ran a month-old tree; ten hours
lost on 2026-08-25. A further 16 were unpushed the same day.

*Action:* rule adopted — push at the end of each unit of work. Mechanise it with a `pre-push`
-adjacent check or a session-end reminder that reports `git log --oneline @{u}..` when it is
non-empty.

*Done when:* something other than memory reports the backlog.

### E2 — the consumer's environment cannot be reproduced here  (K4)

*Gap, measured:* the A100 failure took a session to attribute because the failing environment
was a different cluster and a fresh clone.

*Action:* a documented minimal repro recipe — fresh clone, no shared cache, cold JIT dir —
runnable on this cluster, so a consumer report can be reproduced instead of reasoned about.

*Done when:* the recipe reproduces the 527-unit symptom on a pre-fix commit.

### E3 — vocabulary tests read the working tree, not the repo  (D1, J4)

*Gap, measured:* `test_bench_target_vocabulary` and `test_bench_config_per_target` scan
`benchmarks/` on disk. Because results are gitignored, another branch's output makes them fail
locally while CI stays green — today 14 directories, including 257 MB of legitimate `mpnn`
output, all now in the attic.

*Action:* scope both to directories containing at least one tracked file. That checks repo
state, which is what the criterion is about, and stops flagging local scratch.

*Done when:* both pass with an unowned untracked directory present, and still fail when a
tracked results directory has no owning target.

---

## Order

**A1 → A3 → A2** first: version the package, make it buildable elsewhere, then move the
consumer onto it. **C1** next — it is small and a known-wrong test. Then **B1**, which every
remaining verification item depends on, and **B2** on top of it. **D** and **E** follow; **E3**
can be taken any time and is an hour's work.

The acceptance test, from the standards doc, is one sentence:

> On a machine that is not this one, a person who is not the author checks out a tagged
> version, installs it, runs the suite, upgrades `team-gm` to that tag, and gets the same
> numerical result and a measured step-time improvement — using only what is written down.

A1–A3 make the first clause possible. B makes the middle verifiable. D makes "only what is
written down" true.

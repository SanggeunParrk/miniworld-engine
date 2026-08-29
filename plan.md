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

### A1 — DONE — the version number does not distinguish two incompatible packages  (I1, I2, I4)

*Gap, measured:* `version = "0.1.0"` has not moved across **191 commits including a package
rename**. `git tag` holds `archive/gate-fuse-v1` and `archive/ln-bwd-cuda-v1` and no version
tag. The consumer's pinned tree declares `name = "miniworld-kernels", version = "0.1.0"`; main
declares `name = "miniworld-engine", version = "0.1.0"`. `import miniworld_kernels` →
`ModuleNotFoundError`. Nothing in either package's metadata lets a consumer tell them apart.
`CHANGELOG.md` is 264 good lines, all under `[Unreleased]`.

*Action:* the rename is a breaking change, so the next version is **1.0.0**, not 0.2.0 — a
0.x bump would understate it. Move `[Unreleased]` to `## [1.0.0] - 2026-08-25` with an
explicit **Breaking** section naming `miniworld-kernels` → `miniworld-engine` and the import
path change. Tag `v1.0.0`. Add `tests/layout/test_version_is_released.py`: the version in
`pyproject.toml` must have a matching `## [<version>]` heading in `CHANGELOG.md`, and
`[Unreleased]` must not be the only section.

*Done when:* the test fails on a version bump without a changelog entry, `git tag -l 'v*'`
lists `v1.0.0`, and `pip download miniworld-engine==1.0.0` from the tag resolves.

### A2 — PARTIAL — the only consumer is 191 commits behind and cannot advance  (K1, H4)

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

*Where it stands:* the migration is done and verified as far as it can be from here; what remains
is not a code change.

Done: submodule moved, repointed, pinned to the `v1.0.0` TAG rather than a SHA; pyproject, 13
source files, README and `uv.lock` migrated; every symbol team-gm imports verified present in
v1.0.0; the `MINIWORLD_KERNELS` enum VALUE deliberately left alone, because four config YAMLs
carry that string and renaming it is a config break needing its own deprecation.

Not done, and both are the user's call rather than mine:

* **Not committed.** 11 of the 13 migrated files carry the user's uncommitted work. Folding
  someone else's WIP into a commit is not mine to do.
* **The environment predates the floor.** Running the migrated modules on a GPU, 8 of 14 import
  and 6 fail with `infer_schema(func): Parameter input_shape has unsupported type list[int]` from
  `layernorm/triton/main.py`. Not a migration bug: team-gm's env has **torch 2.6.0** and
  miniworld-engine declares **torch>=2.8**. The consumer is below the declared floor, which is
  what a floor is for. `uv sync` resolves it -- 106 packages, the whole CUDA 13 stack, several GB,
  and a move from cu12 to cu13. That is a decision about someone's working environment, not a
  step I take unasked.

### A3 — DONE — six hardcoded personal paths make one kernel unbuildable elsewhere  (G1)

*Gap, measured:* `src/miniworld_engine/kernels/transition/cuda/__init__.py` lines 22–23,
48–49, 74–75 — `-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/include` and
`.../external/cutlass/include`, in three separate build functions.

*Action:* resolve mathdx from, in order: an explicit setting, `MATHDX_HOME`/`NVIDIA_MATHDX_HOME`,
the installed `nvidia-mathdx` package, then a documented failure that names the variable to
set (E4). Add `tests/layout/test_no_machine_paths.py`: no shipped source under `src/` may contain
`/home/<user>` or another absolute path outside the package or toolchain.

*Done when:* the new test fails on today's tree and passes after, and the kernel builds on a
machine with mathdx installed anywhere else.

---

## B. Prove it somewhere that is not this machine

### B1 — DONE (scoped by decision) — no GPU runs in CI  (J1)

*Gap, measured:* both CI jobs are `runs-on: ubuntu-latest`. The "gpu" step runs
`pytest --collect-only -m gpu` — it proves GPU tests can be *collected*. **103 kernels, 0
executed by CI.** `run_all` (`ok 94, failed 0, skipped 9`), the numerical suite (`98 passed,
2 skipped`) and opcheck (`5 passed`) are real and entirely manual.

*Action:* a self-hosted runner is not available, so make the gate Slurm-shaped instead: a
`scripts/ci-gpu.sbatch` that runs `pytest -m gpu`, `run_all`, and opcheck, writes a JSON
verdict with the commit SHA and the device, and a test that
fails when the newest verdict does not match `HEAD` or is older than N days. That converts
"the author ran it" into a dated, checkable artifact.

*Done when:* something dated and checkable stands behind the GPU claims.

*Where it stands:* the verdict system was built and cut -- a script plus a five-test gate was more
apparatus than the problem justified. The smaller answer reuses the artifact that already exists:
`run_all` writes a per-card manifest, it is committed, and it now carries a `#provenance` row with
the version, commit, tree state and date. `docs/supported.md` cites those manifests, so a support
claim now points at a dated artifact naming the code it was produced against.

*Closed, scoped to the release.* `tests/registry/test_a_release_has_been_run_on_a_card.py` is one
file: the version in `pyproject.toml` must appear in some manifest's `#provenance`, and that
manifest must have been produced from a clean tree. Both directions verified -- bump the version
and it fails naming the version; flip the provenance to `dirty` and it fails saying so.

Deliberately NOT a freshness gate. "The newest manifest must match HEAD" goes red on every
ordinary commit until someone finds a card, and a gate that is red by default is one that gets
switched off -- that is what cut the first attempt. The version only moves at a release, which is
the one moment where "nobody has run this" should stop the line.

What remains true: CI executes 0 of the 116 gpu-marked tests. A self-hosted runner would change
that and is **excluded by decision** -- it needs a runner token, a resident daemon on a cluster GPU
node, and that node's capacity held for CI. So J1 is met in the only sense available here: the
evidence exists, it is dated, and its absence fails a release.

The cost is written where it can mislead. `CONTRIBUTING.md` now says a green CI means nothing about
the kernels, with the day's own example: 95 tolerance bands narrowed 5x and three kernels ungated
from sm100, both green before and after, both would have been green if wrong.

### B2 — DONE — no end-to-end test proves the kernels are substitutable  (K2)

*Gap, measured:* 47 test files, all unit or contract. None runs a model with these kernels and
compares against the same model without them.

*Action:* one test that builds a small `team-gm` block twice — reference ops and
miniworld-engine ops — feeds identical seeded input, and asserts agreement at the *declared*
per-kernel tolerances (B2/P2a) compounded across depth. Mark `gpu`; it belongs in the B1
verdict.

*Done when:* it fails if any single kernel is swapped for a deliberately wrong one.

*Closed, after three ways of being vacuous.* A default-built Pairformer zero-initialises its
output projections, so both stacks were the IDENTITY -- `out == in` bitwise at L=64/128/256, and a
weight perturbed by 0.05 moved nothing because no weight reached the output. Randomising fixed
that. Then the residual diluted the branch, so a projection replaced with noise moved the output
only 1.7x more than the honest error; measuring `out - in` instead traded dilution for bf16
cancellation and reported 23%. The scale was chosen by sweeping it: at 0.15 the branch is 0.75x
the input, honest error 1.30e-02, corrupted 1.09e-01 -- 8.4x apart. Budget 6e-02, 4x the measured
value, the same margin as C2.

### B3 — DONE — the wheel is never built by anything automatic  (H1)

*Gap, measured:* verified by hand today and it passes — 563 files, `py.typed`, 186 autotune
JSONs, `registry.csv`, 91 configs, imports clean from an isolated `--target` install. Nothing
re-checks it, so the next `package-data` edit can silently drop the shipped cache.

*Action:* a CI job that builds the wheel, installs it into a clean venv with `--no-deps`, and
imports `miniworld_engine.kernels` and `miniworld_engine.autotune.cache` with `src/` absent
from the path. Assert the autotune JSON count against `registry.csv` rather than a magic
number.

*Done when:* deleting a `package-data` glob turns CI red.

*Closed.* The `wheel` job builds it, counts each asset against the tree rather than a written-down number, asserts the notebook and the A/B sets are absent, and imports from a `--target` install with `src/` off the path.

### B4 — DONE (scoped) — the toolchain range is unpinned at the top and untested at the edges  (G3)

*Gap, measured:* `torch>=2.8`, `triton` with no floor at all, `requires-python >=3.10`. CI
varies Python (3.10, 3.12) and nothing else.

*Action:* give `triton` a floor matching the oldest version the kernels are known to compile
under. Add a second CPU job at the torch floor. State the tested combinations in the doc L4
asks for; do not claim untested ones.

*Done when:* `pyproject.toml` has no unbounded-below dependency and the supported-set page
lists only combinations something ran.

*Closed as far as evidence allows.* `triton>=3.3` from code evidence; einops/jaxtyping/numpy keep no floor deliberately, because nothing has run against an older release of any of them and a guessed floor reads like a measured one. `docs/supported.md` states what ran.

---

## C. Carried forward from the library standard

### C1 — DONE — the determinism test is wrong  (B5, P8)

*Gap, measured:* 8 failures in the last GPU run with differences up to `1.6e+04` — output-
scale, not reduction noise. The driver helpers use unseeded `torch.randn`, so the two calls
compare different inputs. **The test is wrong, not the kernels.**

*Action:* seed immediately before each checker call so both calls see identical input. Then
write the README determinism sentence P8 was blocked on.

*Done when:* the test passes for the right reason — verified by making one kernel
deliberately non-deterministic and watching it fail.

*Closed, and it moved the goalposts.* The seeding fix works -- the control proves inputs are identical (a checker's torch-computed reference matched across two calls while the kernel's did not). What it found is that the file's promise was false: `augmented_attention_bwd_atomic_triton` is not bitwise reproducible, and the sample had picked the one atomics kernel that is. `NOT_BITWISE` now names the exception with its cause, checked from both sides.

### C2 — DONE — tolerance is one global number  (B2, P2b)

*Gap, measured:* `DEFAULT_RTOL = 5e-2` for all 103 kernels. Stage C of the last GPU run
recorded per-kernel `rel`, so the data to calibrate exists.

*Action:* set each kernel's `rtol` from its measured `rel` with a stated margin; leave the
default only where there is no measurement, and say so in the row.

*Done when:* no kernel's declared `rtol` is more than the stated margin above its measured
`rel`, checked by `tests/registry/test_declared_tolerance.py`.

### C3 — DONE — `arch` conflates an enforced gate with an intention  (G2, P7)

*Gap, measured:* `arch` is `sm80`×94, `sm90`×2, `sm100`×7. For cute/cuda it gates execution;
for triton it means "written for". The conflation currently skips **3 sm100 triton kernels
that would have run and been checked on sm86**.

*Action:* split into `arch` (minimum required, enforced) and `tuned_for` (informational).
Re-run `run_all` and confirm the 3 kernels move from skipped to checked.

*Done when:* the skip count drops by 3 and `tests/registry/test_arch_gating.py` distinguishes the two.

---

## D. Make it usable by someone who is not the author

### D1 — DONE — the README is organised around the repository, not a newcomer  (L1, H3)

*Gap, measured:* sections are `Critical Safety`, `Layout`, `Benchmarking`, `torch.compile`,
`Supported hardware`, `Status`, `CLI`, `Toolchain`. There is no installation path for an empty
machine and no first-result walkthrough.

*Action:* a quickstart at the top — install, verify, run one kernel, read one number — whose
commands are copied verbatim into a test that runs them.

*Done when:* the quickstart commands are executed by CI, so they cannot rot.

*Closed.* Four steps at the top of the README, three of them GPU-free, and every `# cpu` block is
run by `tests/layout/test_quickstart_runs.py` as one script -- which is what a reader pastes.
The first version ran them line by line, which broke every multi-line `python -c`, and then passed
against a deliberately broken command because a shell script's exit code is its LAST line's. Both
fixed; the probe fires now.

### D2 — DONE — the lab notebook was a third of the repo  (L2)

*Gap, measured:* `docs/` has 124 tracked files; **101** are `notebook/**/vN.md`
plus `.py`/`.txt`/ncu dumps.

*Action:* move them under `docs/notebook/` with an index explaining what they are and that
they are not maintained. Consumer-facing pages stay at `docs/`. Nothing is deleted (F3, L2).

*Done when:* `docs/` root holds only pages written for a consumer.

### D3 — DONE — no troubleshooting page  (L3)

*Gap, measured:* every failure mode that cost time this month — stale JIT lock, missing
mathdx include, autotune cache miss, unsupported arch, a `build all` that reports 527 units
instead of 859 — is undocumented. E4 makes the *messages* good; there is no page.

*Action:* one page, one section per failure, each with the exact message, the cause, and the
command that fixes it.

*Done when:* each section quotes a message string that exists in the source, checked by a test
so renamed messages cannot silently orphan a section.

*Closed.* Ten sections, one per failure that cost time today. Anchors are declared in the test
rather than guessed from the heading -- the messages are f-strings, so the heading shows an
interpolated example while the source holds the template, and guessing the overlap failed on four
correct sections first. A heading with no anchor and no `None` fails, so a new section cannot slip
past the check.

### D4 — DONE — the supported set is a paragraph  (L4)

*Action:* a page listing tested (card, arch, torch, triton, CUDA, Python) combinations, each
citing the B1 verdict that tested it. Untested rows are marked untested.

*Done when:* it cites verdicts rather than asserting.

*Closed.* `docs/supported.md`. Every row names its artifact, and a section lists what has NOT been run -- the 9 kernels declared sm90/sm100 that nothing here has executed.

---

## E. Keep it from regressing

### E1 — DONE — work sits unpushed  (M1)

*Gap, measured:* 70 commits sat local while a second cluster ran a month-old tree; ten hours
lost on 2026-08-25. A further 16 were unpushed the same day.

*Action:* rule adopted — push at the end of each unit of work. Mechanise it with a `pre-push`
-adjacent check or a session-end reminder that reports `git log --oneline @{u}..` when it is
non-empty.

*Done when:* something other than memory reports the backlog.

*Closed.* `.githooks/post-commit` prints, after every commit, how many commits the remote does not
have, the oldest one's age, and the first five subjects. It does not block and does not push --
refusing a commit for being unpushed is nonsense, and refusing a push is the opposite of what is
wanted. `git config miniworld.noahead true` silences it.
`tests/layout/test_unpushed_work_is_reported.py` builds a bare remote and a clone and runs the
hook against both states, rather than reading it: silent when everything is pushed, and naming the
commit when one is not.

### E2 — DONE — the consumer's environment cannot be reproduced here  (K4)

*Gap, measured:* the A100 failure took a session to attribute because the failing environment
was a different cluster and a fresh clone.

*Action:* a documented minimal repro recipe — fresh clone, no shared cache, cold JIT dir —
runnable on this cluster, so a consumer report can be reproduced instead of reasoned about.

*Done when:* the recipe reproduces the 527-unit symptom on a pre-fix commit.

*Closed, and it needed no GPU.* A detached worktree at `0854ac4^` enumerates **527** units where
main enumerates **859** -- the whole report, settled in seconds on CPU. That is the recipe's point:
most of what makes a report irreproducible is accumulated state, not hardware, so the doc strips
caches first and reaches for a card last. `tests/builder/test_build_matrix.py` now pins the count
against the registry, so the regression fails in CI instead of in someone's ten-hour job.

### E3 — DONE — vocabulary tests read the working tree, not the repo  (D1, J4)

*Gap, measured:* `test_bench_target_vocabulary` and `test_bench_config_per_target` scan
`benchmarks/` on disk. Because results are gitignored, another branch's output makes them fail
locally while CI stays green — today 14 directories, including 257 MB of legitimate `mpnn`
output, all now in the attic.

*Action:* scope both to directories containing at least one tracked file. That checks repo
state, which is what the criterion is about, and stops flagging local scratch.

*Done when:* both pass with an unowned untracked directory present, and still fail when a
tracked results directory has no owning target.

*Closed.* Both scans go through one `tracked_subdirectories` helper that asks `git ls-files`, and
it raises rather than falling back to `iterdir()` -- a silent fallback restores exactly what it
replaces. Verified both ways: a leftover directory no longer turns them red, and removing a
tracked `configs/bench.yaml` still does.

---

## F. Closed by the repo cleanup

Not planned; found by tidying, and each one a defect the plan would not have reached.

| | what was wrong | how it is closed |
|---|---|---|
| F1 | `third_party/` held no third-party code — 45 files of first-party scratch that the *name* exempted from ruff and ty (112 unseen findings), pointing at an `_ct_cutlass` that exists nowhere | deleted; the one paragraph worth keeping moved to CONTRIBUTING |
| F2 | one kernel log split across `docs/` (prose) and `profiles/` (captures), with `docs/` already holding captures too | one tree, then placed under the kernel it is about: `kernels/<family>/notes/` |
| F3 | `load()` did not release torch's build lock when a build raised, so the next call in the same process waited on a lock nobody held | `load_extension` takes the lock with it on failure |
| F4 | `MINIWORLD_CONFIG_DIR` accepted a path that does not exist; four audit scripts had pointed at `configs/grid` since P10 moved it | rejected with the value and both ways out |
| F5 | `devices.py` resolved its manifest dir as `parents[3]/configs/devices` — the repo only in an editable install, so a wheel wrote its per-card record nowhere | `autotune/manifests/`, package data |
| F6 | `AutotuneKernel` declared seven groups; four had no call site, and the field was `frozenset[str]` so an unknown name silently unlocked nothing | three groups, typed, rejected by `configure` |
| F7 | `--bench-budget` compared one drained-stream launch against do_bench's queued median — 10x different quantities — so the first config in grid order won. 349 of 1244 shipped cache entries carry that fingerprint | feature removed; **the A5000/A6000 caches still need rebuilding** |
| F8 | 27 test files derived the repo root by fixed depth; `tests/build/` was invisible to pytest (`norecursedirs`), losing 65 tests silently | depth-independent derivation; a guard that fails on any test file in a skipped directory |
| F9 | 360 rendered SVGs (5.9 of 9.1 MB) cited by nothing, and 29 `.gitkeep` holding directories both writers already `mkdir` | data kept, renders ignored |

The one that is not closed is **F7's consequence**: the shipped caches were built with the
budget on for 51 ops, so 349 entries are the grid's first config rather than the fastest. Rebuild
is separate work and is not started.

## G. Open technical work, indexed here and detailed elsewhere

Two documents carry substantial open work and were invisible from this file, so "what is left"
lived in three places and a reader could not know which. The detail stays where it is; this is the
index.

### G1 — the cute backend is mid-port to quack 0.5.0  (`docs/design/quack-0.5.0-cute-port.md`)

*Gap:* phases 0-3 landed at `a1aea37`. Phase 4 is blocked on a cutlass-dsl 4.4.2 -> 4.5.2
launch/stream ABI change, so the cute backend tracks two quack versions at once.

*Done when:* the compat shim in `kernels/_quack_compat.py` is unnecessary and the port document
becomes a record.

### G2 — residual fusion is done for two op families, explicit for four  (`docs/design/residual-fusion.md`)

*Gap:* Transition and TriangleMultiplication fuse the residual into the kernel epilogue.
TriangleAttention, OuterProductMean and the two team-gm attention layers still add it explicitly,
which is a separate pass over the pair tensor each time.

*Done when:* the "remaining" column of that document's table is empty, or the entries left in it
say why they will stay.

### G3 — half a cache build's compile CPU goes to configs it throws away
(`docs/records/where-the-cache-build-spends-its-time-a6000.md`)

*Gap:* 13,875 of the A6000 rebuild's 869,844 configs ran the full 60 s compile budget and were
SIGKILLed. Those 1.6% took 231 of the build's 429 compile CPU-hours, and they are CONCENTRATED:
205 of 461 rounds kill nothing at all, while `adaln_gemm_gate` at three lengths burned 27.5
CPU-hours on kills that were 98%, 94% and 48% of those rounds -- 556 of its 15,552 configs. The
per-round median is 19%; the build-wide 54% is those few rounds. It is not the same mechanism
as shared-memory overflow -- which `autotune/viability.py` already predicts and skips -- but the
same shape of problem: a config whose outcome is knowable before it is compiled.

Lowering the budget is not the answer on its own. The per-kernel distributions differ by an order
of magnitude, and the tail of one kernel is the body of another:

    transition_fwd_b2b_ktiled      p50 0.35s  p90 1.76s  p99 17.29s  max 47s   0 killed
    augmented_attention_bwd_split  p50 1.66s  p90 8.52s  p99 60.00s  max 60s  11 killed, 18% of CPU

At a 20 s budget the first kernel loses configs that compile fine.

*Done when:* the killed set is either predicted per kernel from probe data (as shared memory is)
or bounded by a budget chosen from the measured distribution -- and the build reports which it
did. The evidence it needs now exists: every round prints its distribution, and every killed
config is named in the `.smem` log (`smem_log.killed()`). `autotune/compile_budget.py` is the
first half of the answer: 63% of the kills ruled out from the probe round already paid for, at
0.19% wrongly ruled out. It is not wired into the build.

### G4 -- a round re-verifies configs triton's on-disk cache already holds

*Gap:* a compiled binary is shared across input SIZES. Measured on two kernels with one shared
`TRITON_CACHE_DIR`, running L=256, 512, 1024: the first length added 37-38 cache entries, every
later length added **zero**. Triton's compile key carries the constexprs and the argument
specialisation, not the argument values, so 256 and 512 are the same binary.

The build nonetheless submits every config to the pool again at each new length, to find out they
are all cache hits. `.compiled` stops that WITHIN a unit and across its restarts, but it is per
shard and its key carries the autotune round, so a second length matches nothing.

The parent could answer it itself: triton's cache key is ~0.34 ms to compute and the lookup is a
stat. Filtering the payload list before chunking would skip the pool entirely for a warm length.

*Not sized yet.* The measurement that would size it needs a LARGE grid run twice at two lengths
in one cache dir; the run above used 36-config grids where the chunk size collapses to 1 and a
fork per config dominates everything, so it does not scale up. Of the rebuild's 869,844 compile
attempts only 221,487 produced a distinct binary, so about three quarters were re-verification --
but at what per-config cost is exactly the unmeasured part.



### G5 -- DONE -- the shape key carries the whole shape

*What was wrong:* `K`, `ND`, `H`, `HEAD_DIM`, `N`, `D` are SHAPE -- dimensions of the tensors the
kernel reads, exactly as the row count is -- but they stood in `key=[...]` as separate entries
beside `shape_key`, as though they were a different kind of thing. `shape_key` is the one parameter
that exists ONLY to be keyed on (a404fb9: asserted to be read in no body), which makes it the right
place to carry all of it.

*Done.* 87 of 88 autotune kernels, across nine families:

    key entries   238 -> 111        shape axes   122 -> 2        flags   29 -> 22
    key length    was 1:6  2:38 3:29 4:8 5:5 6:2
                  now 1:72 2:10 3:5  4:1

The two remaining shape axes are `transition_fold_triton`'s `['N', 'K']`, which is correct and
stays: that kernel reads only the WEIGHTS and never the activation, so N and K are its whole shape
and it has no `shape_key` at all (6e1d9e8).

*The mechanism:* `autotune/shape_key.pack`, with the three bucket helpers taking the axes as
keywords -- `atom_key(L, H=H, HEAD_DIM=D)`. Axes pack by SORTED NAME, so two launchers cannot
disagree about what a positional argument meant (3d47a78, 7c16d16 were exactly that), and the axis
NAMES are folded in, so a launcher naming a different set MISSES rather than colliding -- 6948c77
cost 1.73x because a lossy key collided. Passing no axes returns the old value unchanged.

*The guard, and what it caught.* `tools/key_gaps.py` reads the launches, takes the axes of each
`shape_key=` expression and INTERSECTS them across every launch of a kernel: an axis counts as keyed
only if EVERY launcher folds it. Four holes in it were found by the families themselves, each one a
way for a forgotten site to pass silently:

    conditional  `both_key(M, K=K) if shape_key is None else pack(shape_key, K=K)`   family 2
    aliased      `import _ln_bwd_persistent as _ln_bwd_persistent_jit`               family 3
    positional   `_shape_key(shape_key, M)` passed by position, not keyword           family 4
    runtime arg  an axis that is not `tl.constexpr` at all -- outside its reach       family 5

The first three are fixed in the tool (7e2a32e and the family commits) and verified by removing one
site's fold and watching that kernel reappear. The fourth cannot be: the tool checks constexprs, and
`_bwd_elem_kernel`'s `N` is a runtime argument. It was found by reading, and then by a one-off sweep
-- every axis each key has LOST since f3ca735 against the axes its launches now fold -- which also
found `checks/gated_projection.py`'s backward site, folded in neither family 1 nor 5.

Resolution is by IMPORT, not by directory or by name: four kernels are called `_attn_fwd`,
`gated_projection_gate_gemm_triton` is DEFINED in `bias_only_attention/`, and
`layernorm_linear/triton/mmajor_bwd.py` launches a `layernorm` kernel.

*Open, from the same reading of the bodies:* five flags are kept with no measurement either way --
`HAS_W`, `HAS_SB`, `HAS_BIAS`, `HAS_ROWSCALE`, `FROM_PREACT`. Each adds a row-vector load, broadcast
across the tile: small against an (M, N) tile, but not nothing. What would settle them is the
comparison SAVE_GATE got -- same shape, both flag values, same card -- which needs a cache holding
both values, which the next full build produces.

*Next:* the cache must be rebuilt on the new keys. Every shipped entry is stale by construction.

### G6 -- the miss fallback is unmeasured on the kernels it exists for

`heuristic_subset` could not launch at all on three adaln backward kernels: ranking by distance
from the middle of every block axis puts every axis at its middle AT ONCE, the offsets compound,
and all 24 candidates land in the largest-tile corner. Measured on an A5000 with a stale cache,
`adaln_bwd_dw_triton` asked for 294,912 B of shared memory against a 101,376 B limit -- 2.9x, with
no launchable config in the subset. Fixed in 72c131b by reserving a quarter of the cap for the
smallest tiles, chosen from the same industry centre so the existing test still holds.

*What is still open:* the fix makes the subset launchable; it does not say what it costs. The
docstring's claim -- warps=4/stages=2 lands within 5% of the full-grid winner in 83% of buckets --
was measured for the old subset, over this repo's 374-bucket sweep. Giving a quarter of the cap to
the smallest tiles cannot improve that number and nobody has re-measured it.

The measurement is cheap and the data already exists: for each bucket in a built cache, compare the
full-grid winner against the best of `heuristic_subset(grid)` under both rules. It needs a cache
built on the current keys, which is the same prerequisite as everything else here.

Worth doing at the same time: the three kernels found this by dying, which means no test walks the
fallback path. `tests/autotune` exercises `heuristic_subset` on synthetic configs, not on a real
grid against a real shared-memory limit.


---

### G7 -- the tile visit order is a tuned axis on 20 kernels, and its ladder is measured on four

*What it is.* A matmul kernel splits its output into tiles and gives one to each program. Two
programs share memory when they share a row or a column, and whether that sharing becomes a cache
hit is decided by the ORDER the programs run in. A 2-D grid cannot express the choice -- CUDA varies
axis 0 fastest, so `pid_m = program_id(0)` pins every kernel to one end of the axis, whatever suits
the shape. `GROUP_M` is the order as one tuned number; `kernels/_tiles.py` holds it and the reason.

*Done.* 20 kernels take the axis and launch a 1-D grid through `tile_grid`/`tile_order`. 18 cannot
and say why, one line each, in `kernels/tile_order_exempt.csv` -- 8 are 1-D grids over M alone (no
column-tile axis exists) and 10 are attention (the contraction is a loop in a jit helper, not a grid
axis). A test refuses a new GEMM that is neither. The ladder is `1 4 16 65536`, and it carries BOTH
ends on purpose: 7 of the 20 walked columns first before the axis existed and 13 walked rows first,
so whatever a kernel did before stays reachable and a card that gains nothing loses nothing.

*Measured so far,* A6000, full grid, tuned against tuned (both arms free to choose tile, warps and
stages), best time per rung:

    transition_expand_swiglu_triton            G=1 49.15us  vs 2-D order 58.37us   1.187x
    trimul_outproj_gemm_gate_saveact_triton    G=4 33.79us  vs 2-D order 35.84us   1.061x
    transition_fwd_b2b_ktiled_triton           G=4 1019.9us vs 2-D order 1068.0us  1.047x
    cond_transition_expand_swiglu_triton        all four rungs 15.36us              1.000x

Three of four gain and one gains nothing -- the fourth's operand fits L2, which is what `_tiles.py`
predicts. The middle rungs win twice, which is the evidence they earn their place; before this,
`4` and `16` rested on one kernel at one shape.

*What is still open.* The remaining 347 of 351 (op, dtype, shape, width) units. The question the
full sweep answers is not only "where does it pay" but whether a FOUR-rung ladder is worth 4x the
config grid on every future build -- `transition_expand_swiglu_triton` alone searches 6,904 configs
per unit now. If the answer is that two rungs capture the gain, the ladder should shrink.

Also unmeasured: `trimul_gemm_gate_mmajor_triton` is exempt because one program produces BOTH the
left and right outputs from one row block of x, and putting channels on the grid would make several
programs re-read the same rows. Whether breaking that fusion to gain the order axis would pay is a
judgement, not a measurement.

## H. Closed by verifying what the repo claims

Separate from F: F was found by tidying the tree, these were found by RUNNING the things this
repo says it does -- a clean clone, the GPU-marked suite, the build-system audit, the coverage
replay. Seven of the nine are checks that could not fail, could not pass, or answered a question
nobody asked. One is a 3.2x defect in the build itself, and one was found by a verification of my
own failing.

| | what was wrong | how it is closed |
|---|---|---|
| H1 | `_precompile_round` decided whether to run the parallel pool by bare config sig while the fork guard decided by `<kernel>\t<sig>`. Rounds interleave across a unit's kernels, and `augmented_attention_fwd`/`_bwd_atomic` have byte-identical grids (4x4x6x9 = 864), so the second kernel's round was skipped as "already compiled", the pool never ran, and every config fell through to the fork the pool exists to avoid. The comment called it deliberate: *"The sig alone still goes in `_COMPILED` so the ROUND skip works exactly as before."* | one record, one key. `_COMPILED`/`_mark_compiled` deleted -- always written beside `_mark_outcome` from the same zip. Measured same node, cold caches per arm: **5759s -> 1818s, 864 forks -> 0** |
| H2 | two tests PINNED H1 -- "the ROUND skip must still see it", "legacy line still suppresses recompiling in the round" | replaced by one that fails on the old code |
| H3 | `test_the_hooks_directory_is_the_one_git_uses` asserted `core.hooksPath` unconditionally. It is per-clone local config and CONTRIBUTING says so ("once per clone"), so the test could only pass on the author's machine | unset skips; wired elsewhere -- the case that is a defect -- still fails |
| H4 | `dev audit` printed 139 FAIL and exited 1 on every default run. 88 were reachability with no `--shards` to look in; 51 were coverage against the key `cpu`. Both are missing INPUTS reported as broken ARTIFACTS, and `check_key_spread` in the same file already had the right shape | one WARN naming the flag, then return. With real inputs: reach 88 OK, coverage 91 OK, exit 0 |
| H5 | the `gpu` marker is documented as "needs a CUDA device" and CI only ever excludes it, so nothing exercised what it means: `pytest tests/` on a login node produced 11 failures reading "Found no NVIDIA driver" | `tests/conftest.py` skips them without a card. Collection untouched, so `--collect-only -m gpu` still reports its 100 |
| H6 | the audit's brute/prune/keys checks -- 264 findings over 88 live autotuners -- ran nowhere automatic. They could not: H4 made the command exit 1 every time | a CI step, after H4 |
| H7 | `cache.py` documents `_CACHE_MISSES` as "the only direct measure of whether the cache covers a workload" and pointed at `miniworld-engine audit`, which is the STATIC check and never reads it. The replay it meant, `builder.audit`, had **no caller in src, tests or benchmarks** | `dev audit --replay`, refusing without a card rather than reporting an empty miss set as a pass; a test pins the wiring |
| H8 | `missing_pairs 0` read as "the cache covers the workload" when it means "every declared bucket is present". Declared work is (op, dtype, shape bucket); the cache key also carries each kernel's constexprs. Measured on an A6000: **363 lookups the module matrix asks for and the cache does not serve, across 42 of 91 ops** | the number carries its own limit next to it; the 363 recorded in `docs/records/cache-coverage-replay-a6000.md` as the rebuild's work list |
| H10 | one unit reported `grid=864` and, under it, `527 configs, best 0.0338ms`. The other 337 appeared nowhere -- 39% of the searched space, in a build whose purpose is to search it. The EXCLUSIONS are legitimate, and traced on the card: of the 54 `BLOCK_M1=256,BLOCK_M2=256` configs, 26 exceed the 60s compile budget (register spill, all of `num_warps` 1 and 2) and the other 28 compile and then fail at launch -- `OutOfResources: shared memory, Required: 196608, Hardware limit: 101376`, and at `num_stages=1` too, so it is the tile, not the staging. What was the defect is that none of it was recoverable afterwards: `prune_configs` returns the full list so nothing is pruned, triton swallows the OutOfResources itself so no exception reaches the capture layer, and a config scoring +inf is correctly not stored. I published two wrong reconstructions before measuring, and a third from a probe whose config pin had not applied | counted where the drop happens and printed: `unusable=337 (39% of the searched space could not run on this card)`, stated even when 0. Cost: four GPU jobs to establish that the exclusions themselves were fine, which the first reading had already said |
| H9 | `_CACHE_MISSES` only ever grew, so a before/after over one process returned the before twice -- it could not observe a filled cache. Found by exactly that happening: a scoped fill covered four keys, merged, and the second replay named the same four in 0s | `clear_cache_misses()`, called by `builder.audit`; the docstring names what the clear cannot fix (triton memoises per autotuner instance, so the replay must be a fresh process) |

And one that is not a defect but the thing H8 exposed:

**`build all` with no flags now produces a complete cache.** Two work lists exist and neither
covers the other -- `op_units` tunes every declared kernel but only through its own driver, so a
module's real constexpr combinations never occur; `cases` reaches those and reaches 48 of 91
kernels. The default was the first alone. It is now both, in order, with a merge between them, and
the second runs with `settings.fill_gaps` so a key the first pass already tuned costs a 3-config
re-rank rather than a full-grid sweep. Without that flag the module pass re-benches everything the
op sweep did -- the 244 GPU-h that made these an either/or.

*Verified, scoped.* Each measurement in its own process, `gated_projection` on one card:
**BEFORE 4 misses -> pass 2 (8 units, 0 failed, 8 (op, bucket) rows merged) -> AFTER 0.** The
mechanism closes the gap it was added to close. Four keys is evidence for 363, not proof of it;
the full number is only answerable by the rebuild.

*Settled, by control.* The A/B's two arms chose different winners for all three keys of the unit
-- 5.1 vs 6.1 us, 8.2 vs 9.2 us, one quantisation step apart at this shape -- so "the patch does
not change the result" needed a same-arm baseline. Three more runs of the PATCHED arm, same node,
same card, cold caches each:

| compared | same winner | worst cost of taking the other's pick |
|---|---|---|
| patched vs patched (b, b1) | 1 of 3 | +2.9%, +20.0%, +0.0% |
| patched vs patched (b1, b2) | 2 of 3 | +0.0%, +0.0%, +12.5% |
| HEAD vs patched (a, b) | 0 of 3 | +3.0%, +20.0%, +12.5% |
| HEAD vs patched (a, b1) | 1 of 3 | +0.0%, +20.0%, +12.5% |

Identical code disagrees with itself as much as the two arms disagree, at the same magnitudes.
The winner instability is the bench's own, not the patch's: at 5-8 us on this shape `do_bench`
resolves to about 1 us, and one step is 12-20%. Wall time varies too -- 1818s, 1903s, 1879s for
the same arm -- and so does the compile-monster set the SIGKILL budget catches (41, 43, 44
known-bad across identical runs).

Two things follow, and the second is the one that matters for the rebuild:

* the capture fix is exonerated. It changes which code path compiles, not what is benched
  (527/525, 288/288, 593/593 configs; 4 ops recorded in every run).
* **a single tuning measurement does not identify the fastest config at this shape.** What
  absorbs it is that an entry stores the top 5 and `_cached_subset` hands all 5 back, so triton
  picks among them at run time rather than trusting the build's ordering. That is load-bearing,
  and it was not written down anywhere.

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

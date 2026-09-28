# Vast H100 workspace (2026-09-27)

**Retired:** the user terminated this instance on 2026-09-27. Do not connect or
resume synchronization. Current GPU work uses local Slurm with `normal_h100`
QoS; see [local-h100.md](local-h100.md). The following is a historical record.

Remote workspace: `/workspace/miniworld-engine`. Local source: this checkout.
The initial remote source is engine `afd54410a0bf59a204b6ca00af38909a684cf50f`;
all 11,543 tracked file blobs matched the Git index after transfer.

## Environment

Use `source /workspace/setup/env.sh` before remote work. The dedicated environment
is `/workspace/envs/engine`; the image's system Python/CUDA are not the engine environment.

The environment matches MiniWorld's local `cu128` runtime: Python 3.10.20,
PyTorch 2.10.0+cu128, Triton 3.6.0, CUDA nvcc 12.8.93, CUTLASS DSL 4.5.2,
quack-kernels 0.5.0, BF16 kernels. Exact selected Python dependency pins are in
`/workspace/setup/requirements-cu128.txt`. CUDA 12.8 development libraries are
required in addition to nvcc (PyTorch extensions include `cusparse.h`, etc.).
Public `nvidia-mathdx==25.6.0` supplies optional cuBLASDx headers.
Final environment snapshots are `/workspace/setup/pip-freeze.txt`,
`conda-explicit.txt`, and `packages-final.json`; transitive alignment pins are in
`requirements-transitive-match.txt`. FlashAttention 4.0.0b19 is installed to match
the local runtime; its separate optional kernels were not part of this benchmark.

This matches the measured Python 3.10 runtime, not the engine's separate local
Python 3.12 development environment. The driver is host-controlled (595.71.05);
the image's original toolkit is CUDA 13.1 and remains available for system tools.
The two full H100 80GB HBM3 GPUs have 132 SMs each, 700 W limits, and NV18 connectivity.

## Synchronization

From the local engine checkout:

```sh
scripts/vast-sync.sh status
scripts/vast-sync.sh push          # tracked source -> remote; retains overwritten files in remote backups
scripts/vast-sync.sh pull          # remote results/setup -> local .bench/vast-20260927
scripts/vast-sync.sh fetch-source  # remote source -> separate local review directory
scripts/vast-sync.sh watch         # result pull every 120 seconds, until midnight Asia/Seoul
```

The helper uses the task-specific local SSH key. Override `VAST_SSH_HOST`,
`VAST_SSH_PORT`, `VAST_SSH_KEY`, and `VAST_REMOTE_ROOT` when moving instances.
Never copy the private key into the remote workspace. No Vast API key is required.

Local source is authoritative; push source changes between jobs, so running jobs
keep a stable implementation. Remote changes are fetched into
`.bench/vast-20260927/remote-source/` for review before integrating them locally.
Neither direction uses `--delete`. Existing tracked remote files overwritten by
`push` are backed up under `/workspace/sync-backups/<UTC timestamp>/`.
Python environments are reproduced from pins, not rsynced across machines.
Generated GPU/toolchain caches stay device/environment-specific; do not relabel
historical cache identities to make them valid.

## Verification and results

Remote results live in `/workspace/vast-results`; local copies live in
`.bench/vast-20260927/results`. Each GPU gets its own Triton, Inductor, extension,
and native-JIT cache under `/workspace/cache`, and an independent sequential job queue.
This prevents two benchmark processes from timing on the same GPU simultaneously.

`source-verification.json` records source identity. `environment.json` records
GPU properties, dependency versions and attention binary ABI/hash checks.
`gpu*/status.json` and pytest XML preserve execution outcomes, including failures.
The retained historical benchmark can catch errors and return exit code zero:
inspect the per-mode JSON `error` fields, not just the process exit code.

Historical module comparisons reuse the unchanged
`docs/records/verdicts/version-compare-20260923/bench.py` in a separate result directory.
The regular `benchmarks/runners/bench.py` produces current CSV/provenance records.
Compare only matched shapes, dtypes, gradients, dropout, compilation and graph regimes.
Older records may predate later source improvements, especially single-direction TriMul.
The supplemental normalization replay follows the earlier eager-call graph protocol
and is labeled diagnostic, separate from compiled production module timings.

## NCU

Nsight Compute 2025.4.1 is installed. Initial actual-kernel probes on both GPUs
failed with `ERR_NVGPUCTRPERM`; ordinary CUDA execution passed. The host reports
`RmProfilingAdminOnly: 1`, and the container lacks `CAP_SYS_ADMIN`.
A host/support change is needed for performance-counter access. Do not describe
NCU as working merely because `ncu --version` succeeds. Initial error logs are
`ncu-probe-gpu0.log` and `ncu-probe-gpu1.log` in the result directory.

Do not stop or destroy the rented instance as part of synchronization or tests.

## Measured outcome

The portability suite passed 110 pytest cases (29 on GPU0, 81 on GPU1).
After dependency alignment, six additional engine inference smoke tests and a
GPU1 CUDA execution probe passed. The standard benchmark runner produced 12
successful rows across the two GPUs.
Additional attention checks cover starting/ending attention at L384 and L768,
nonzero weights, full gradients, changed-input/weight/gradient/mask graph replay,
and native CUDA dispatch traces. This is an environment/reproduction check;
it does not replace release sanitizer qualification or qualify experimental kernels.

Representative BF16 complete training F+B timings, in milliseconds, using the
historical module harness with CUDA Graphs and live dropout RNG:

| Module | L384 historical | GPU0 L384 | L768 historical | GPU1 L768 |
| --- | ---: | ---: | ---: | ---: |
| Bidirectional TriMul | 1.060 | 1.012 | 4.190 | 3.979 |
| Transition | 0.562 | 0.550 | 2.140 | 2.090 |
| OPM | 2.035 | 1.988 | 7.625 | 7.351 |
| PairWeightedAveraging | 1.586 | 1.588 | 4.132 | 4.004 |
| TokenDiT | 0.466 | 0.465 | 0.940 | 0.949 |

GPU0 also measured L768 TriMul at 4.032 ms: about 1.3% apart from GPU1.
These are comparisons to recorded historical results, not simultaneous runs of
identical source revisions on the original and rented hosts. Single-direction
TriMul's larger apparent improvement includes later source changes and must not
be attributed to the rented hardware. PWA L384 inference was 0.432 ms versus
0.407 ms historically (6% slower).

The separate normalization diagnostic preserves the historical eager-call graph
timing protocol. GPU0 LayerNorm M8192/D1024 F+B was 0.0568 ms versus 0.0443 ms
historically (28% slower). Explicit atomic recheck was 0.0566 ms; persistent was
0.1124 ms, so switching to persistent does not resolve the difference. GPU1 reproduced the small LayerNorm deviation at 0.0580 ms (31% slower). No global
dispatch override was installed. Other GPU0 diagnostic ratios were 0.888–1.054.
Historical LayerNorm records explicitly calibrated atomic/persistent first;
new-device tuning/cache and runtime differences remain relevant to comparison.

Machine-readable records: `.bench/vast-20260927/results/report.json` and
`historical-comparison.csv`; raw pytest XML, benchmark JSON/CSV and traces remain
under the corresponding `gpu0` / `gpu1` directories. Bootstrap failures caused by
missing CUDA development headers were retained separately, then successfully
rerun after installing the headers. An initial attention graph-check harness
stream error was fixed and rerun; no production kernel changes were needed.

## New session quick start and coordination rules

Start on the local server in `/home/psk6950/miniworld-engine`. Read root `AGENTS.md`,
this document, and `git status --short`; retain any other session's modifications.
The SSH private key already exists locally and must never be pasted into a chat,
checked in, or copied to Vast. No password/API key is needed.

```sh
ssh -F /dev/null -i /home/psk6950/.ssh/vast_engine_20260927 \
  -o BatchMode=yes -p 27779 root@202.122.49.242
# On Vast:
source /workspace/setup/env.sh
cd /workspace/miniworld-engine
```

Direct endpoint above is verified. The supplied fallback endpoint is
`root@ssh2.vast.ai -p 34447`; it has not been validated. Local port forwarding is
optional; benchmarks do not require port 8080. Host/port may change if the rented
instance is replaced. A successful TCP connection is not proof of SSH login.

Prefer executing jobs from the local checkout through the lock-aware helper:

```sh
scripts/vast-sync.sh status
scripts/vast-sync.sh push experiments/my_run/bench.py  # explicitly include new files
scripts/vast-sync.sh run 0 python experiments/my_run/bench.py
scripts/vast-sync.sh run 1 bash -c 'python experiments/my_run/bench.py > /workspace/vast-results/my-run.log 2>&1'
scripts/vast-sync.sh pull
```

The helper activates the environment, selects the physical GPU and separates
native/Inductor/Triton/extension caches per GPU. Each job holds
`/workspace/locks/gpu0.lock` or `gpu1.lock` exclusively and
`/workspace/locks/source.lock` shared. A source push needs that source lock
exclusively. Thus two GPUs can run independently, but a push fails while either
managed job is active. Locks are advisory: direct SSH jobs must obey the same
protocol. Lock failure is not permission to bypass another session's job.
Check `nvidia-smi` before the first launch for older jobs that predate these locks.
If a command starts child workers, wait for them; do not daemonize inside `run`
and then release the lock while they are still using the GPU/source.

For long jobs use a persistent local terminal/tmux or a managed execution session
and record its handle and remote log path in your experiment README. Never kill
another session's process or perform a global cache cleanup. Shared environment
updates also require all jobs to finish; update dependency pins and run import/
execution checks afterwards. Device-specific caches are rebuilt, not copied.

Source rules:

1. Edit locally, inspect the diff, then push between jobs. `push` transfers tracked
   working-tree contents, not just HEAD; `source-commit.txt` alone does not prove
   the current uncommitted source. Record changed-file hashes/diffs for candidates.
2. Pass every required new untracked source file as an extra `push` argument.
   Do not pass secrets or generated binaries. Deleted files are intentionally not
   removed remotely; handle a required deletion explicitly after checking users.
3. Remote edits must be fetched using `fetch-source` and reviewed locally. This
   fetch covers `src/`; retrieve experimental files separately into a staging
   directory. Never blindly bidirectionally rsync the whole checkout.
4. Save each experiment in its own directory and results under
   `/workspace/vast-results/<experiment>/`. Pull copies them to
   `.bench/vast-20260927/results/<experiment>/`. Avoid reusing another run's names.
5. The remote checkout has no `.git`. All 11,543 original tracked blobs were
   checked against the local Git index; keep a per-experiment manifest for later
   working-tree edits. Do not confuse the archived training capsule's pinned
   engine with the installed current engine.

The existing result watcher uses `.bench/vast-20260927/sync-watch.pid`,
`sync-watch.lock`, and `sync-watch.log`. `watch` uses flock to prevent duplicates
and normally ends at 2026-09-28 00:00 Asia/Seoul for today's session. It pulls
results/setup only; source pushes are explicit. Starting it on another date uses
that date's midnight. For continued use beyond today, verify rental availability
and start the watcher for the newly authorized period.

NCU is intentionally omitted at the user's request. No host support request or
privilege escalation is required for ordinary optimization work. Actual timings,
CUDA Graph replay, PyTorch/CUPTI kernel traces and compute-sanitizer remain usable.
The large-D work is on branch `wip/main-20260928` under `experiments/trimul_large_d_vast/` (ported to `h100_wide_training.py`).
The subsequent Transition L384/768 x D128/256/384/512 coverage, PyTorch comparison,
and validation commands are in `experiments/transition_shapes_vast/README.md`.
That work fixed a D128 backward input-slot release race and uses the new
`_input_barrier_v1` extension identity. The final report documents the full-shape
checks and the narrower isolated-kernel racecheck scope at D384/512, where mixed
full-module race instrumentation raised internal sanitizer errors.

Large-D checkpoint portability note: the conda development libraries include
cuBLASLt 12.8.5 whereas the PyTorch cu128 wheel includes 12.8.4. A matching pip
version list does not prove which shared object is loaded. The large-D experiment
wrappers explicitly preload the two wheel cuBLAS libraries and record mapped
paths; this preserves the historical frozen Lt algorithm identities. Original
portability benchmark timings above used the initial environment. Avoid silently
mixing results from the two runtime selections.

Sanitizer detail discovered during large-D work: `compute-sanitizer` injection
can change preload ordering. For the explicit frozen Lt plans use
`experiments/trimul_large_d_vast/sanitize.sh`, which supplies separate
`--preload-library` flags and pins the experiment's ctypes Lt loader to the
wheel's absolute library path. The algorithm assertions stay active. Proposed
Conda downgrades were not applied; other sessions continue using the existing
shared environment. The two CUDA binary inspection tools (`cuobjdump`,
`nvdisasm`, version 12.8) were installed for the D256 SASS/register audit.

The large-D full-workload sanitizer checks passed with isolated Compute Sanitizer
2026.1.1.0 at `/workspace/tools/sanitizer132/bin/compute-sanitizer`. The old
2025.4.1 racecheck reports vendor-kernel hazards on both baseline and candidate;
retain those logs and use the documented newer tool. See the experiment RESULTS.md.

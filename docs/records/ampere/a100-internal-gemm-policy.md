# A100 internal GEMM policy and stalled build units

Date: 2026-09-12. Reported build: job 42700, A100 gpu05, source fb677741,
`/home/psk6950/miniworld-engine/.bench/a100-cache-20260911/source`.

## Diagnosis

The builder excludes top-level `impl=cute` using `build/gpu_to_kernels/sm_80.csv`.
A MiniWorld TriMul backward still calls `front_bwd_dW -> cute.dispatch.mm`.
Before this fix that dispatcher unconditionally warmed both cuBLAS and Quack.
Consequently the GPU policy did not exclude the nested unsupported backend.
The dispatch cache also used shape alone, ignoring device, dtype and operand strides.

The locally installed Quack `cache/jit.py` catches `RuntimeError` around both the
exclusive file lock and the compilation body, prints `lock timeout`, then calls
that compilation body again without the disk cache. The reported `Gemm Sm80 is
not implemented yet` can therefore produce that message without lock contention.
This verifies a misleading diagnostic and a bad backend entry path; it does **not**
locate the reported workers' precise futex/deadlock site.

## Changes

- Internal `mm`, `addmm`, `bmm` and the two composite `pick` callers use the same
  matrix policy before candidate warmup, import or cached-winner selection.
  Quack/CuTe candidates are excluded on SM80 and SM86. Supported cards can still
  calibrate them. Policy is a trace-time device constant under torch.compile.
- Dispatch keys include operand devices, dtypes, shapes, strides and candidate order.
- `build`, `bench_kernel` and `bench_module` accept `--unit-timeout-seconds`
  (finite positive wall seconds; default 7200). Compilation and measurement share
  the deadline. The default is a safety limit, not an estimated unit duration.
- A timeout kills the unit's process tree, records rc=124 / `timed_out=true`,
  releases only its claim and advances its GPU slot to the next queued unit.
  The traversal includes children launched from threads and separate `setsid`
  compiler sessions. It freezes parents and checks process start times; it never
  scans or signals all processes owned by the user or all workers on the node.
- Unit logs append attempt headers and PID/PGID diagnostics. Skip detection reads
  only the current attempt. Existing shard bytes are retained; a `.failed` sidecar
  prevents an old complete shard from satisfying resume after a failed rebuild.
  A later successful attempt removes that sidecar. There is no unbounded immediate
  retry loop: failed units are retried by a subsequent resume.
- A partial merge cannot turn a timeout into successful pre-bench build status.
  Main's existing protection against pruning artifacts after incomplete builds is
  preserved; the standalone fb677741 patch also prevents pruning after failed units.

## Validation and limits

CPU tests cover SM80/SM86/SM90/SM100 policy agreement, disabled calibration, stale
CuTe winners, layout/dtype keys, separate-session compiler descendants, claims,
partial shards, permanent skips and the next unit running on the same GPU slot.
GPU tests on an **allocated A6000** verify all three GEMM primitives' eager and
fullgraph compiled outputs/gradients with Quack replaced by a failing sentinel,
and masked unidirectional/bidirectional TriMul training at L=128 and L=384.
Final results:

- Main CPU suite: **2958 passed, 31 skipped** (219 GPU tests deselected).
- Standalone fb677741 patch, builder/autotune/skip tests: **1269 passed, 1 skipped**.
- Allocated A6000 GPU tests: **9 passed** (3 GEMM and 6 masked TriMul training cases).
- Ruff and ty: clean on both source trees.

Logs are retained under `.scratch/a100-policy-fix/`.

A100 job recovery has **not** been performed in this environment: job 42700 does
not exist on the connected Slurm controller, its gpu05 has A6000s, and the reported
source directory is absent. A100 live validation requires that cluster's SSH host.
No reported GPU 0/1 worker, claim or shard has been modified.

## Recovery on the actual A100 allocation

1. Inspect job ownership, worker PIDs/ancestry, CUDA visibility and current log
   timestamps on the actual cluster. Preserve a manifest of completed shard paths
   and checksums and the command lines for the two stalled units.
2. Keep healthy workers' source snapshot immutable. Prepare a patched source copy
   and separate recovery logs/shards. Do not cancel the whole Slurm job and do not
   use global `--reclaim` while any worker shares the original shard directory.
3. Terminate only the two verified stuck unit process trees. The old parent can
   otherwise dequeue more work from the unpatched source, so arrange its slot
   handoff before terminating a unit; do not assume killing a child pauses a slot.
4. On allocated A100 compute resources, validate that the reported TriMul and
   Pairformer backward paths cannot enter Quack. Retry the recorded unit arguments
   with patched source and a finite timeout. Keep healthy GPUs 0/1 running.
5. Preserve the original shards as evidence and merge compatible measurements
   with the normal provenance/source/config checks. A source-generation change
   does not authorize relabeling old claims/shards as new-source completions.
   Close the incident only after successful A100 retries and coverage validation.

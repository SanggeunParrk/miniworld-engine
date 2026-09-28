# A5000 cache qualification — 2026-09-13

Validated with capacity limits, not strict 100% coverage.

| Check | Result |
| --- | --- |
| Required keys | 1,002 / 1,008 usable |
| Invalid required caches | 0 |
| Alternative driver inputs | 772 / 772 measured |
| Real module replay | 22 evaluation/training cases, zero cache misses |
| GPU regressions | 229 passed, 5 skipped (two isolated suites) |
| FP32 numerical/determinism | 61 passed, 47 skipped |
| CPU contracts | 2,996 passed, 31 skipped |

The six missing keys belong to augmented-attention split/reduce backward at
A48, L4096 and L8192 (BF16/FP32 cores). `dq_expand` plus `dbias` alone requires
24 GiB and 96 GiB respectively, before other live tensors, exceeding this
23.56 GiB A5000. These are accepted capacity exclusions. CLI strict coverage
still exits 1; no smaller augmentation was substituted.

Replay covers token L384, augmented atom L1024, SWA atom L4096, both evaluation
and training, with production automatic dispatch. The JSON records each case.
The generic kernel checker additionally exercises shapes outside build-all's
production matrix; fallback warnings on those shapes are not missing matrix keys.

## Completion blockers repaired

The original finish job 1680369 exited 1 after its runtime replay falsely called
LayerNormLinear's cache stale. Capture stamps the declared configuration grid,
while the reader compared it with the shape/device-pruned subset. The reader now
validates against the declared grid, then intersects winners with legal tiles.
The proven A5000/A6000 unsafe tile remains excluded. All 22 module replays pass.

The first combined regression process ran out of memory in repeated
`torch._dynamo.explain` training forwards, followed by ten OOM-only failures.
Isolated replay established that these were not numerical mismatches.
The varying-depth test now uses fullgraph compilation with AOTAutograd, runs
backward between depths and verifies exactly four graphs for depths 1,2,3,4,2,1.
`explain()` resets Dynamo per invocation and could not verify that reuse.
The unchanged L384 depth schedule passes on A5000. Remaining GPU tests run in a
separate process and also pass. No tolerance was relaxed.

Earlier build fixes retain partial timings without treating failed units as
complete, disable dispatch calibration only inside build children, release failed
claims, stop on poisoned CUDA contexts and preserve compiled artifacts when
coverage remains incomplete. Original shards and JIT build artifacts are retained.

The four historical Xid31 incidents were not reproduced; their original cause
remains unconfirmed. See [MMU follow-up](a5000-mmu-followup.md) and
[memory-access investigation](a5000-memory-access-fix.md).

Raw logs: `.scratch/a5000-final-check-20260913/`. Build coverage:
`.scratch/a5000-finish/cache-audit.json`. Detailed evidence is in
[a5000-production-audit.json](a5000-production-audit.json).

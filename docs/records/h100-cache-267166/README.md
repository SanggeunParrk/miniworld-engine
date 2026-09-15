# Partial H100 cache from build 267166

This branch preserves the H100 measurements produced before the 24-hour build
limit, together with the H100 implementation and the update to remote main
`d2266a035de11384c46f8cc980e6460f60925413`. This is an incomplete cache build.

The original measurements were merged against matching source snapshot
`c65fe203`: **1,119 shards, 84 operations, 2,299 merged workload/bucket rows,
zero rejected shard/op records**. The 72 source-compatible runtime cache files are under
`src/miniworld_engine/autotune/data/*/NVIDIA H100 80GB HBM3 (sm90).json`.

After integrating remote main, the current source checks report **72 OK and
12 STALE** operations. All 12 stale files are native CuTe/CUDA operations:
their shared source/configuration fingerprint changed. Their measured identity
has been retained in [stale-native/](stale-native/) outside the runtime data
directory, ready for comparison or use with the original source snapshot.
See [status-after-update.json](status-after-update.json) for the full 84-operation
check before archiving those 12 files.
The final [runtime-status.json](runtime-status.json) confirms 72 OK runtime
operations, all matching the local compiler environment, and 12 archived native
operations. The archive and all 2,241 members passed SHA-256 verification.
An OK identity means the stored measurements match the current implementation;
it does not certify that every shape or configuration has been built.

## Build completion

| Phase | Planned units | Successful units | Failed units |
| --- | ---: | ---: | ---: |
| Module-driven | 411 | 403 | 8 |
| Direct kernel drivers | 2,048 | 674 | 33 |
| Total | 2,459 | 1,077 | 41 |

There are 1,341 units without a terminal result in the main build log. The 41
failures comprise two capacity/OOM cases, six attention illegal-access cases,
and 33 AdaLN misaligned-address cases. Partial units can contain useful earlier
measurements; they are not certified complete. The old shards predate explicit
`_unit_complete` markers. Do not infer completion merely from their existence.

CuTe contributed 213 successful direct-driver units across seven operations;
the five CUDA operations contributed 167. The seven CuTe operations include
LayerNormLinear forward, folded-stat forward and input gradient; transition
SwiGLU, gate backward and input gradient; and TriMul output projection/gate.

## Preserved evidence

- [merge.json](merge.json): original merge scope and rejection list.
- [manifest.json](manifest.json): archive/member SHA-256 checksums, sizes and
  final `ran` values where a unit log recorded one.
- `measurements.tar.gz`: unchanged shard JSON files and unit logs (2,241 files,
  about 47 MiB compressed). It includes unsuccessful/unfinished unit evidence.
- [build-output.txt](build-output.txt): complete main build log, including the
  Slurm time-limit termination on 2026-09-13 at 19:39:59.
- [environment.txt](environment.txt): the build's package versions.

To recover the original measurements, extract the archive into a new scratch
directory. Use the normal `dev merge --shards <directory> --gpu
"NVIDIA H100 80GB HBM3 (sm90)"` command in the recorded compiler environment;
its source/provenance checks still apply. Do not rewrite identities to make a
stale measurement appear current. The runtime cache holds ranked candidates;
the archive retains the full captured candidate timings for further analysis.

Triton compiler binaries, native object caches, Python environments and lock/
claim files are machine-local artifacts and are not included in this archive.

## Validation after merging remote main

CPU job 267620 ran `python -m pytest tests/ -m 'not gpu' -q` in the project
environment: **3,449 passed, 34 skipped, 225 deselected**, with no failures.
Ruff passed for the merge resolutions and adjusted tests; `git diff --check`
passed. CPU job 267621 confirmed the final runtime cache status above.
These checks do not replace GPU numerical, memory-safety or performance checks.

See [the memory investigation](../h100-memory-access-investigation.md) for the
OOM allocation analysis and the separately unresolved device-access errors.

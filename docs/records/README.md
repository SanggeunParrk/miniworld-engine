# Records

Dated findings. Each describes what was true when it ran and is **not** updated when the code
changes — the same rule as `src/miniworld_engine/kernels/<family>/notes/`, for the ones that belong
to no single kernel.

| | what it records |
|---|---|
| `naming-audit.md` | the defects found while renaming 111 kernels to `docs/kernels/naming.md`'s rules. The old names are its *subject*, so they stay. Current names: `registry.csv`; the mapping: `docs/kernels/rename-map.tsv`. |
| `tiling-audit.md` | one sweep of every kernel's tile axes. Kernel names are the ones `registry.csv` held at the time. |
| `pairformer-b200-latency.md` | Pairformer pair-track latency on B200 (sm100). |
| `pairformer-h100-latency.md` | the same on H100 (sm90). |
| `where-the-cache-build-spends-its-time-a6000.md` | the compile/bench split of an A6000 rebuild, and the three things that were idling: a second autotune key compiling on one core, a pool at 50% occupancy, and compile never overlapping bench. |
| `cache-coverage-replay-a6000.md` | the 363 lookups the module matrix asks for and the shipped cache does not serve, against a static coverage check that reports zero missing. Work list for the pending rebuild. |

The two latency files were under `benchmarks/runners/`, which `docs/benchmarks.md` forbids —
"do not add curated markdown reports under `benchmarks/`; write durable explanations under
`docs/`" — and nothing cited them. They are the only evidence in this repository of anything
running on sm90 or sm100 hardware, which `docs/supported.md` lists as never exercised here, so
they are kept rather than deleted. They are also not a substitute for a device manifest: no
`#provenance`, no commit, no way to know what code produced them.

The two audits sat in `docs/kernels/` while each opened by saying it was a record and not current
documentation. `docs/` is for pages written to be read as true now.

`cache-coverage-replay-a6000.md` is kept for a different reason than the others: it is not
superseded, it is *pending*. It is the first output of `dev audit --replay`, which had existed
with no caller, and it stays until a rebuilt cache makes it empty.

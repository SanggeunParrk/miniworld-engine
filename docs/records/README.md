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

The two latency files were under `benchmarks/runners/`, which `docs/benchmarks.md` forbids —
"do not add curated markdown reports under `benchmarks/`; write durable explanations under
`docs/`" — and nothing cited them. They are the only evidence in this repository of anything
running on sm90 or sm100 hardware, which `docs/supported.md` lists as never exercised here, so
they are kept rather than deleted. They are also not a substitute for a device manifest: no
`#provenance`, no commit, no way to know what code produced them.

The two audits sat in `docs/kernels/` while each opened by saying it was a record and not current
documentation. `docs/` is for pages written to be read as true now.

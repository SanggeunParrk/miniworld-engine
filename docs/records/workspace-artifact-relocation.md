# MiniWorld workspace cleanup — 2026-09-11

Moved 1,067 untracked MiniWorld work files/directories from the adjacent `team-gm`
repository into `.scratch/a6000-2026-09/` in this repository. The same-filesystem
renames preserve contents and inodes; no benchmark evidence was deleted.
Team-GM application code, environments, checkpoints, and existing changes remain in place.

The destination preserves the per-experiment folder names. Executable scripts (274)
now refer to the new root. Raw JSON, CSV and logs retain their original bytes, hashes,
and historical path provenance; resolve an old `team-gm/<entry>` artifact path by
substituting `.scratch/a6000-2026-09/<entry>` here. This does not change the source or
configuration identity under which historical measurements were made.

The complete movement inventory and script before/after hashes are in
[relocation-manifest.json](../../.scratch/a6000-2026-09/relocation-manifest.json).
The sweep page now lives only at [autotune-sweep-grid.html](../autotune-sweep-grid.html)
(and its archived shortcut); the Team-GM root shortcut was moved too.

Current generic atom DiT measurements are in `.scratch/a6000-2026-09/mw-atom-dit/`.
Reusable benchmark support is in `benchmarks/runners/bench.py`, target `dit_atom`.

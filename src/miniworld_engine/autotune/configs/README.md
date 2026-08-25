# Packaged config sets

The autotune search space that **ships inside the wheel**. `configs.default_config_dir()`
resolves `grid` from here when `MINIWORLD_CONFIG_DIR` is not set, which is what makes the
package usable without exporting an environment variable.

Only the default set lives here, and after `plan.md` P10 it lives here ONLY: `cli.resolve_config_dir`
now falls back to this directory when the repo root has no `configs/<name>`, so `grid` has a single
home instead of a repo-root copy kept byte-identical by a test. The repo root's `configs/` still
holds the A-B sets used during development (`blk16` … `blk128`, `warp4`, `warp8`, `mixed1`,
`mixed2`, `accuracy`); those are development inputs, not runtime data, and a consumer has no use
for them. A set that exists in both places resolves to the repo's — that is where an experiment
edits it.

`autotune/manifests/` is **not** a config set: it is the tracked per-GPU
record of which kernels each card was observed to run, read by `autotune/devices.py`, and it holds
one CSV per GPU rather than one per op. Passing it where a config set is expected fails at
`resolve_config_dir` with "3/91 ops covered", which is the right answer but a confusing one if this
paragraph does not exist.

`grid` must stay the set the shipped `data/**` caches were tuned over: the cache reader intersects a
stored entry against the live config list, so a narrower default would resolve every shipped entry
to nothing and silently re-tune on every call. `tests/test_default_config_set.py` asserts exactly
that against the committed caches, and also that there is only one copy of `grid`.

The generator writes here too: `src/miniworld_engine/tools/gen_shards.py --out
src/miniworld_engine/autotune/configs/grid`.

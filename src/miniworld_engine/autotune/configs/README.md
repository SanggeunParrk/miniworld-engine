# Packaged config sets

The autotune search space that **ships inside the wheel**. `configs.default_config_dir()`
resolves `grid` from here when `MINIWORLD_CONFIG_DIR` is not set, which is what makes the
package usable without exporting an environment variable.

Only the default set lives here. The repo root's `configs/` holds the A-B sets used during
development (`blk16` … `blk128`, `warp4`, `warp8`, `mixed1`, `mixed2`, `accuracy`); those are
development inputs, not runtime data, and a consumer has no use for them.

`configs/devices/` sits alongside them but is **not** a config set: it is the tracked per-GPU
record of which kernels each card was observed to run, read by `autotune/devices.py`, and it holds
one CSV per GPU rather than one per op. Passing it where a config set is expected fails at
`resolve_config_dir` with "3/91 ops covered", which is the right answer but a confusing one if this
paragraph does not exist.

`grid` must stay the set the shipped `data/**` caches were tuned over: the cache reader
intersects a stored entry against the live config list, so a narrower default would resolve
every shipped entry to nothing and silently re-tune on every call. `tests/test_default_config_set.py`
asserts exactly that against the committed caches.

`configs/grid/` also exists at the repo root, and that copy is not a leftover: `cli.resolve_config_dir`
maps a short config-set name to `repo/configs/<name>`, so in a source checkout every build, bench
and accuracy run resolves `grid` THERE, while a wheel install reaches this copy through
`default_config_dir()`. Two consumers, two paths. `tests/test_default_config_set.py` asserts the
two copies are byte-identical so they cannot drift. Collapsing them means teaching the resolver to
fall back here when the repo root has no such set — see `plan.md` P10.

# Packaged config sets

The autotune search space that **ships inside the wheel**. `configs.default_config_dir()`
resolves `grid` from here when `MINIWORLD_CONFIG_DIR` is not set, which is what makes the
package usable without exporting an environment variable.

Only the default set lives here. The repo root's `configs/` holds the A-B sets used during
development (`blk16` … `blk128`, `warp4`, `warp8`, `mixed1`, `mixed2`, `accuracy`, `devices`);
those are development inputs, not runtime data, and a consumer has no use for them.

`grid` must stay the set the shipped `data/**` caches were tuned over: the cache reader
intersects a stored entry against the live config list, so a narrower default would resolve
every shipped entry to nothing and silently re-tune on every call. `tests/test_default_config_set.py`
asserts exactly that against the committed caches.

While `configs/grid/` still exists at the repo root — every sweep launcher points
`MINIWORLD_CONFIG_DIR` at it — `tests/test_default_config_set.py` also asserts the two copies are
byte-identical, so they cannot drift. The root copy goes away once no running job depends on it.

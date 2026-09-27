# Packaged Triton config spaces

- `default/`: small per-kernel starting spaces, selected unless explicitly overridden.
- `grid/`: the complete declared global domains, selected with `build all grid`.
- `accuracy`, `blk*`, `warp*`, `mixed*`: historical development comparison sets.

Every op has a CSV in both default and grid. The global domains and the existing
resource/shape predictors remain intact. Explicit `MINIWORLD_CONFIG_DIR` selects
its directory before kernels import. Sharding and multiple-GPU global builds
remain supported. Defaults are not measurement results.

Runtime can reuse a compatible measured global winner outside the default search
space after source/environment/key, global-domain and resource validation. A miss
still searches the compact default set. Custom directories use only their own
declared candidates.

`dev/make_default_configs.py` generates the initial compact declarations. Review
per-kernel changes and retain measured fast shapes before promotion; generation
never rewrites grid or measured caches. See [v2.1 policy](../../../../docs/releases/2.1.0.md).

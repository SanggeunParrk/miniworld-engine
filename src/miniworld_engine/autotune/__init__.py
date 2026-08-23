"""Per-GPU autotune-config cache (backend-agnostic: Triton / CuTe / CUDA).

Ship the top-K tuned configs per ``(gpu, dtype, op, shape-bucket)`` so runs do not pay the
full-grid autotune tax and performance is reproducible across machines. Two runtime entry
points share one cache format and one storage layer:

* **Triton** — the candidate list comes from :func:`autotune.configs.configs_for`, i.e. from
  ``configs/<set>/<op>.csv``, and :func:`cache.install_cache_reader` narrows it to the cached
  top-K for the shape actually running. That reader is installed HERE, at package import, because
  it patches ``Autotuner.__init__`` and every kernel module imports this package before declaring
  an autotuner.
* **CuTe / CUDA** — these fix their tile/cluster/stage config at build time and have no autotune
  loop, so they call :func:`select_config` to *pick one* cached config (falling back to the
  kernel's own ``default_config`` on a miss).

Both warn ONCE on a miss (unknown GPU, unseen shape, or a stale cache — detected via
``config_space_hash``). A triton miss falls back to a BOUNDED heuristic subset
(``settings.autotune_miss_cap``, 24 by default), not to the 205,266-config grid; a cute miss
falls back to the kernel's ``default_config``. ``settings.configure(run_autotune=True)`` ignores
the cache and lifts the cap (full re-tune / no pin).

INVARIANT: config choice is PERFORMANCE-ONLY — every candidate config computes the same math —
so a missing / stale / wrong cache can only ever be slower, never incorrect.
"""

from __future__ import annotations

from .configs import configs_for, missing_ops, registered_ops, use_config_dir
from .cache import (
    as_cfg_dict,
    install_cache_reader,
    config_space_hash,
    gpu_key,
    key_bucket_of,
    operand_bytes,
    select_config,
    shape_bucket,
    store_ranked_configs,
    tensor_dtype_of,
)

__all__ = [
    "as_cfg_dict",
    "install_cache_reader",
    "configs_for",
    "missing_ops",
    "registered_ops",
    "use_config_dir",
    "config_space_hash",
    "gpu_key",
    "key_bucket_of",
    "operand_bytes",
    "select_config",
    "shape_bucket",
    "store_ranked_configs",
    "tensor_dtype_of",
]


# Must run before ANY kernel module constructs its Autotuner, which importing this package
# guarantees: a kernel reaches `configs_for` through `miniworld_engine.autotune`, so this line has
# already executed by the time its @triton.autotune decorator runs.
install_cache_reader()

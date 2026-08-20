"""Per-GPU autotune-config cache (backend-agnostic: Triton / CuTe / CUDA).

Ship the top-K tuned configs per ``(gpu, dtype, op, shape-bucket)`` so runs do not pay the
full-grid autotune tax and performance is reproducible across machines. Two runtime entry
points share one cache format and one storage layer:

* **Triton** — the candidate list comes from :func:`autotune.configs.configs_for`, i.e. from
  ``configs/<set>/<op>.csv``. Triton picks among those rows and nothing narrows them.
* **CuTe / CUDA** — these fix their tile/cluster/stage config at build time and have no autotune
  loop, so they call :func:`select_config` to *pick one* cached config (falling back to the
  kernel's own ``default_config`` on a miss).

Both warn ONCE on a miss (unknown GPU, unseen shape, or a stale cache — detected via
``config_space_hash``) and fall back to the full grid / default. ``MINIWORLD_RUN_AUTOTUNE=1``
ignores the cache (full re-tune / no pin).

INVARIANT: config choice is PERFORMANCE-ONLY — every candidate config computes the same math —
so a missing / stale / wrong cache can only ever be slower, never incorrect.
"""

from __future__ import annotations

from .buckets import elem_bucket_of
from .configs import configs_for, missing_ops, registered_ops, use_config_dir
from .cache import (
    as_cfg_dict,
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
    "configs_for",
    "missing_ops",
    "registered_ops",
    "use_config_dir",
    "config_space_hash",
    "elem_bucket_of",
    "gpu_key",
    "key_bucket_of",
    "operand_bytes",
    "select_config",
    "shape_bucket",
    "store_ranked_configs",
    "tensor_dtype_of",
]

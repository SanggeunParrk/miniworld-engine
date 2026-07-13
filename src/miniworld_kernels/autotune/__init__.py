"""Per-GPU Triton autotune-config cache.

Ship the top-K tuned Triton configs per ``(gpu, dtype, op, shape-bucket)`` so runs do not
pay the full-grid autotune tax and performance is reproducible across machines. At runtime a
Triton ``early_config_prune`` hook narrows a kernel's config grid to the cached top-K for the
running GPU/dtype/shape; on a miss (unknown GPU, unseen shape, or a stale cache) it warns once
and falls back to the full grid. ``MINIWORLD_RUN_AUTOTUNE=1`` ignores the cache (full re-tune).

INVARIANT: config choice is PERFORMANCE-ONLY — every config in a kernel's grid computes the
same math — so a missing / stale / wrong cache can only ever be slower, never incorrect.
"""

from __future__ import annotations

from .cache import (
    config_space_hash,
    gpu_key,
    key_bucket_of,
    make_cache_prune,
    run_autotune_enabled,
    shape_bucket,
    store_ranked_configs,
    tensor_dtype_of,
)

__all__ = [
    "config_space_hash",
    "gpu_key",
    "key_bucket_of",
    "make_cache_prune",
    "run_autotune_enabled",
    "shape_bucket",
    "store_ranked_configs",
    "tensor_dtype_of",
]

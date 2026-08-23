"""Per-GPU cache of the best LayerNorm backward path.

The three backward impls (atomic / partial / persistent) are all correct on any
CUDA arch, but *which* is fastest at a given (d, M) was only measured on H100. For
any other GPU we don't want to guess: instead, the first time a shape is seen on an
uncalibrated GPU we time the three paths on the real tensors, pick the winner, and
persist the choice keyed by GPU. Subsequent runs (and re-imports — this repo is
meant to be used as a submodule) read the cache and dispatch instantly.

The cache only ever selects among *correct* kernels, so a stale/corrupt cache can
never produce wrong numbers — at worst a suboptimal (but valid) path; on any error
we fall back to the static H100 heuristic.

The calibrated choices persist to the in-repo ``autotune/data/ln_bwd_dispatch/`` tree
(committed to git, shared across machines) — never to ``~/.cache`` or an env-var path,
so a stale per-user cache can never shadow the repo's committed choices.

Controls (``miniworld_engine.settings``; these were MINIWORLD_LN_AUTOTUNE / MINIWORLD_LN_BWD
until settings.py replaced every MINIWORLD_* switch with a field):
  layernorm_dispatch = "auto" (default) | "off" | "force"
      off   -> never calibrate, always static heuristic
      force -> calibrate even on known archs (H100), ignoring the static fast-path
  layernorm_bwd_path = "persistent" | "partial" | "atomic" | "cuda" | None
      a hard pin that bypasses both the cache and the heuristic
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import torch

_SUBDIR = "ln_bwd_dispatch"
# In-repo, committed cache root (sibling of the autotune-config data tree). Single
# canonical location: no env override, no ~/.cache — calibrated choices live in git.
_CACHE_ROOT = Path(__file__).resolve().parents[2] / "autotune" / "data"


def autotune_mode() -> str:
    from miniworld_engine import settings  # noqa: PLC0415
    return settings.current().layernorm_dispatch


def _cache_dir() -> Path:
    return _CACHE_ROOT / _SUBDIR


@functools.lru_cache(maxsize=8)
def gpu_key(device_index: int) -> str:
    """Stable per-GPU key: name + compute capability + triton version."""
    import triton

    name = torch.cuda.get_device_name(device_index)
    cc = torch.cuda.get_device_capability(device_index)
    raw = f"{name}_sm{cc[0]}{cc[1]}_triton{triton.__version__}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def mbucket(m: int) -> int:
    """Coarse M bucket (largest power of two <= M) so we calibrate once per scale."""
    return 1 << max(0, m.bit_length() - 1)


def _file(device_index: int) -> Path:
    return _cache_dir() / f"{gpu_key(device_index)}.json"


@functools.lru_cache(maxsize=8)
def _load(device_index: int) -> dict:
    """Load (and memoize) the on-disk cache for a device. Corrupt file -> empty."""
    fp = _file(device_index)
    try:
        return json.loads(fp.read_text())
    except (OSError, ValueError):
        return {}


def lookup(device: torch.device, n: int, mb: int) -> str | None:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    entry = _load(idx).get(f"{n}|{mb}")
    return entry["path"] if entry else None


def store(device: torch.device, n: int, mb: int, path: str, times_ms: dict[str, float]) -> None:
    """Persist a choice. Atomic write; merges with the in-memory + on-disk cache."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    data = _load(idx)
    data[f"{n}|{mb}"] = {"path": path, "ms": {k: round(v, 6) for k, v in times_ms.items()}}
    try:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        fp = _file(idx)
        tmp = fp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(fp)
    except OSError:
        pass  # read-only fs etc. — keep the in-memory choice, just don't persist
    _load.cache_clear()
    _load(idx).update(data)  # refresh memoized copy

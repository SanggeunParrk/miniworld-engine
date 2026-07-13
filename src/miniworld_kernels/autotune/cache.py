"""Per-GPU Triton autotune-config cache — see package docstring.

Two cache roots, runtime-preferred:
  * shipped   : ``src/miniworld_kernels/autotune/data/<op>/<gpu_key>.json`` (committed defaults)
  * runtime   : ``<cache-root>/autotune/<op>/<gpu_key>.json`` where cache-root is
                ``$MINIWORLD_KERNELS_CACHE_DIR`` | ``$XDG_CACHE_HOME/miniworld_kernels`` |
                ``~/.cache/miniworld_kernels`` (written by the builder / RUN_AUTOTUNE regen).

Cache JSON schema (v1)::

    {"schema": 1, "gpu": "<gpu_key>", "op": "<op>",
     "config_space_hash": "<12-hex>",          # hash of the kernel's FULL config grid
     "provenance": {"triton": "...", "torch": "...", "built_utc": "..."},
     "entries": {"<dtype>|<bucket>": [{"kwargs": {...}, "num_warps": N, "num_stages": N,
                                       "ms": <median>}, ... top-K]}}

``config_space_hash`` invalidates an entry when the kernel's grid changes, so a stale cache
degrades to a warn + full-grid fallback instead of silently pinning old tiles.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import torch

SCHEMA = 1
_SHIPPED_ROOT = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# keys / roots
# --------------------------------------------------------------------------- #
def run_autotune_enabled() -> bool:
    """``MINIWORLD_RUN_AUTOTUNE=1`` -> ignore the cache and run the full autotune grid."""
    return os.getenv("MINIWORLD_RUN_AUTOTUNE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _runtime_root() -> Path:
    base = os.environ.get("MINIWORLD_KERNELS_CACHE_DIR")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "miniworld_kernels")
    return Path(base) / "autotune"


def gpu_key(device_index: int | None = None) -> str:
    """Stable per-GPU key: name + compute capability, e.g. ``NVIDIA A100 80GB PCIe (sm80)``."""
    if not torch.cuda.is_available():
        return "cpu"
    if device_index is None:
        device_index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_index)
    cc = torch.cuda.get_device_capability(device_index)
    return f"{name} (sm{cc[0]}{cc[1]})".replace("/", "_")


def shape_bucket(**dims: int) -> str:
    """Canonical shape-bucket string from the size-defining dims, e.g. ``H2=1024`` or
    ``K=512,ND=2048``. Callers pass the dims that actually change the best config (NOT the
    batched/row count M, which is handled by tile-M autotune); keep it small + stable."""
    return ",".join(f"{k}={int(v)}" for k, v in sorted(dims.items()))


# --------------------------------------------------------------------------- #
# config (de)serialization + config-space hash
# --------------------------------------------------------------------------- #
def _sig(config) -> tuple:
    """Hashable signature of a triton.Config: (sorted kwargs, num_warps, num_stages)."""
    return (tuple(sorted(config.kwargs.items())), int(config.num_warps), int(config.num_stages))


def _sig_from_dict(d: dict) -> tuple:
    return (tuple(sorted(d["kwargs"].items())), int(d["num_warps"]), int(d["num_stages"]))


def config_to_dict(config, ms: float | None = None) -> dict:
    d = {"kwargs": dict(sorted(config.kwargs.items())),
         "num_warps": int(config.num_warps), "num_stages": int(config.num_stages)}
    if ms is not None:
        d["ms"] = float(ms)
    return d


def config_space_hash(configs) -> str:
    """12-hex hash of the kernel's full config grid; changes iff the grid changes."""
    sigs = sorted(_sig(c) for c in configs)
    return hashlib.sha1(repr(sigs).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# load / store
# --------------------------------------------------------------------------- #
_load_cache: dict[tuple[str, str], dict | None] = {}


def _load(op: str, gk: str) -> dict | None:
    """Load the cache for (op, gpu). Runtime root wins over shipped. Memoized; corrupt -> None."""
    key = (op, gk)
    if key in _load_cache:
        return _load_cache[key]
    result = None
    for root in (_runtime_root(), _SHIPPED_ROOT):
        fp = root / op / f"{gk}.json"
        if fp.exists():
            try:
                result = json.loads(fp.read_text())
                break
            except Exception:  # noqa: BLE001 -- corrupt cache -> treat as miss
                result = None
    _load_cache[key] = result
    return result


def store_ranked_configs(
    op: str, gk: str, dtype: str, bucket: str, ranked: list[tuple[object, float]],
    config_space_h: str, *, top_k: int = 5,
) -> Path:
    """Persist the top-K (config, ms) for (op, gpu, dtype, bucket) to the RUNTIME cache.

    ``ranked`` is a list of ``(triton.Config, median_ms)`` sorted fastest-first. Resets the
    file's entries if the config-space hash changed (kernel grid was edited)."""
    fp = _runtime_root() / op / f"{gk}.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    data = None
    if fp.exists():
        try:
            data = json.loads(fp.read_text())
        except Exception:  # noqa: BLE001
            data = None
    if data is None or data.get("config_space_hash") != config_space_h:
        import datetime as _dt  # noqa: PLC0415 -- stamp only when writing
        try:
            import triton as _triton  # noqa: PLC0415
            triton_ver = getattr(_triton, "__version__", "?")
        except Exception:  # noqa: BLE001
            triton_ver = "?"
        data = {
            "schema": SCHEMA, "gpu": gk, "op": op, "config_space_hash": config_space_h,
            "provenance": {"triton": triton_ver, "torch": torch.__version__,
                           "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")},
            "entries": {},
        }
    data["entries"][f"{dtype}|{bucket}"] = [config_to_dict(c, ms) for c, ms in ranked[:top_k]]
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(fp)
    _load_cache.pop((op, gk), None)  # invalidate memo
    return fp


# --------------------------------------------------------------------------- #
# runtime prune hook
# --------------------------------------------------------------------------- #
_warned: set[tuple] = set()


def _warn_once(op: str, gk: str, tag: str, reason: str) -> None:
    key = (op, gk, tag, reason)
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(
        f"[miniworld.autotune] {reason} for op '{op}' on '{gk}' ({tag}). Falling back to the "
        f"full autotune grid — this run may be slower and the chosen config may be suboptimal. "
        f"Build a tuned cache for this GPU with the autotune cache-builder "
        f"(see docs/operations/dispatch-cache.md).",
        stacklevel=4,
    )


def _get(named_args, kwargs, name):
    if name in named_args:
        return named_args[name]
    return kwargs.get(name)


def make_cache_prune(op: str, *, dtype_of, bucket_of, base_prune=None):
    """Build a Triton ``early_config_prune`` callback that narrows the grid to the cached
    top-K for the running (gpu, dtype, shape-bucket).

    ``dtype_of(named_args, kwargs) -> str`` and ``bucket_of(named_args, kwargs) -> str`` extract
    the cache sub-keys from the kernel's runtime args. ``base_prune`` (optional) is another
    ``early_config_prune`` (e.g. a device-smem filter) run FIRST for safety — the cache only
    ever narrows within what the base prune already deemed launchable.
    """

    def prune(configs, named_args, **kwargs):
        base = list(base_prune(configs, named_args, **kwargs)) if base_prune else list(configs)
        if run_autotune_enabled() or not base:
            return base
        try:
            gk = gpu_key()
            dtype = str(dtype_of(named_args, kwargs))
            bucket = str(bucket_of(named_args, kwargs))
        except Exception:  # noqa: BLE001 -- never let cache lookup break a kernel launch
            return base
        data = _load(op, gk)
        if data is None:
            _warn_once(op, gk, dtype, "no tuned autotune cache")
            return base
        if data.get("config_space_hash") != config_space_hash(configs):
            _warn_once(op, gk, dtype, "tuned autotune cache is STALE (kernel config grid changed)")
            return base
        entry = data.get("entries", {}).get(f"{dtype}|{bucket}")
        if not entry:
            _warn_once(op, gk, f"{dtype}|{bucket}", "no tuned autotune cache entry for this shape")
            return base
        want = {_sig_from_dict(e) for e in entry}
        kept = [c for c in base if _sig(c) in want]
        return kept or base

    prune._miniworld_op = op  # noqa: SLF001 -- introspection tag (which op a kernel tunes as)
    return prune

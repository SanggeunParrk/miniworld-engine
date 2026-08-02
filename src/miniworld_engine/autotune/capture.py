"""Generic autotune-cache BUILDER via Triton-autotuner instrumentation.

Instead of hand-replicating each kernel's launch in a per-kernel builder (fragile, and how the
int32-offset bug slipped in), this patches ``triton.runtime.autotuner.Autotuner._bench`` to
record every (config -> measured ms) as it is benchmarked during a REAL module forward/backward
run. Each timing is keyed by the SAME ``(op, dtype, shape-bucket)`` the runtime cache-prune uses
(read off the kernel's ``make_cache_prune`` hook via its ``_miniworld_op`` /
``_miniworld_dtype_of`` / ``_miniworld_bucket_of`` tags), so a built cache is guaranteed to hit.

Usage (on the target GPU, with the full grid unlocked so every config is benched):

    MINIWORLD_RUN_AUTOTUNE=1
    from miniworld_engine.autotune import capture
    capture.install()
    ... run each wired module fwd+bwd across representative shapes ...
    capture.flush(top_k=5)     # writes <cache-root>/autotune/<op>/<gpu>.json for every op seen

Any wired kernel that fires during the run is captured automatically — no per-kernel code. The
cross-check against the hand-built pilot caches (transition_split_fwd / trimul_bidir_front)
validates the capture path. Config choice is performance-only, so this never affects numerics.
"""

from __future__ import annotations

from .cache import (
    as_cfg_dict,
    config_space_hash,
    gpu_key,
    store_ranked_configs,
)

# op -> {"grid": [configs] | None, "entries": {(dtype, bucket): {sig: (config, ms)}}}
_CAPTURE: dict = {}
_orig_bench = None


def _median(t) -> float:
    """do_bench(quantiles=(0.5,0.2,0.8)) returns [median, q20, q80]; be tolerant of a scalar."""
    if isinstance(t, (list, tuple)):
        return float(t[0])
    return float(t)


def _sig(config) -> tuple:
    d = as_cfg_dict(config)
    kw = tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                      for k, v in d["kwargs"].items()))
    return (kw, d["num_warps"], d["num_stages"])


def _record_one(autotuner, config, meta, ms) -> None:
    ecp = getattr(autotuner, "early_config_prune", None)
    op = getattr(ecp, "_miniworld_op", None)
    if not op:
        return
    if ms == float("inf"):
        return
    nargs = getattr(autotuner, "nargs", None) or {}
    dtype = str(ecp._miniworld_dtype_of(nargs, meta))   # noqa: SLF001
    bucket = str(ecp._miniworld_bucket_of(nargs, meta))  # noqa: SLF001
    slot = _CAPTURE.setdefault(op, {"grid": None, "entries": {}})
    if slot["grid"] is None:
        slot["grid"] = list(autotuner.configs)
    ent = slot["entries"].setdefault((dtype, bucket), {})
    sig = _sig(config)
    prev = ent.get(sig)
    if prev is None or ms < prev[1]:   # keep the fastest reading for this config
        ent[sig] = (config, ms)


def install() -> None:
    """Patch Autotuner._bench to capture per-config timings. Idempotent."""
    global _orig_bench
    if _orig_bench is not None:
        return
    from triton.runtime.autotuner import Autotuner

    _orig_bench = Autotuner._bench

    def _bench(self, *args, config, **meta):
        res = _orig_bench(self, *args, config=config, **meta)
        try:
            _record_one(self, config, meta, _median(res))
        except Exception:  # noqa: BLE001 -- capture must never perturb a real bench
            pass
        return res

    Autotuner._bench = _bench


def uninstall() -> None:
    global _orig_bench
    if _orig_bench is not None:
        from triton.runtime.autotuner import Autotuner
        Autotuner._bench = _orig_bench
        _orig_bench = None


def reset() -> None:
    _CAPTURE.clear()


def flush(top_k: int = 5, gpu: str | None = None) -> list:
    """Write the captured top-K per (op, dtype, bucket) to the runtime cache. Returns a list of
    (op, dtype, bucket, n_configs, path) for logging."""
    gk = gpu or gpu_key()
    written = []
    for op, slot in sorted(_CAPTURE.items()):
        grid = slot["grid"] or []
        if not grid:
            continue
        csh = config_space_hash(grid)
        for (dtype, bucket), ent in sorted(slot["entries"].items()):
            ranked = sorted(ent.values(), key=lambda cm: cm[1])   # (config, ms) fastest-first
            fp = store_ranked_configs(op, gk, dtype, bucket, ranked, csh, top_k=top_k)
            written.append((op, dtype, bucket, len(ranked), str(fp)))
    return written


def summary() -> str:
    lines = []
    for op, slot in sorted(_CAPTURE.items()):
        n_buckets = len(slot["entries"])
        n_grid = len(slot["grid"] or [])
        lines.append(f"  {op}: grid={n_grid} buckets={n_buckets}")
        for (dtype, bucket), ent in sorted(slot["entries"].items()):
            best = min(ent.values(), key=lambda cm: cm[1]) if ent else None
            tag = f"{dtype}|{bucket}"
            if best:
                lines.append(f"    {tag}: {len(ent)} configs, best {best[1]:.4f}ms "
                             f"{as_cfg_dict(best[0])['kwargs']}")
    return "\n".join(lines) if lines else "  (nothing captured)"

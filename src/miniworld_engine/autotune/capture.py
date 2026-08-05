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

from pathlib import Path

from .cache import (
    as_cfg_dict,
    config_space_hash,
    config_to_dict,
    gpu_key,
    store_ranked_configs,
)
from .cache import _sig_from_dict as _sig_from_dict

# op -> {"grid": [configs] | None, "entries": {(dtype, bucket): {sig: (config, ms)}}}
_CAPTURE: dict = {}
_orig_bench = None
_orig_compile = None

# Wall-clock budget (seconds) for compiling a SINGLE config during a build, enforced by forking the
# whole triton compile into a child and SIGKILLing it on overrun (see install()). A healthy config
# compiles in a few seconds; a register-spill config runs 10-20 min across make_llir (libLLVM) and
# make_cubin (ptxas). This is the SOLE compile-time guard and is arch-agnostic BY CONSTRUCTION: it
# judges each config by how long it actually takes to compile on THIS GPU, so every arch keeps only
# the configs that compile in time on it — no static num_warps/num_stages heuristic (which would be
# tuned to one arch). This is the arch-specific BACKSTOP; a static ``_drop_compile_monsters`` prune
# in cache.py already removes the always-bad ``num_warps>=16`` / ``num_stages>=5`` tail up front, so
# this only has to catch a given arch's residual oddball blowups. The full grid /
# ``config_space_hash`` is untouched, so all caches stay valid. 60s cleanly separates the two
# populations: a healthy config (even at large D) compiles in a few to a few-tens of seconds, while
# a register-spill monster runs 10-20 MINUTES — so 60s never kills a usable config yet kills fast.
_COMPILE_BUDGET_S = 60


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
    """Patch Autotuner._bench (capture timings) and triton.compile (bound compile time). Idempotent.

    A register-spill config makes triton's compile pipeline run for 10-20 MINUTES — split across
    ``make_llir`` (in-process libLLVM) and ``make_cubin`` (ptxas subprocess). NEITHER can be bounded
    by a Python SIGALRM (both block in native code; PEP 475 restarts the syscall — confirmed by two
    py-spy stacks), and a subprocess timeout would only catch the ptxas half. The only guard that
    covers BOTH halves and is arch-agnostic is to run the WHOLE compile in a forked child bounded by
    wall-clock: the child compiles into triton's on-disk cache and exits; the parent waits up to
    ``_COMPILE_BUDGET_S`` and SIGKILLs the child on overrun (a hard kill of a process, immune to PEP
    475). On success the parent re-calls compile — now an instant disk-cache hit. On overrun/failure
    it raises, so ``_bench`` records +inf and the config is skipped. Each GPU thus keeps exactly the
    configs that compile in time on IT — no static num_warps/num_stages heuristic (which would be
    tuned to one arch). The child does a PURE ``compile`` (never ``warmup``), so it touches neither
    torch nor the CUDA driver and is safe to fork after the parent has initialized CUDA.
    """
    global _orig_bench, _orig_compile
    if _orig_bench is not None:
        return
    import os  # noqa: PLC0415
    import signal  # noqa: PLC0415
    import time  # noqa: PLC0415
    import triton.compiler as _tc  # noqa: PLC0415
    import triton.compiler.compiler as _tcc  # noqa: PLC0415
    from triton.runtime.autotuner import Autotuner

    _orig_compile = _tcc.compile

    def _fork_compile(*args, **kwargs):  # noqa: ANN002, ANN003
        if _COMPILE_BUDGET_S <= 0:
            return _orig_compile(*args, **kwargs)
        pid = os.fork()
        if pid == 0:  # child: compile into the on-disk cache, no torch/CUDA touched, then exit
            try:
                _orig_compile(*args, **kwargs)
                os._exit(0)
            except BaseException:  # noqa: BLE001 -- any failure -> parent treats config as unusable
                os._exit(3)
        deadline = time.monotonic() + _COMPILE_BUDGET_S
        while True:
            w, st = os.waitpid(pid, os.WNOHANG)
            if w == pid:
                break
            if time.monotonic() > deadline:
                try:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                except OSError:
                    pass
                raise RuntimeError(
                    f"triton compile exceeded {_COMPILE_BUDGET_S}s (register-spill config); skipped")
            time.sleep(0.05)
        if os.waitstatus_to_exitcode(st) != 0:
            raise RuntimeError("triton compile failed in isolated child; config skipped")
        return _orig_compile(*args, **kwargs)  # disk-cache hit -> instant

    _tcc.compile = _fork_compile
    _tc.compile = _fork_compile  # the name create_binder rebinds via `from ..compiler import compile`

    _orig_bench = Autotuner._bench

    def _bench(self, *args, config, **meta):
        try:
            res = _orig_bench(self, *args, config=config, **meta)
        except Exception:  # noqa: BLE001 -- a config that fails to compile/run simply loses
            res = float("inf")
        try:
            _record_one(self, config, meta, _median(res))
        except Exception:  # noqa: BLE001 -- capture must never perturb a real bench
            pass
        return res

    Autotuner._bench = _bench


def uninstall() -> None:
    global _orig_bench, _orig_compile
    if _orig_bench is not None:
        from triton.runtime.autotuner import Autotuner
        Autotuner._bench = _orig_bench
        _orig_bench = None
    if _orig_compile is not None:
        import triton.compiler as _tc  # noqa: PLC0415
        import triton.compiler.compiler as _tcc  # noqa: PLC0415
        _tcc.compile = _orig_compile
        _tc.compile = _orig_compile
        _orig_compile = None


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


def dump_shard(path: str) -> int:
    """Serialize this process's captured timings to a standalone JSON SHARD file (NOT the
    in-repo cache). Parallel capture jobs each ``dump_shard`` to their OWN file; a single
    ``merge_shards`` writer folds them into the committed cache — so no env var and no
    concurrent writers ever touch the in-repo tree. Returns the number of ops dumped."""
    import json  # noqa: PLC0415
    out: dict = {}
    for op, slot in _CAPTURE.items():
        grid = slot["grid"] or []
        entries = {f"{d}|{b}": [config_to_dict(c, ms) for c, ms in ent.values()]
                   for (d, b), ent in slot["entries"].items()}
        out[op] = {"grid": [config_to_dict(c) for c in grid], "entries": entries}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return len(out)


def merge_shards(shard_paths, top_k: int = 5, gpu: str | None = None, only_ops=None) -> list:
    """Fold shard files (from ``dump_shard``) into the in-repo cache as the SOLE writer.

    Unions buckets across shards, keeping the fastest reading per config. ``only_ops`` (a set)
    restricts the write to those op names — so a targeted build never rewrites an op that already
    has a good cache. Returns ``(op, dtype|bucket, n_configs, path)`` rows for logging."""
    import json  # noqa: PLC0415
    gk = gpu or gpu_key()
    agg: dict = {}
    for sp in shard_paths:
        try:
            d = json.loads(Path(sp).read_text())
        except Exception:  # noqa: BLE001 -- skip unreadable/partial shard
            continue
        for op, slot in d.items():
            if only_ops is not None and op not in only_ops:
                continue
            a = agg.setdefault(op, {"grid": slot.get("grid", []), "entries": {}})
            if not a["grid"]:
                a["grid"] = slot.get("grid", [])
            for bk, lst in slot.get("entries", {}).items():
                ent = a["entries"].setdefault(bk, {})
                for cd in lst:
                    sig = _sig_from_dict(cd)
                    ms = float(cd.get("ms", float("inf")))
                    prev = ent.get(sig)
                    if prev is None or ms < prev[1]:
                        ent[sig] = (cd, ms)
    written = []
    for op, a in sorted(agg.items()):
        grid = a["grid"]
        if not grid:
            continue
        csh = config_space_hash(grid)
        for bk, ent in sorted(a["entries"].items()):
            dtype, bucket = bk.split("|", 1)
            ranked = sorted(ent.values(), key=lambda cm: cm[1])
            fp = store_ranked_configs(op, gk, dtype, bucket, list(ranked), csh, top_k=top_k)
            written.append((op, bk, len(ranked), str(fp)))
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

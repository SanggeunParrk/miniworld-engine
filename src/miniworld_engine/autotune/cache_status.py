"""Up-front staleness scan for the committed autotune caches -- a ``git status`` for the tuned
``data/<op>/<gpu>.json`` files.

The runtime reader (``cache._cached_subset``) already invalidates a cache four ways -- the config
grid changed (``config_space_hash``), the bucket keys mean something else (``key_scheme``), the
toolchain moved (``env_identity``), or the kernel source / ``key=[...]`` changed (``op_identity``)
-- but it only surfaces that as a per-launch WARNING, i.e. AFTER a benchmark has already measured
the heuristic fallback and mislabelled it as the kernel. This module recomputes the two
CODE-DRIVEN fingerprints (grid + key-scheme) without a GPU or a kernel launch, so a stale cache is
caught before it is trusted -- run it before a benchmark, and enforce it in CI so a grid edit that
was not followed by a rebuild fails the commit.

``env_identity`` is reported too but kept SEPARATE: it is machine-specific (a cache built under one
triton/cuda is legitimately stale under another), so it is not a "the committed cache is wrong"
signal the way a source change is -- CI must not fail on it. ``op_identity`` needs the live
autotuner object (the JIT source), so it stays a runtime-only check; the grid hash already catches
the common regression (a narrowed ladder), which is what this scanner is for.
"""
from __future__ import annotations

import csv
import functools
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from miniworld_engine.autotune.cache import (
    _stored_rev,
    build_rev,
    DRIVER_ID_SCHEME,
    _scheme_stale,
    config_space_hash,
    driver_identity,
    env_identity,
    op_identity,
)
from miniworld_engine.autotune.configs import configs_for

_DATA = Path(__file__).resolve().parent / "data"
_REGISTRY = Path(__file__).resolve().parent.parent / "kernels" / "registry.csv"


@functools.lru_cache(maxsize=1)
def _registry_symbols() -> dict[str, tuple[str, str]]:
    """op name -> (import module, symbol) from registry.csv, so an op's live autotuner (the
    JIT-source fingerprint ``op_identity`` reads) can be resolved WITHOUT a GPU -- importing a
    ``@triton.autotune`` kernel only defines it, it does not launch."""
    out: dict[str, tuple[str, str]] = {}
    with _REGISTRY.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            f, sym = row.get("file", ""), row.get("symbol", "")
            if f.endswith(".py") and sym:
                mod = f[:-3].replace("/", ".")
                out[row["kernel"]] = (mod, sym)
    return out


def _current_op_identity(op: str) -> str | None:
    """The live source/key fingerprint for ``op``, or None if it has no registry autotuner (a
    dispatch-only cache) or cannot be imported on this machine."""
    sym = _registry_symbols().get(op)
    if sym is None:
        return None
    mod_name, attr = sym
    try:
        autotuner = getattr(importlib.import_module(mod_name), attr)
        return op_identity(autotuner)
    except Exception:
        return None


@dataclass(frozen=True)
class CacheStatus:
    op: str
    gpu: str
    #: "OK" | "STALE" | "UNKNOWN" -- code-driven verdict (grid + key-scheme), the CI signal.
    verdict: str
    reason: str
    #: True/False/None: does the toolchain fingerprint match THIS machine (None = no env stored).
    #: Machine-specific, so it is NOT folded into ``verdict``.
    env_matches: bool | None

    @property
    def stale(self) -> bool:
        return self.verdict == "STALE"


def scan(gpu_substr: str | None = None) -> list[CacheStatus]:
    """Recompute the code-driven fingerprints for every committed cache JSON and diff them against
    what the file stored. ``gpu_substr`` filters the GPU key (e.g. ``"A100"``). CPU-only."""
    cur_env = env_identity()
    out: list[CacheStatus] = []
    for op_dir in sorted(p for p in _DATA.iterdir() if p.is_dir()):
        op = op_dir.name
        for jf in sorted(op_dir.glob("*.json")):
            gpu = jf.stem  # "NVIDIA A100 80GB PCIe (sm80)"
            if gpu_substr and gpu_substr not in gpu:
                continue
            try:
                data = json.loads(jf.read_text())
            except (OSError, json.JSONDecodeError):
                out.append(CacheStatus(op, gpu, "UNKNOWN", "unreadable JSON", None))
                continue
            verdict, reason = "OK", ""
            # A RUNTIME-DISPATCH cache (ln_bwd_dispatch / bias_only_dispatch) is a different file
            # shape entirely -- a flat {bucket: choice} map with no fingerprints -- and it is not
            # produced from a config grid, so none of the checks below apply to it. Detect it by
            # STRUCTURE (no config_space_hash key) and skip before touching `configs_for`: that
            # call REGISTERS the name it is given, and a name with no CSV lands in the config
            # module's `_STRANDED` set, which makes a later `use_config_dir()` raise -- poisoning
            # any process that scans and then builds.
            if "config_space_hash" not in data:
                out.append(CacheStatus(op, gpu, "UNKNOWN", "runtime-dispatch cache (no config grid)",
                                       None))
                continue
            try:
                cur_grid = config_space_hash(configs_for(op))
            except Exception:
                cur_grid = None
            if cur_grid is None:
                verdict, reason = "UNKNOWN", "no config grid for this op"
            elif _stored_rev(data) != build_rev(op):
                verdict, reason = "STALE", f"build_rev declared {build_rev(op)} (registry.csv)"
            elif _scheme_stale(op, data.get("key_scheme")):
                verdict, reason = "STALE", "bucket key scheme changed (cache.KEY_SCHEME)"
            elif data.get("config_space_hash") != cur_grid:
                # NOT stale: a grid edit is served incrementally (`cache.configs_to_bench` benches
                # what the grid ADDED, `store_ranked_configs` drops what it removed). Reported so a
                # human can see the op owes a top-up build, but it never fails CI -- charging a
                # full rebuild for a narrowing is what made ladder maintenance cost 25 GPU-hours.
                verdict, reason = "OK", "config grid changed -- incremental build pending"
            else:
                # Kernel SOURCE / key=[...] fingerprint -- the "kernel commit hash". Editing the
                # @triton.jit body or its key re-partitions or invalidates the tuned entries even
                # when the config grid is untouched, so a cache built before the edit is stale.
                # Only flagged when BOTH sides are known: a cache from before op_identity was
                # recorded stores None, and not every op imports on every machine.
                cur_src = _current_op_identity(op)
                stored_src = data.get("op_identity")
                # Scheme-gated: a stamp written by a DIFFERENT version of the hashing scope is
                # not comparable, and treating the disagreement as drift is a false STALE nothing
                # can clear. Absent/older scheme -> skip, exactly as an absent hash does.
                cur_drv = driver_identity(op)
                stored_drv = data.get("driver_identity")
                if data.get("driver_id_scheme") != DRIVER_ID_SCHEME:
                    stored_drv = None
                # `op_identity` STALE, `driver_identity` reported. The two look alike and are not:
                # a different kernel body makes a recorded time a claim about code that no longer
                # exists, while a different BUILD DRIVER only changes which buckets got built --
                # resetting on the latter deleted 32 of 38 tuned buckets from
                # cond_transition_expand_swiglu for an edit that said nothing against them.
                if cur_src is not None and stored_src is not None and cur_src != stored_src:
                    verdict, reason = "STALE", "kernel source/key changed (op_identity)"
                elif cur_drv is not None and stored_drv is not None and cur_drv != stored_drv:
                    verdict, reason = "OK", "build driver changed -- coverage may differ; rebuild the op"
            env_stored = data.get("env_identity")
            env_matches = None if env_stored is None else (env_stored == cur_env)
            out.append(CacheStatus(op, gpu, verdict, reason, env_matches))
    return out


def format_report(rows: list[CacheStatus], *, gpu_substr: str | None = None) -> str:
    stale = [r for r in rows if r.verdict == "STALE"]
    unknown = [r for r in rows if r.verdict == "UNKNOWN"]
    ok = [r for r in rows if r.verdict == "OK"]
    env_mismatch = [r for r in rows if r.env_matches is False]
    lines = []
    scope = f" (gpu~={gpu_substr!r})" if gpu_substr else ""
    lines.append(f"autotune cache status{scope}: "
                 f"{len(ok)} OK, {len(stale)} STALE, {len(unknown)} UNKNOWN "
                 f"({len(rows)} caches)")
    if stale:
        lines.append("\nSTALE -- rebuild before trusting (build the op, then `dev merge`):")
        for r in stale:
            lines.append(f"  {r.op:44} {r.gpu:32} {r.reason}")
    if env_mismatch:
        lines.append(f"\nENV mismatch on this machine ({len(env_mismatch)}) -- built under a "
                     "different triton/cuda; a rebuild HERE would relabel, not necessarily change:")
        for r in sorted({(r.op, r.gpu) for r in env_mismatch}):
            lines.append(f"  {r[0]:44} {r[1]}")
    if unknown:
        lines.append(f"\nUNKNOWN ({len(unknown)}): no config grid (dispatch-only caches, expected).")
    if not stale:
        lines.append("\nNo grid/scheme-stale caches. ✓")
    return "\n".join(lines)

"""Choose module invocations that cover the missing kernel keys of a verified derivation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from miniworld_engine._atomic import write_json

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "kernels" / "registry_kernel.units.json"


def source_identity() -> str:
    """Hash dispatch inputs; tuning results and page edits do not invalidate a derivation."""
    files = [ROOT / "settings.py", ROOT / "kernels" / "registry_module.csv",
             ROOT / "kernels" / "registry.csv"]
    for directory in (ROOT / "modules", ROOT / "kernels"):
        files.extend(path for path in directory.rglob("*.py")
                     if "notes" not in path.relative_to(directory).parts)
    files.extend((ROOT / "build" / "gpu_to_kernels").glob("*.csv"))
    files.extend(ROOT / "autotune" / name for name in                 ("builder.py", "checkpoint_cases.py", "derive.py", "module_registry.py", "shape_key.py"))
    h = hashlib.sha256()
    for path in sorted(set(files)):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def stamp(path: Path, registry: Path, source: str) -> None:
    data = json.loads(path.read_text())
    if not data.get("complete") or data.get("errors"):
        raise ValueError("cannot publish an incomplete derivation")
    if source != source_identity():
        raise ValueError("dispatch sources changed during derivation; repeat it")
    data["source_identity"] = source
    data["registry_identity"] = hashlib.sha256(registry.read_bytes()).hexdigest()
    write_json(path, data)


def load(arch: str, registry: Path | None = None, path: Path | None = None) -> dict:
    from miniworld_engine.autotune.derive import normalise_arch, registry_path
    registry = registry if registry is not None else registry_path(arch)
    path = path if path is not None else registry.with_suffix(".units.json")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("kernel derivation evidence must be an object")
    if (not data.get("complete") or data.get("errors")
            or data.get("source_identity") != source_identity()
            or data.get("registry_identity") != hashlib.sha256(registry.read_bytes()).hexdigest()
            or normalise_arch(data.get("arch", "")) != normalise_arch(arch)):
        raise ValueError("kernel derivation is incomplete or stale; run dev derive")
    return data


def is_verified(arch: str, registry: Path) -> bool:
    """Incomplete publications must not hide a valid fallback plan."""
    try:
        load(arch, registry)
    except (OSError, ValueError):
        return False
    return True


def label(unit, case) -> str:
    from miniworld_engine.autotune.derive import DeriveUnit
    return DeriveUnit(case.name, case.stream_for(unit.dim_index), unit.length,
                      tuple(case.dims[unit.dim_index].items()), unit.impl, unit.dtype,
                      unit.compute, "train" if unit.train else "eval",
                      (unit.switch, str(unit.value)) if unit.switch else None,
                      case.augmentation_for(unit.dim_index, unit.train)).label


def select(work: list, cases: list, evidence: dict, missing: set[str]) -> list:
    """Greedy set cover, with no removal unless a retained unit supplies every missing key."""
    by_name = {c.name: c for c in cases}
    candidates = []
    for unit in work:
        key = label(unit, by_name[unit.case])
        if key not in evidence:
            raise ValueError(f"no derivation evidence for {key}")
        candidates.append((unit, set(evidence[key]) & missing))
    reachable = set().union(*(keys for _, keys in candidates)) if candidates else set()
    declared_modules = {key.split("[", 1)[0] for key in evidence}
    if set(by_name) >= declared_modules and missing - reachable:
        raise ValueError(f"missing keys have no runnable module unit: {sorted(missing - reachable)}")
    remaining = missing & reachable
    selected = []
    while remaining:
        # At equal coverage use the smaller tensor invocation.
        index = max(range(len(candidates)),
                    key=lambda i: (len(candidates[i][1] & remaining),
                                   -candidates[i][0].length))
        unit, keys = candidates.pop(index)
        selected.append(unit)
        remaining.difference_update(keys)
    return selected


def _plan_suffix(arch: str) -> Path:
    from miniworld_engine.autotune.derive import normalise_arch

    tag = normalise_arch(arch)
    if not tag.startswith("sm") or not tag[2:].isdigit():
        raise ValueError(f"invalid GPU architecture: {arch!r}")
    return Path(tag) / source_identity() / "registry_kernel.csv"


def generated_registry(arch: str) -> Path:
    """Writable plans are isolated from the installed package and keyed by source revision.

    MINIWORLD_PLAN_CACHE_DIR overrides only derived dispatch plans, not tuned kernel
    choices. Otherwise use the standard XDG cache location, including in wheel installs.
    """
    import os

    override = os.environ.get("MINIWORLD_PLAN_CACHE_DIR")
    if override:
        root = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        if not base.is_absolute():
            base = Path.home() / ".cache"
        root = base / "miniworld-engine" / "plans"
    return root / _plan_suffix(arch)


def packaged_registry(arch: str) -> Path:
    """A matching verified plan can ship in a read-only wheel."""
    return ROOT / "autotune" / "plans" / _plan_suffix(arch)


def registry_candidates(arch: str) -> tuple[Path, ...]:
    from miniworld_engine.autotune.derive import REGISTRY_KERNEL

    return generated_registry(arch), packaged_registry(arch), REGISTRY_KERNEL


def ensure(arch: str, *, workers: int = 1) -> Path:
    """Verify or derive the entire plan BEFORE running any autotune work.

    The recorder patches dispatch and requires compile_wrap=disable at import time;
    its subprocess must never contaminate the real build process. A failed derivation
    leaves existing plans intact and prevents an expensive, uncertifiable build.
    """
    import fcntl
    import os
    import subprocess
    import sys
    import tempfile

    for current in registry_candidates(arch):
        if is_verified(arch, current):
            return current
    out = generated_registry(arch)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / ".derive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            load(arch, out)
            return out
        except (OSError, ValueError):
            pass
        print(f"Refreshing kernel plan for {arch}; no tuning runs until it is verified.",
              flush=True)
        with tempfile.TemporaryDirectory(prefix="derive-", dir=out.parent) as td:
            temporary = Path(td) / out.name
            env = {**os.environ, "MINIWORLD_COMPILE_WRAP": "disable",
                   "QUACK_CACHE_ENABLED": "0"}
            command = [sys.executable, "-m", "miniworld_engine.cli", "dev", "derive",
                       "--arch", arch, "--out", str(temporary),
                       "--workers", str(max(1, workers))]
            result = subprocess.run(command, env=env, check=False)
            if result.returncode:
                error_report = temporary.with_suffix(".errors.json")
                if error_report.exists():
                    error_report.replace(out.with_suffix(".errors.json"))
                raise ValueError(f"kernel plan derivation failed (exit {result.returncode}); "
                                 "no tuning started")
            load(arch, temporary)
            temporary.replace(out)
            temporary.with_suffix(".units.json").replace(out.with_suffix(".units.json"))
        load(arch, out)
    return out

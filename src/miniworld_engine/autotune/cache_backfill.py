"""Backfill ``driver_identity`` into caches written before the field existed, from git history.

``driver_identity`` only guards a cache once that cache carries one, so a freshly-added field is
dormant across every cache already committed -- and a cache whose build driver changed since it was
tuned keeps reporting OK. The provenance is not lost, though: git knows the commit that last wrote
each cache file, and therefore what the driver looked like when it was written. Recovering the hash
from that commit is exact for the question being asked ("which driver produced this cache?"), so the
guard turns on immediately instead of waiting for every cache to be rebuilt.

Stamping the CURRENT driver hash instead would be a lie -- it would assert the cache was built by
code it has never seen, and permanently hide the drift this field exists to surface.
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from miniworld_engine.autotune.cache import (
    DRIVER_ID_SCHEME,
    _registry_driver,
    _imported_driver_scope,
    _scoped_driver_source,
    driver_identity,
)

_REPO = Path(__file__).resolve().parents[3]
_DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Backfilled:
    op: str
    gpu: str
    historical: str
    current: str

    @property
    def drifted(self) -> bool:
        return self.historical != self.current


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(("git", *args), cwd=_REPO, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _driver_path_at(commit: str, mod_name: str, fn_name: str) -> str | None:
    """Where ``fn_name``'s driver source lived AT ``commit``.

    Today's path is tried first, but the drivers have been reorganised twice -- a single
    ``kernels/drivers.py``, then abbreviated family modules (``drivers_adaln.py``,
    ``drivers_attn.py``, ``drivers_ln.py``), then today's ``kernels/drivers/<family>.py`` package --
    and no name transform maps `drivers.augmented_attention` onto `drivers_attn.py`. So the fallback
    stops guessing paths and SEARCHES: any driver-ish module under ``kernels/`` at that commit that
    actually DEFINES this function is the one that built the cache. Guessing skipped 70 of 234
    caches, silently, and they were the oldest ones."""
    rel = "src/" + mod_name.replace(".", "/") + ".py"
    if _git("cat-file", "-e", f"{commit}:{rel}") is not None:
        return rel
    base = "src/" + "/".join(mod_name.split(".")[:-2])          # .../kernels
    tree = _git("ls-tree", "-r", "--name-only", f"{commit}:{base}") or ""
    needle = f"def {fn_name}("
    for cand in sorted(x for x in tree.splitlines() if "driver" in x.lower() and x.endswith(".py")):
        src = _git("show", f"{commit}:{base}/{cand}")
        if src and needle in src:
            return f"{base}/{cand}"
    return None


def _sibling_reader(commit: str):
    """Read a sibling driver module AS OF ``commit``, for :func:`_imported_driver_scope`.

    Without this the historical stamp would hash an old own-module against TODAY's siblings and
    land on a hash no past state ever produced -- a fabricated stamp, which is worse than none:
    it reads as fresh. Reuses :func:`_driver_path_at`'s search, so a sibling that has since been
    renamed is still found (``fn_name=""`` makes the needle match any definition, which is right
    here -- we want the module, not one function in it)."""
    def read(mod_name: str) -> str | None:
        base = "src/" + mod_name.replace(".", "/")
        # BOTH spellings, and the package one is not optional: every driver imports the shared
        # helpers from `kernels.drivers`, which is a PACKAGE. Resolving only `<mod>.py` made the
        # historical side read nothing there while the live side read `__init__.py` -- an
        # asymmetry, not a difference, and it drifted all 228 caches at once.
        for rel in (f"{base}.py", f"{base}/__init__.py"):
            src = _git("show", f"{commit}:{rel}")
            if src is not None:
                return src
        return None
    return read


def _hash_scoped(src: str, fn_name: str, mod_name: str = "", commit: str = "") -> str | None:
    import hashlib

    scoped = _scoped_driver_source(src, fn_name)
    if scoped is None:
        return None
    if mod_name and commit:
        scoped += "\n" + _imported_driver_scope(mod_name, src, read_source=_sibling_reader(commit))
    body = "\n".join(ln.rstrip() for ln in scoped.splitlines() if ln.strip())
    return hashlib.sha1(body.encode()).hexdigest()[:12]


#: Why caches were passed over on the last :func:`backfill` -- reported, never silent: an
#: unreported skip is what let a rename hide 37% of the corpus.
LAST_SKIPPED: "Counter[str]" = Counter()


def backfill(*, apply: bool = False) -> list[Backfilled]:
    """Recover each cache's driver fingerprint as of the commit that last wrote it.

    ``apply=False`` (default) reports without touching a file. Caches that already carry a
    ``driver_identity``, have no registry driver, or have no git history are skipped."""
    out: list[Backfilled] = []
    skipped: Counter[str] = Counter()
    for jf in sorted(_DATA.glob("*/*.json")):
        op = jf.parent.name
        try:
            data = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            skipped["unreadable JSON"] += 1
            continue
        if (data.get("driver_identity")
                and data.get("driver_id_scheme") == DRIVER_ID_SCHEME):
            skipped["already stamped (current scheme)"] += 1
            continue
        # A stamp from an OLDER scheme is not comparable and is re-derived here; the value is
        # recomputed from git either way, so this overwrites nothing that was ever trusted.
        ref = _registry_driver(op)
        if ref is None:
            skipped["no registry driver (dispatch-only cache)"] += 1
            continue
        mod_name, fn_name = ref
        commit = (_git("log", "-1", "--format=%H", "--", str(jf.relative_to(_REPO))) or "").strip()
        if not commit:
            skipped["cache never committed"] += 1
            continue
        rel = _driver_path_at(commit, mod_name, fn_name)
        if rel is None:
            skipped["driver file absent at that commit"] += 1
            continue
        old_src = _git("show", f"{commit}:{rel}")
        if old_src is None:
            skipped["driver unreadable at that commit"] += 1
            continue
        historical = _hash_scoped(old_src, fn_name, mod_name, commit)
        current = driver_identity(op)
        if historical is None or current is None:
            skipped["could not hash the driver scope"] += 1
            continue
        out.append(Backfilled(op, jf.stem, historical, current))
        if apply:
            data["driver_identity"] = historical
            data["driver_id_scheme"] = DRIVER_ID_SCHEME
            # No trailing newline: match `_atomic.write_json`, the canonical writer, or every
            # rebuild re-diffs the file purely to strip one.
            jf.write_text(json.dumps(data, indent=2, sort_keys=True))
    LAST_SKIPPED.clear()
    LAST_SKIPPED.update(skipped)
    return out


def format_report(rows: list[Backfilled], *, applied: bool) -> str:
    drift = [r for r in rows if r.drifted]
    verb = "stamped" if applied else "would stamp"
    lines = [f"driver_identity backfill: {verb} {len(rows)} caches from git history; "
             f"{len(drift)} were built by a driver that has since changed"]
    if drift:
        lines.append("\nDRIFTED (these become STALE once stamped -- rebuild them):")
        for op in sorted({r.op for r in drift}):
            r = next(x for x in drift if x.op == op)
            lines.append(f"  {op:46} {r.historical} -> {r.current}")
    if LAST_SKIPPED:
        lines.append(f"\nSKIPPED ({sum(LAST_SKIPPED.values())}) -- these keep NO driver guard:")
        for why, n in sorted(LAST_SKIPPED.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {n:4d}  {why}")
    return "\n".join(lines)

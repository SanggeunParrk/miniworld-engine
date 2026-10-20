"""Prebuilt kernels shipped inside the engine package, looked up by content key before any build.

A release wheel of miniworld-engine carries ``miniworld_engine/_prebuilt/<arch>-cu<XY>/`` per GPU arch and CUDA build
(``sm90-cu129``, ``sm90-cu130``; ``sm100-*`` later), each a ``manifest.json`` mapping content keys to artifacts (torch extension ``.so`` files and cubins) under ``artifacts/``. The
directory is produced at release time by ``miniworld-engine dev bundle-kernels`` (it is not in git), and an sdist or
git install simply has none, so every artifact is JIT-built as before. ``MINIWORLD_ENGINE_KERNEL_BUNDLES``
(``os.pathsep``-separated directories holding a ``manifest.json``) adds bundles from elsewhere, e.g. a freshly built
one.

A bundle is used only when its manifest matches the running torch and CUDA (``torch.version.cuda``) and the current
device's arch; each artifact is then found by its key (``_codegraph``), which covers the sources and headers it
compiles, its flags, the versions and (for a torch extension) the Python ABI, so a bundle can never hand out a binary
built from other code, and an engine release that does not touch a kernel's graph keeps the same binary.

``record`` is the bundle builder's side: with ``MINIWORLD_ENGINE_BUNDLE_RECORD=<file>`` set, every artifact a process
resolves (prebuilt, cached or built) is appended to that JSON-lines file, with the recipe that rebuilds it.

See docs/kernels/prebuilt-kernels.md.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import threading
from pathlib import Path

MANIFEST = "manifest.json"
#: The manifest format this engine reads.
MANIFEST_FORMAT = 2
BUNDLE_ENV = "MINIWORLD_ENGINE_KERNEL_BUNDLES"
RECORD_ENV = "MINIWORLD_ENGINE_BUNDLE_RECORD"
#: Where a release wheel carries its kernels: ``<package>/_prebuilt/<arch>/``. ``MINIWORLD_ENGINE_PREBUILT_DIR`` points
#: it elsewhere (the bundle builder does, so a release build sees only the bundle it is building).
PREBUILT_DIR = Path(os.environ.get("MINIWORLD_ENGINE_PREBUILT_DIR") or Path(__file__).resolve().parents[1] / "_prebuilt")


class PrebuiltMissing(RuntimeError):
    """An artifact had to be compiled while ``settings.jit_fallback`` is False."""


@dataclasses.dataclass(frozen=True)
class Bundle:
    root: Path
    manifest: dict

    @property
    def entries(self) -> dict:
        return self.manifest.get("artifacts", {})


def engine_version() -> str:
    from miniworld_engine import __version__

    return __version__


def _candidate_dirs() -> list[Path]:
    dirs = [Path(part.strip()).expanduser() for part in os.environ.get(BUNDLE_ENV, "").split(os.pathsep) if part.strip()]
    if PREBUILT_DIR.is_dir():
        dirs += sorted(p for p in PREBUILT_DIR.iterdir() if p.is_dir())
    return list(dict.fromkeys(d.resolve() for d in dirs if (d / MANIFEST).is_file()))


def mismatch(manifest: dict, *, arch: str | None = None) -> str | None:
    """Why a bundle's manifest does not fit this process, or None when it does. The engine version is not compared:
    the keys already cover every byte an artifact was compiled from."""
    from miniworld_engine.kernels import _codegraph as cg

    if manifest.get("format") != MANIFEST_FORMAT:
        return f"manifest format {manifest.get('format')} (this engine reads {MANIFEST_FORMAT})"
    if manifest.get("key_scheme") != cg.KEY_SCHEME:
        return f"key scheme {manifest.get('key_scheme')} (this engine uses {cg.KEY_SCHEME})"
    versions = cg._versions()
    for field in ("torch", "cuda"):
        if manifest.get(field) != versions[field]:
            return f"built for {field} {manifest.get(field)}, this is {versions[field]}"
    arch = arch or cg.target_arch()
    if arch != "cpu" and manifest.get("arch") != arch:
        return f"arch {manifest.get('arch')} != {arch}"
    return None


@functools.lru_cache(maxsize=1)
def bundles() -> tuple[Bundle, ...]:
    """The usable bundles (manifest matches this process), discovered once per process. Bundles for another arch, or for another
    CUDA build when one fits, are skipped silently (a wheel carries several); if none fits, that is said once on stderr."""
    from miniworld_engine.kernels import _codegraph as cg
    from miniworld_engine.kernels._nvcc import build_log

    found, skipped = [], []
    arch = cg.target_arch()
    for root in _candidate_dirs():
        try:
            manifest = json.loads((root / MANIFEST).read_text())
        except (OSError, ValueError) as exc:
            build_log(f"ignoring kernel bundle {root}: unreadable manifest ({exc})")
            continue
        if arch != "cpu" and manifest.get("arch") != arch:
            continue
        why = mismatch(manifest, arch=arch)
        if why:
            skipped.append(f"{root.name} ({why})")
            continue
        found.append(Bundle(root, manifest))
    if skipped and not found:  # one build per CUDA ships side by side: only say so when none of them fits
        build_log(f"no prebuilt kernels for this torch / CUDA; ignoring {', '.join(skipped)}")
    return tuple(found)


def enabled() -> bool:
    try:
        from miniworld_engine import settings
    except ImportError:
        return True
    return bool(getattr(settings.current(), "prebuilt_kernels", True))


@dataclasses.dataclass(frozen=True)
class Hit:
    path: Path
    entry: dict
    bundle: Path


def find(kind: str, name: str, key: str) -> Hit | None:
    """The installed artifact of ``kind`` (``ext`` / ``cubin``) named ``name`` with content key ``key``, or None."""
    if not enabled():
        return None
    for b in bundles():
        entry = b.entries.get(key)
        if entry is None or entry.get("kind") != kind or entry.get("name") != name:
            continue
        path = b.root / entry["path"]
        if path.is_file():
            return Hit(path, entry, b.root)
    return None


def describe() -> str:
    """One line naming the bundles in use (for the miss message)."""
    found = bundles() if enabled() else ()
    if not found:
        return "no prebuilt kernels in this install for this torch / CUDA / GPU arch"
    return ", ".join(f"{b.root.name} ({len(b.entries)} artifacts)" for b in found)


# --------------------------------------------------------------------------------------------------- recording
_RECORDED: set[str] = set()
_REC_GUARD = threading.Lock()


def record(entry: dict) -> None:
    """Append one resolved artifact to ``$MINIWORLD_ENGINE_BUNDLE_RECORD`` (once per key per process)."""
    target = os.environ.get(RECORD_ENV)
    if not target:
        return
    with _REC_GUARD:
        if entry["key"] in _RECORDED:
            return
        _RECORDED.add(entry["key"])
        line = json.dumps(entry, sort_keys=True) + "\n"
        import fcntl

        # several processes append to one file; a recipe line can be large, so lock around the write
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode())
        finally:
            os.close(fd)


def recording() -> bool:
    return bool(os.environ.get(RECORD_ENV))

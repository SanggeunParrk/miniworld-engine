"""The one loader of Anthropic's uplifting-biomolecular-modeling release (third_party/anthropic/upstream) for the modules' ``"anthropic"``
backends.

    mod = load("opt_core.kernels.dtk_kernels")                                    # common/opt_core on sys.path
    mod = load("bz_sampler_dit", path="boltz2/opt/forward/sampler/src")           # a model folder's own file

``load`` puts ``common/opt_core`` (and ``path``, a release directory) on ``sys.path``, checks the release revision of UPSTREAM.json and the
sha256 of the imported file against it, and imports. Every failure raises :class:`ReleaseMissing` with the reason; ``missing(...)`` returns
that reason (or ``None``) for a backend's ``_supports_anthropic``. ``MINIWORLD_ANTHROPIC_ROOT`` overrides the release root.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

#: The release revision the backends were written against (UPSTREAM.json "revision").
REVISION = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
_THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party" / "anthropic"


class ReleaseMissing(RuntimeError):
    """The release (or one file of it) is not there, or is not the revision the backends were written against."""


def root() -> Path:
    return Path(os.environ.get("MINIWORLD_ANTHROPIC_ROOT") or _THIRD_PARTY / "upstream").resolve()


@cache
def _manifest() -> dict[str, str]:
    """path -> sha256 of UPSTREAM.json, after the revision check."""
    upstream = root().parent / "UPSTREAM.json"
    if not upstream.is_file():
        raise ReleaseMissing(f"no {upstream} (the Anthropic release is not checked out)")
    data = json.loads(upstream.read_text())
    if data.get("revision") != REVISION:
        raise ReleaseMissing(f"release revision {data.get('revision')} != {REVISION} (the backends were written against {REVISION[:8]})")
    return {f["path"]: f["sha256"] for f in data["files"]}


@cache
def load(module: str, *, path: str = "common/opt_core") -> ModuleType:
    """Import ``module`` from the release directory ``path`` (``common/opt_core`` is always on ``sys.path``), after checking its file's
    sha256 against UPSTREAM.json."""
    base = root()
    rel = f"{path}/{module.replace('.', '/')}"
    candidates = [rel + ".py", rel + "/__init__.py"]
    hit = next((c for c in candidates if (base / c).is_file()), None)
    if hit is None:
        raise ReleaseMissing(f"{module}: none of {candidates} under {base}")
    want = _manifest().get(hit)
    if want is None:
        raise ReleaseMissing(f"{hit} is not a file of release {REVISION[:8]}")
    if hashlib.sha256((base / hit).read_bytes()).hexdigest() != want:
        raise ReleaseMissing(f"{hit} differs from release {REVISION[:8]} (sha256 mismatch)")
    for d in dict.fromkeys(("common/opt_core", path)):
        p = str(base / d)
        if p not in sys.path:
            sys.path.insert(0, p)
    loaded = sys.modules.get(module)
    if loaded is not None and Path(getattr(loaded, "__file__", "") or "").resolve() != (base / hit).resolve():
        raise ReleaseMissing(f"{module} is already imported from {getattr(loaded, '__file__', None)}, not from the release")
    return importlib.import_module(module)


def missing(module: str, *, path: str = "common/opt_core") -> str | None:
    """Why ``load(module, path=path)`` fails (a short reason), or ``None`` when it loads."""
    try:
        load(module, path=path)
    except Exception as e:  # any import failure is the reason the backend cannot run
        return f"release {module} unavailable: {type(e).__name__}: {e}"
    return None


def unavailable(module: str, device, *, path: str = "common/opt_core") -> str | None:
    """:func:`missing` plus the device: the release's kernels run on CUDA tensors; which card it serves is the release's own decision
    (its providers select a cell for the device and refuse by name when they have none, see :func:`kernel_selection`), not this
    adapter's. The reason it cannot serve here, or ``None``."""
    why = missing(module, path=path)
    if why is not None:
        return why
    if getattr(device, "type", device) != "cuda":
        return "the release's kernels run on CUDA tensors only"
    return None


def kernel_selection(provider: ModuleType, *args, **kwargs):
    """``(Selection, None)`` for the row an ``opt_core.kernels`` provider's ``select(*args, **kwargs)`` names -- a kernel row or a stock row
    (``STOCK_ROWS``: SDPA, torch_swiglu, torch_math, the cuEquivariance op, the engine's own module ...): the release serves that row for the
    cell, so the backend serves it too -- else ``(None, reason)`` when the provider refuses the call by name (no cell)."""
    try:
        sel = provider.select(*args, **kwargs)
    except provider.Refusal as e:
        return None, f"the provider refuses the call: {str(e)[:240]}"
    return sel, None

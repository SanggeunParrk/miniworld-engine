"""Which kernels exist, and which of them ran on this card.

Two facts, two sources, no inference between them:

  * WHAT EXISTS is declared by the repo in ``kernels/registry.csv``. It is data, not something
    to be discovered. The previous version inferred it by walking the AST for
    ``@triton.autotune`` decorators plus whatever symbols the bench happened to import, and that
    guessing produced a string of wrong answers: backends read off the directory name (nine
    Triton kernels living under ``cute/`` were labelled cute), six ``__global__`` kernels in .cu
    files and six ``@cute.kernel`` collectives missing entirely, and host launchers counted as
    kernels.

  * WHAT RUNS is decided by running it. Not by scanning for ``assert capability == 9``, not by
    grepping error strings out of old artifacts. Launch the kernel: it works or it does not, and
    the failure message is the reason. Anything a run has not covered is ``untested`` -- which is
    a hole to close, not a verdict to guess at.

The registry is the denominator and it does not come from the run, so a kernel nothing reaches
stays visible instead of dropping out of both sides and reading as 100% covered.
"""

from __future__ import annotations

import csv
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
_REGISTRY = _PKG / "kernels" / "registry.csv"
#: Per-card manifests, beside the tuned caches in `autotune/data/`. They used to be resolved as
#: `<repo>/configs/devices` via `parents[3]`, which is only the repo in an editable install: from a
#: wheel that path is `<site-packages>/../../configs/devices` and does not exist, so `run_all`
#: wrote its record nowhere.
_DEVICES = _PKG / "autotune" / "manifests"

_FIELDS = ["kernel", "backend", "family", "file", "status", "detail"]


def registry() -> list[dict]:
    """Every kernel the repo declares. Add a kernel to the CSV when you add a kernel."""
    with _REGISTRY.open(newline="") as handle:
        return list(csv.DictReader(handle))


def registered_kernels() -> frozenset[str]:
    return frozenset(r["kernel"] for r in registry())


def manifest_path(gpu_key: str) -> Path:
    return _DEVICES / f"{gpu_key}.csv"


def record(gpu_key: str, results: dict[str, tuple[bool, str]]) -> Path:
    """Merge what a run observed -- ``kernel -> (ran, detail)`` -- into this GPU's manifest.

    MERGE, not overwrite. Every caller is a PARTIAL run: `bench_kernel all` reaches 29 of the 103
    declared kernels and `bench_module all` reaches a different subset, so writing `untested` for
    everything a run did not touch means the file only ever describes the last command. Measured:
    one `bench_kernel all` took the committed manifest from 94 ok / 6 failed to 11 ok / 92
    untested, discarding every result the module benches had established -- in a TRACKED file, so
    the loss would have been committed.

    A kernel this run did not touch keeps whatever the manifest already said about it; one that
    has never been seen is `untested`. A run that DID touch a kernel always wins, so a kernel that
    starts failing is recorded as failing.
    """
    prior = {r["kernel"]: r for r in load_manifest(gpu_key)}
    rows = []
    for entry in registry():
        ran, detail = results.get(entry["kernel"], (None, ""))
        if ran is None:
            was = prior.get(entry["kernel"], {})
            status, detail = was.get("status", "untested"), was.get("detail", "")
        else:
            status = "ok" if ran else "failed"
        rows.append({
            "kernel": entry["kernel"], "backend": entry["backend"],
            "family": entry["family"], "file": entry["file"],
            "status": status, "detail": detail,
        })
    _DEVICES.mkdir(parents=True, exist_ok=True)
    path = manifest_path(gpu_key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_manifest(gpu_key: str) -> list[dict]:
    path = manifest_path(gpu_key)
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def runnable_kernels(gpu_key: str) -> frozenset[str]:
    """Kernels this card was observed to run. Not "expected to" -- observed."""
    return frozenset(r["kernel"] for r in load_manifest(gpu_key) if r["status"] == "ok")


def untested_kernels(gpu_key: str) -> frozenset[str]:
    """Declared but never exercised here. These are the holes."""
    tested = {r["kernel"] for r in load_manifest(gpu_key) if r["status"] != "untested"}
    return frozenset(registered_kernels() - tested)


def kernels_for(gpu_key: str, family: str) -> tuple[str, ...]:
    return tuple(r["kernel"] for r in load_manifest(gpu_key)
                 if r["family"] == family and r["status"] == "ok")

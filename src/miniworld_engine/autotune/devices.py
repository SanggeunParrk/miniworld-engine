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

#: `dtype` is part of a row's IDENTITY, not a note on it. A manifest row says what a kernel
#: measured, and what it measures depends on the precision it ran at: `layernorm_fwd_saveact_triton`
#: is 2.8e-03 in bf16 and 1.7e-07 in fp32, four orders apart. The file used to hold one row per
#: kernel, so a run at the other precision REPLACED the first one's number and left nothing saying
#: which precision was in the file -- and the bands are calibrated from these numbers
#: (`test_a_declared_band_is_above_what_that_kernel_measured`), so a single fp32 run would have
#: silently re-priced every bf16 band four orders too tight. Every row written before this column
#: existed was bf16: it was the only precision the drivers ever ran (see the driver dtype fix).
_FIELDS = ["kernel", "backend", "family", "file", "status", "detail", "dtype"]

#: What a row with no `dtype` cell means. See `_FIELDS`.
_LEGACY_DTYPE = "bf16"

#: Written as the first row, kernel `#provenance`. A manifest is the only evidence that any kernel
#: has ever run on a given card, and without this it does not say WHEN or against WHICH code -- so
#: a file from six months and two rewrites ago is indistinguishable from one produced this morning.
#: Carried in-band rather than in a sidecar because the evidence and its provenance must not be
#: separable; a sidecar is a thing to lose.
_PROVENANCE = "#provenance"


def _provenance_row() -> dict:
    import subprocess
    from datetime import datetime, timezone

    def _git(*args: str) -> str:
        try:
            out = subprocess.run(["git", *args], cwd=_PKG.parent.parent, capture_output=True,
                                 text=True, check=False, timeout=20)
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    version = ""
    try:
        from importlib.metadata import version as _v
        version = _v("miniworld-engine")
    except Exception:
        pass
    return {
        "kernel": _PROVENANCE, "backend": version,
        "family": _git("rev-parse", "HEAD")[:12],
        # The tree state of the CODE, with this directory excluded. The provenance answers "against
        # which source was this produced", and the manifests are the evidence, not the source. A
        # run at the second precision would otherwise always say `dirty`: the run at the first one
        # has just rewritten the file it is standing in. (Same failure as computing this row after
        # opening the manifest for writing, which truncates it -- fixed once already, one line up.)
        "file": "dirty" if _git("status", "--porcelain", "--",
                                ".", ":(exclude)src/miniworld_engine/autotune/manifests")
        else "clean",
        "status": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "detail": _git("describe", "--tags", "--always"),
        # The provenance row is about the FILE, not about one measurement, so it names no
        # precision. Present because DictWriter refuses a row missing a declared field.
        "dtype": "",
    }


def registry() -> list[dict]:
    """Every kernel the repo declares. Add a kernel to the CSV when you add a kernel."""
    with _REGISTRY.open(newline="") as handle:
        return list(csv.DictReader(handle))


def registered_kernels() -> frozenset[str]:
    return frozenset(r["kernel"] for r in registry())


def manifest_path(gpu_key: str) -> Path:
    return _DEVICES / f"{gpu_key}.csv"


def _declares(entry: dict) -> frozenset[str]:
    """The precisions a registry row says its kernel runs at. Blank means bf16, as everywhere."""
    return frozenset(a.strip() for a in (entry.get("dtypes") or "bf16").split("|") if a.strip())


def record(gpu_key: str, results: dict[str, tuple[bool, str]],
           dtype: str | None = None, skipped: dict[str, str] | None = None) -> Path:
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

    ``dtype`` is the precision this run measured at, defaulting to the drivers' own. A run merges
    into ITS precision's rows and leaves the other precision's alone -- the two are different
    measurements of the same kernel, not competing answers, and before the column existed the
    second one overwrote the first.
    """
    if dtype is None:
        from miniworld_engine.kernels.drivers import DTYPE_MODE
        dtype = DTYPE_MODE
    prior = load_manifest(gpu_key)
    mine = {r["kernel"]: r for r in prior if r["dtype"] == dtype}
    rows = []
    for entry in registry():
        # Only kernels this precision applies to. A bf16-only kernel has nothing to say about an
        # fp32 run -- writing it `untested` there invents a hole that can never be filled, and one
        # fp32 run put 50 of them in the file. A row exists for a (kernel, precision) the registry
        # declares, and for no other.
        if dtype not in _declares(entry):
            continue
        ran, detail = results.get(entry["kernel"], (None, ""))
        if ran is not None:
            status = "ok" if ran else "failed"
        elif entry["kernel"] in (skipped or {}):
            # A THIRD status, because there are three answers and "failed" is not one of them for a
            # kernel this card cannot run. Six arch-gated kernels sat at `failed` in the committed
            # manifest -- recorded before the arch gate existed, then carried forward untouched by
            # every run since, because a skipped kernel never reached `results` and so never
            # updated its own row. The verdict outlived the run that produced it.
            status, detail = "skipped", skipped[entry["kernel"]]
        else:
            was = mine.get(entry["kernel"], {})
            status, detail = was.get("status", "untested"), was.get("detail", "")
        rows.append({
            "kernel": entry["kernel"], "backend": entry["backend"],
            "family": entry["family"], "file": entry["file"],
            "status": status, "detail": detail, "dtype": dtype,
        })
    # Every OTHER precision's rows, carried through -- but only for precisions the registry still
    # declares. A row for a precision a kernel no longer runs at is evidence about a configuration
    # that no longer exists, and it does not merely sit there: the declared bands are calibrated
    # from these numbers, so a stale fp32 row would price a band the kernel is never held to, and a
    # stale bf16 row (this family had eight, left over from when the driver was pinned to fp32 and
    # the numbers in them are fp32 numbers) would price the bf16 band from the wrong precision
    # entirely. Written after this run's rows, so the file reads as one block per precision.
    declared = {e["kernel"]: _declares(e) for e in registry()}
    rows += [r for r in prior
             if r["dtype"] != dtype and r["dtype"] in declared.get(r["kernel"], frozenset())]
    _DEVICES.mkdir(parents=True, exist_ok=True)
    path = manifest_path(gpu_key)
    # BEFORE opening the file. `_provenance_row` shells out to `git status`, and opening the
    # manifest for writing truncates it -- so computing the row inside the `with` block reported
    # `dirty` on every run, including one measured from a checkout with nothing modified. The
    # act of recording made the record say the tree was unclean.
    provenance = _provenance_row()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerow(provenance)
        writer.writerows(rows)
    return path


def load_manifest(gpu_key: str, dtype: str | None = None) -> list[dict]:
    """This card's rows, optionally only those measured at ``dtype``.

    A row with no `dtype` cell is bf16 -- it was written before the column existed, when bf16 was
    the only precision the drivers ran. Filled in here rather than in the file so an old manifest
    stays readable.
    """
    path = manifest_path(gpu_key)
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["kernel"] != _PROVENANCE]
    for r in rows:
        r["dtype"] = (r.get("dtype") or "").strip() or _LEGACY_DTYPE
    return [r for r in rows if dtype is None or r["dtype"] == dtype]


def provenance(gpu_key: str) -> dict | None:
    """Version, commit, tree state and date behind this card's manifest, or None if it predates
    the row being written."""
    path = manifest_path(gpu_key)
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["kernel"] == _PROVENANCE:
                return {"version": r["backend"], "commit": r["family"], "tree": r["file"],
                        "date": r["status"], "describe": r["detail"]}
    return None


def runnable_kernels(gpu_key: str) -> frozenset[str]:
    """Kernels this card was observed to run. Not "expected to" -- observed."""
    return frozenset(r["kernel"] for r in load_manifest(gpu_key) if r["status"] == "ok")


def untested_kernels(gpu_key: str) -> frozenset[str]:
    """Declared but never exercised here. These are the holes."""
    # `skipped` counts as tested: the card was asked and gave a definite answer. A hole is a
    # kernel nothing has ever tried here, not one this card is known not to run.
    tested = {r["kernel"] for r in load_manifest(gpu_key) if r["status"] != "untested"}
    return frozenset(registered_kernels() - tested)


def kernels_for(gpu_key: str, family: str) -> tuple[str, ...]:
    """This family's kernels that ran here, once each. A kernel measured at both precisions has a
    row per precision, and callers want the kernel list, not the measurement list."""
    out: list[str] = []
    for r in load_manifest(gpu_key):
        if r["family"] == family and r["status"] == "ok" and r["kernel"] not in out:
            out.append(r["kernel"])
    return tuple(out)

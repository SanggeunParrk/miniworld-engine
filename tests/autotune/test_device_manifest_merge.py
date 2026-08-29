"""A partial run must not erase what earlier runs established about this card.

`autotune/manifests/<gpu>.csv` is a TRACKED record of which kernels this card was observed to run.
Every writer is a partial run: `bench_kernel all` reaches 29 of the 103 declared kernels and
`bench_module all` reaches a different subset. `record` wrote the whole registry every time and
marked everything the run did not touch `untested`, so the file only ever described the last
command -- and the two bench commands overwrote each other.

Measured: one `bench_kernel all` took the committed manifest from 94 ok / 6 failed to
11 ok / 92 untested. In a tracked file, so committing the run would have committed the loss.
"""
from __future__ import annotations

import csv
import re

import pytest

from miniworld_engine.autotune import devices

GPU = "TEST GPU (sm00)"


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(devices, "_DEVICES", tmp_path)

    def read():
        # Skip the `#provenance` row -- version, commit, tree state and date, written so a
        # manifest says WHEN and against WHICH code it was produced. It is not a kernel, and
        # `devices.load_manifest` filters it out for the same reason; this fixture reads the CSV
        # directly and has to do the same.
        with (tmp_path / f"{GPU}.csv").open(newline="") as fh:
            return {r["kernel"]: (r["status"], r["detail"]) for r in csv.DictReader(fh)
                    if r["kernel"] != devices._PROVENANCE}

    return read


def _some_kernels(n: int) -> list[str]:
    ks = [e["kernel"] for e in devices.registry()]
    assert len(ks) > n, "the registry is too small for this test to mean anything"
    return ks[:n]


def test_a_later_partial_run_keeps_the_earlier_results(manifest):
    a, b = _some_kernels(2)
    devices.record(GPU, {a: (True, "launched by bench")})
    devices.record(GPU, {b: (True, "launched by bench")})
    got = manifest()
    assert got[a] == ("ok", "launched by bench"), "the second run erased the first"
    assert got[b] == ("ok", "launched by bench")


def test_a_kernel_nobody_has_run_stays_untested(manifest):
    a = _some_kernels(1)[0]
    devices.record(GPU, {a: (True, "launched by bench")})
    got = manifest()
    never = [k for k, v in got.items() if k != a]
    assert never, "the registry has only one kernel?"
    assert all(got[k] == ("untested", "") for k in never)


def test_a_run_that_touched_a_kernel_always_wins(manifest):
    """Including downgrades: a kernel that starts failing must be recorded as failing."""
    a = _some_kernels(1)[0]
    devices.record(GPU, {a: (True, "launched by bench")})
    devices.record(GPU, {a: (False, "OutOfResources")})
    assert manifest()[a] == ("failed", "OutOfResources")


def test_the_whole_registry_is_still_written(manifest):
    """Merging must not turn the manifest into only-what-ran.

    "The whole registry" means every kernel that DECLARES the precision being recorded: a bf16-only
    kernel has nothing to say about an fp32 run, and marking it `untested` there invents a hole
    nothing can ever fill. One fp32 run put 50 such rows in the file.
    """
    a = _some_kernels(1)[0]
    devices.record(GPU, {a: (True, "x")}, dtype="bf16")
    want = [e["kernel"] for e in devices.registry()
            if "bf16" in ((e.get("dtypes") or "bf16").split("|"))]
    assert want, "no kernel declares bf16; this would pass vacuously"
    assert sorted(manifest()) == sorted(want)


def test_a_manifest_says_when_and_against_what_it_was_produced(manifest) -> None:
    """A manifest is the only evidence that any kernel has ever run on a given card. Without
    provenance a file from six months and two rewrites ago is indistinguishable from one produced
    this morning, and `docs/supported.md` cites these as its evidence."""
    devices.record(GPU, {})
    prov = devices.provenance(GPU)
    assert prov is not None, "record() wrote no #provenance row"
    for field in ("version", "commit", "tree", "date"):
        assert prov[field], f"provenance has no {field}: {prov}"
    assert prov["tree"] in ("clean", "dirty"), prov["tree"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", prov["date"]), prov["date"]


def test_provenance_is_not_mistaken_for_a_kernel(manifest) -> None:
    """It shares the file with the kernel rows, so every reader has to skip it. `load_manifest`
    does; a reader that does not would report a 104th kernel that does not exist."""
    devices.record(GPU, {}, dtype="bf16")
    got = devices.load_manifest(GPU)
    assert all(r["kernel"] != devices._PROVENANCE for r in got)
    # One row per kernel that declares bf16 -- not per registry row: an fp32-only kernel has no
    # bf16 row at all. What this is about is that the provenance row is not counted as one of them.
    assert len(got) == sum(1 for e in devices.registry()
                           if "bf16" in (e.get("dtypes") or "bf16").split("|"))

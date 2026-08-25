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

import pytest

from miniworld_engine.autotune import devices

GPU = "TEST GPU (sm00)"


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(devices, "_DEVICES", tmp_path)

    def read():
        with (tmp_path / f"{GPU}.csv").open(newline="") as fh:
            return {r["kernel"]: (r["status"], r["detail"]) for r in csv.DictReader(fh)}

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
    """Merging must not turn the manifest into only-what-ran."""
    a = _some_kernels(1)[0]
    devices.record(GPU, {a: (True, "x")})
    assert len(manifest()) == len(devices.registry())

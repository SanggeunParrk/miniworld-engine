"""One kernel, two precisions, two measurements -- and the file has to hold both.

A manifest row records what a kernel measured against its torch reference. What it measures
depends on the precision it ran at, and not by a little: `layernorm_fwd_saveact_triton` is 2.8e-03
in bf16 and 1.7e-07 in fp32, four orders apart, because bf16 carries ~3 decimal digits and fp32
carries ~7.

`record` wrote one row per kernel. So a run at the other precision REPLACED the first one's number
and left nothing in the file saying which precision was in it. That is not only a lost measurement:
`tests/registry/test_declared_tolerance.py` calibrates every declared band from these numbers, so a
single fp32 run would have re-priced every bf16 band four orders too tight -- and a bf16 run after
it would have priced the fp32 bands four orders too loose, which is the direction that hides
regressions.

The column is also why an old manifest still reads: every row written before it existed was bf16,
because that was the only precision the drivers ever ran.
"""
from __future__ import annotations

import csv

import pytest

from miniworld_engine.autotune import devices

GPU = "TEST GPU (sm00)"


@pytest.fixture
def rows(tmp_path, monkeypatch):
    monkeypatch.setattr(devices, "_DEVICES", tmp_path)

    def read():
        with (tmp_path / f"{GPU}.csv").open(newline="") as fh:
            return [r for r in csv.DictReader(fh) if r["kernel"] != devices._PROVENANCE]

    return read


def _two_kernels() -> tuple[str, str]:
    ks = [e["kernel"] for e in devices.registry()]
    return ks[0], ks[1]


def test_the_second_precision_does_not_overwrite_the_first(rows):
    a, _ = _two_kernels()
    devices.record(GPU, {a: (True, "rel out=2.8e-03")}, dtype="bf16")
    devices.record(GPU, {a: (True, "rel out=1.7e-07")}, dtype="fp32")

    got = {(r["kernel"], r["dtype"]): r["detail"] for r in rows()}
    assert got[(a, "bf16")] == "rel out=2.8e-03", "the bf16 measurement was overwritten"
    assert got[(a, "fp32")] == "rel out=1.7e-07"


def test_a_precision_merges_only_with_itself(rows):
    """The merge that `record` exists for, now per precision: a partial fp32 run must keep the
    earlier fp32 results AND leave every bf16 row alone."""
    a, b = _two_kernels()
    devices.record(GPU, {a: (True, "bf16 a"), b: (True, "bf16 b")}, dtype="bf16")
    devices.record(GPU, {a: (True, "fp32 a")}, dtype="fp32")
    devices.record(GPU, {b: (True, "fp32 b")}, dtype="fp32")

    got = {(r["kernel"], r["dtype"]): r["detail"] for r in rows()}
    assert got[(a, "fp32")] == "fp32 a", "the second fp32 run dropped the first one's result"
    assert got[(b, "fp32")] == "fp32 b"
    assert got[(a, "bf16")] == "bf16 a", "an fp32 run touched a bf16 row"
    assert got[(b, "bf16")] == "bf16 b"


def test_loading_can_ask_for_one_precision(rows):
    a, _ = _two_kernels()
    devices.record(GPU, {a: (True, "bf16 a")}, dtype="bf16")
    devices.record(GPU, {a: (True, "fp32 a")}, dtype="fp32")

    bf = devices.load_manifest(GPU, "bf16")
    assert {r["dtype"] for r in bf} == {"bf16"}
    assert next(r for r in bf if r["kernel"] == a)["detail"] == "bf16 a"
    assert len(devices.load_manifest(GPU)) == 2 * len(bf), "unfiltered must return both precisions"


def test_a_row_written_before_the_column_reads_as_bf16(tmp_path, monkeypatch):
    """Every committed manifest predates the column. They must not become unreadable, and they
    must not be read as some other precision -- bf16 is what they are."""
    monkeypatch.setattr(devices, "_DEVICES", tmp_path)
    a, _ = _two_kernels()
    old = ["kernel,backend,family,file,status,detail",
           f"{a},triton,fam,some/file.py,ok,rel out=2.8e-03"]
    (tmp_path / f"{GPU}.csv").write_text("\n".join(old) + "\n")

    got = devices.load_manifest(GPU)
    assert [r["dtype"] for r in got] == ["bf16"]
    assert devices.load_manifest(GPU, "bf16")
    assert not devices.load_manifest(GPU, "fp32")


def test_a_run_that_names_no_precision_uses_the_drivers(rows, monkeypatch):
    """`record` is called from the bench path without a dtype. It must record the precision the
    process is actually running, not a constant."""
    pytest.importorskip("torch")
    from miniworld_engine.kernels import drivers

    a, _ = _two_kernels()
    devices.record(GPU, {a: (True, "x")})
    assert {r["dtype"] for r in rows()} == {drivers.DTYPE_MODE}


def test_the_committed_manifests_all_name_a_precision():
    """The repo's own files, not a fixture: they were backfilled when the column was added, and a
    blank there would silently become bf16 for a row that might not be."""
    for path in sorted(devices._DEVICES.glob("*.csv")):
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["kernel"] == devices._PROVENANCE:
                    continue
                assert (r.get("dtype") or "").strip() in ("bf16", "fp32"), (
                    f"{path.name}: {r['kernel']} has dtype {r.get('dtype')!r}")

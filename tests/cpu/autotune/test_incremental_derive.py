"""``dev derive`` is incremental: a unit records the files it depends on, and only the units whose files changed run again."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from miniworld_engine.autotune import derive, identity, plan

needs_monitoring = pytest.mark.skipif(sys.version_info < (3, 12), reason="derive traces with sys.monitoring (Python >= 3.12)")


def _load(path):
    spec = importlib.util.spec_from_file_location(f"fixture_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_monitoring
def test_a_unit_records_every_file_whose_code_ran_in_any_thread_and_the_data_it_opened(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    (tmp_path / "b.py").write_text("def g():\n    return 2\n")
    (tmp_path / "c.py").write_text("def h():\n    return 3\n")
    (tmp_path / "grid.csv").write_text("x\n1\n")
    monkeypatch.setattr(derive, "_PKG", tmp_path)
    a, b, _c = (_load(tmp_path / f"{n}.py") for n in "abc")
    with derive.trace_unit() as first:
        a.f()
    assert first.package_files()[0] == {"a.py"}
    with derive.trace_unit() as second:            # the same function again: a fresh unit records it again
        a.f()
        worker = threading.Thread(target=b.g)      # the autograd engine runs backward on threads of its own
        worker.start()
        worker.join()
        (tmp_path / "grid.csv").read_text()
    code, opened = second.package_files()
    assert code == {"a.py", "b.py"}
    assert opened == {"grid.csv"}
    assert "c.py" not in code                      # never called, never listed


@needs_monitoring
def test_an_executed_import_statement_is_recorded_by_name(monkeypatch):
    with derive.trace_unit() as seen:
        import importlib

        from miniworld_engine.kernels import _tiles  # noqa: F401
        importlib.import_module("miniworld_engine.kernels._tf32")
    assert "miniworld_engine.kernels._tiles" in seen.modules
    assert "miniworld_engine.kernels._tf32" in seen.modules


def test_a_units_file_set_holds_what_ran_what_it_imports_what_it_launched_and_the_safety_set():
    files = set(identity.unit_files(
        ["kernels/adaln/triton/inference.py"], ["miniworld_engine.kernels._tf32"], ["autotune/configs/grid/adaln_gemm_gate_triton.csv"],
        ["adaln_gemm_gate_triton"]))
    assert "kernels/adaln/triton/inference.py" in files                                # executed
    assert "kernels/_tiles.py" in files                                                  # a constant it imports at module level
    assert "kernels/_tf32.py" in files                                                   # an executed import
    assert "autotune/configs/grid/adaln_gemm_gate_triton.csv" in files                   # data it opened
    assert "kernels/drivers/adaln.py" in files                                           # the launched kernel's driver
    assert set(identity.unit_safety_files()) <= files
    # a file that did not run and is reached from nothing it ran is not in the set
    assert "kernels/rope/triton/main.py" not in files


def test_every_executed_file_is_in_the_recorded_set_whatever_else_is_added(tmp_path, monkeypatch):
    """The trace is a lower bound of the recorded set: nothing that ran can be missing from it."""
    monkeypatch.setattr(identity, "PKG", tmp_path)
    (tmp_path / "kernels").mkdir()
    for name in ("x.py", "y.py"):
        (tmp_path / "kernels" / name).write_text("A = 1\n")
    identity.refresh()
    try:
        files = identity.unit_files(["kernels/x.py", "kernels/y.py", "kernels/gone_since.py"], [], [], [])
    finally:
        identity.refresh()
    assert {"kernels/x.py", "kernels/y.py"} <= set(files)


# ----------------------------------------------------------------------------------------------- full run == incremental run
UNITS = [
    derive.DeriveUnit("m1", "pair", 128, (("c", 1),), "miniworld", "bfloat16", "", "eval", None),
    derive.DeriveUnit("m1", "pair", 256, (("c", 1),), "miniworld", "bfloat16", "", "train", None),
    derive.DeriveUnit("m2", "token", 128, (("c", 1),), "miniworld", "bfloat16", "", "eval", None),
    derive.DeriveUnit("m2", "token", 256, (("c", 1),), "miniworld", "bfloat16", "", "eval", None),
]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """A two-module package: m1 runs ``kernels/k1.py``, m2 runs ``kernels/k2.py``; every unit launches its module's kernel twice."""
    pkg = tmp_path / "pkg"
    (pkg / "kernels").mkdir(parents=True)
    (pkg / "settings.py").write_text("S = 1\n")
    (pkg / "kernels" / "k1.py").write_text("K = 1\n")
    (pkg / "kernels" / "k2.py").write_text("K = 1\n")
    monkeypatch.setattr(identity, "PKG", pkg)
    monkeypatch.setattr(derive, "_PKG", pkg)
    identity.refresh()
    ran: list[str] = []

    @contextlib.contextmanager
    def fake_trace():
        seen = derive.UnitTrace()
        derive._STATE["active"] = seen
        try:
            yield seen
        finally:
            derive._STATE["active"] = None

    def fake_record(unit, cases):
        ran.append(unit.label)
        derive._STATE["active"].code.add(str(pkg / "kernels" / f"k{unit.module[1]}.py"))
        op = f"op_{unit.module}"
        return [(op, "bfloat16", f"shape_key={unit.length}")] * 2 + [(f"other_{unit.module}", "bfloat16", "shape_key=1")], None

    from miniworld_engine.autotune import builder
    from miniworld_engine.modules import dispatch

    monkeypatch.setattr(derive, "units", lambda rows, arch=None: UNITS)
    monkeypatch.setattr(derive, "module_rows", list)
    monkeypatch.setattr(derive, "sm_tag", lambda: "sm86")
    monkeypatch.setattr(derive, "trace_unit", fake_trace)
    monkeypatch.setattr(derive, "record", fake_record)
    for name in ("require_environment", "target_arch", "install_no_calibration", "install_module_apply", "install_native_recorders"):
        monkeypatch.setattr(derive, name, lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "assume_every_card_recorded", lambda: None)
    monkeypatch.setattr(builder, "cases", lambda: [SimpleNamespace(name="m1"), SimpleNamespace(name="m2")])
    yield SimpleNamespace(pkg=pkg, ran=ran, tmp=tmp_path)
    identity.refresh()


def _run(fx, name, reuse=None, workers=1):
    out = fx.tmp / f"{name}.units.json"
    entries, skipped = derive.derive_all([], arch="sm86", per_unit=out, reuse=reuse, workers=workers)
    assert not skipped
    return entries, json.loads(out.read_text())


def test_a_full_run_and_an_incremental_run_publish_the_same_derivation(fixture):
    entries0, doc0 = _run(fixture, "first")
    assert len(fixture.ran) == 4
    assert plan.stale_units(doc0) == {}
    assert list(doc0["units"]) == sorted(doc0["units"])                                 # label order, whatever order the workers finished in
    assert doc0["schema"] == 2
    assert doc0["launch_counts"][UNITS[0].label] == [2, 1]      # the two launches of op_m1 are counted, not collapsed
    assert entries0[("op_m1", "bfloat16", "shape_key=128")]["units"] == 2               # ...so the registry's `units` column is what it was

    (fixture.pkg / "kernels" / "k1.py").write_text("K = 2\n")                           # code of the kernel m1 runs; m2 never ran it
    identity.refresh()
    stale = plan.stale_units(doc0)
    assert set(stale) == {UNITS[0].label, UNITS[1].label}
    reuse = plan.reusable_units(doc0, "sm86")
    assert set(reuse) == {UNITS[2].label, UNITS[3].label}

    fixture.ran.clear()
    entries_inc, doc_inc = _run(fixture, "incremental", reuse=reuse)
    assert sorted(fixture.ran) == sorted([UNITS[0].label, UNITS[1].label])              # only the stale units ran
    fixture.ran.clear()
    entries_full, doc_full = _run(fixture, "full")
    assert len(fixture.ran) == 4
    assert entries_inc == entries_full
    assert doc_inc == doc_full                                                          # byte-for-byte the same evidence


def test_a_comment_edit_and_an_unrelated_file_leave_every_unit_reusable(fixture):
    _, doc = _run(fixture, "first")
    (fixture.pkg / "kernels" / "k1.py").write_text("K = 1  # a comment\n")
    (fixture.pkg / "kernels" / "unrelated.py").write_text("U = 1\n")
    identity.refresh()
    assert plan.stale_units(doc) == {}
    fixture.ran.clear()
    _, again = _run(fixture, "second", reuse=plan.reusable_units(doc, "sm86"))
    assert fixture.ran == []
    for field in ("units", "launch_counts", "file_index", "unit_files"):
        assert again[field] == doc[field]                                               # nothing ran, nothing changed
    assert plan.stale_units(again) == {}                                                # and the tokens now carry the new bytes


def test_a_file_added_to_the_safety_set_stales_every_unit(fixture):
    _, doc = _run(fixture, "first")
    (fixture.pkg / "autotune").mkdir()
    (fixture.pkg / "autotune" / "cache.py").write_text("C = 1\n")
    identity.refresh()
    assert set(plan.stale_units(doc)) == {u.label for u in UNITS}


def test_a_registry_module_row_edit_leaves_every_other_unit_reusable(fixture):
    """``registry_module.csv`` says which units exist; what a unit takes from its row is its label. A unit that opened it does not list it."""
    (fixture.pkg / "kernels" / "registry").mkdir()
    csv = fixture.pkg / "kernels" / "registry" / "registry_module.csv"
    csv.write_text("module,stream\nm1,token_pair\n")
    identity.refresh()
    _, doc = _run(fixture, "first")
    csv.write_text("module,stream\nm1,token_pair\nm3,token_pair\n")
    identity.refresh()
    assert plan.stale_units(doc) == {}
    assert "kernels/registry/registry_module.csv" not in identity.unit_files([], [], ["kernels/registry/registry_module.csv"], [])


def test_all_reused_parallel_run_starts_no_worker(fixture):
    entries, doc = _run(fixture, "first")
    fixture.ran.clear()
    entries2, doc2 = _run(fixture, "second", reuse=plan.reusable_units(doc, "sm86"), workers=4)
    assert fixture.ran == []
    assert entries2 == entries
    assert doc2 == doc


def test_a_source_edited_while_the_units_ran_is_refused(fixture, monkeypatch):
    real = derive.record

    def editing(unit, cases):
        out = real(unit, cases)
        (fixture.pkg / "kernels" / "k2.py").write_text("K = 99\n")
        identity.refresh()
        return out

    monkeypatch.setattr(derive, "record", editing)
    with pytest.raises(ValueError, match="sources changed"):
        _run(fixture, "edited")


def test_dev_derive_twice_reuses_every_unit_and_rewrites_identical_files(fixture, capsys):
    import argparse

    from miniworld_engine import cli

    out = fixture.tmp / "registry_kernel.csv"
    args = argparse.Namespace(arch="sm86", out=str(out), per_unit="", workers=1, prior="", no_reuse=False)
    assert cli.cmd_derive(args) == 0
    assert "0 reused" in capsys.readouterr().out
    first = (out.read_bytes(), out.with_suffix(".units.json").read_bytes())
    assert plan.load("sm86", out)["complete"]

    fixture.ran.clear()
    assert cli.cmd_derive(args) == 0
    assert "4 reused" in capsys.readouterr().out
    assert fixture.ran == []
    assert (out.read_bytes(), out.with_suffix(".units.json").read_bytes()) == first          # byte-identical, nothing recorded

    (fixture.pkg / "kernels" / "k2.py").write_text("K = 2\n")
    identity.refresh()
    assert cli.cmd_derive(args) == 0
    assert "2 reused" in capsys.readouterr().out
    assert sorted(fixture.ran) == sorted([UNITS[2].label, UNITS[3].label])
    assert out.read_bytes() == first[0]                                                      # the keys did not move, so neither did the CSV
    assert not out.with_suffix(".units.json.new").exists()

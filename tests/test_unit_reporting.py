"""Both unit kinds must report the same things.

`build_all` decomposes work into OpUnits, so `--op` is the path a real sweep runs -- and it was the
path missing a summary. The `--case` path had the mirror-image hole: no `record_errors`, so a
capture failing silently stayed silent. Each kind must report the same things.
"""
from __future__ import annotations

from miniworld_engine.autotune import builder, capture


def test_the_reporter_prints_every_section(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(capture, "precompile_summary", lambda: "  [precompile] x")
    monkeypatch.setattr(capture, "summary", lambda: "  [compile-guard] z")
    monkeypatch.setattr(capture, "dump_shard", lambda p: 7)
    monkeypatch.setattr(capture, "record_errors", lambda: "")
    n = builder._report_unit(str(tmp_path / "s.json"))
    out = capsys.readouterr().out
    assert n == 7
    for want in ("[precompile]", "[compile-guard]"):
        assert want in out, f"{want} missing from the unit report"


def test_capture_failures_are_never_swallowed(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(capture, "precompile_summary", lambda: "")
    monkeypatch.setattr(capture, "summary", lambda: "")
    monkeypatch.setattr(capture, "dump_shard", lambda p: 0)
    monkeypatch.setattr(capture, "record_errors", lambda: "AttributeError x3")
    builder._report_unit(str(tmp_path / "s.json"))
    assert "recording failures: AttributeError x3" in capsys.readouterr().out


def test_both_child_paths_go_through_the_one_reporter():
    """Guards against the two paths drifting apart again."""
    import inspect
    src = inspect.getsource(builder._child_main)
    assert src.count("_report_unit(") == 2, (
        "each unit kind must report through _report_unit, not its own copy")
    for gone in ("capture.precompile_summary()", "capture.dump_shard(", "capture.record_errors()"):
        assert gone not in src, f"{gone} is still inlined in _child_main; it belongs in _report_unit"

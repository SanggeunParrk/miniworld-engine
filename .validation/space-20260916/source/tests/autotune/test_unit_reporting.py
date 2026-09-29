"""Both unit kinds must report the same things.

`build_all` decomposes work into OpUnits, so `--op` is the path a real sweep runs -- and it was the
path missing a summary. The `--case` path had the mirror-image hole: no `record_errors`, so a
capture failing silently stayed silent. Each kind must report the same things.
"""
from __future__ import annotations

import pytest

from miniworld_engine.autotune import builder, capture


def test_the_reporter_prints_every_section(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(capture, "precompile_summary", lambda: "  [precompile] x")
    monkeypatch.setattr(capture, "summary", lambda: "  [compile-guard] z")
    monkeypatch.setattr(capture, "dump_shard", lambda p, **kw: 7)
    monkeypatch.setattr(capture, "record_errors", lambda: "")
    n = builder._report_unit(str(tmp_path / "s.json"))
    out = capsys.readouterr().out
    assert n == 7
    for want in ("[precompile]", "[compile-guard]"):
        assert want in out, f"{want} missing from the unit report"


def test_capture_failures_are_never_swallowed(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(capture, "precompile_summary", lambda: "")
    monkeypatch.setattr(capture, "summary", lambda: "")
    monkeypatch.setattr(capture, "dump_shard", lambda p, **kw: 0)
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


@pytest.mark.parametrize(("ran", "errors", "expected"), [
    (False, "", False), (True, "record failed", False), (True, "", True),
])
def test_completion_requires_a_successful_run_and_capture(monkeypatch, tmp_path, ran, errors,
                                                        expected):
    seen = []
    monkeypatch.setattr(capture, "precompile_summary", lambda: "")
    monkeypatch.setattr(capture, "summary", lambda: "")
    monkeypatch.setattr(capture, "record_errors", lambda: errors)
    monkeypatch.setattr(capture, "dump_shard",
                        lambda path, **kw: seen.append(kw["unit_complete"]) or 1)
    builder._report_unit(str(tmp_path / "s.json"), complete=ran)
    assert seen == [expected]


@pytest.mark.parametrize("pin", [None, "atomic", "persistent"])
def test_module_child_matches_derived_dispatch(monkeypatch, tmp_path, pin):
    from types import SimpleNamespace

    import torch

    from miniworld_engine import settings
    from miniworld_engine.kernels.layernorm import compile_native, dispatch

    monkeypatch.setattr(settings, "_ACTIVE", settings.Settings())
    monkeypatch.setattr(builder, "cases", lambda: [SimpleNamespace(name="transition")])
    monkeypatch.setattr(capture, "install", lambda: None)
    monkeypatch.setattr(capture, "load_compile_state", lambda path: 0)
    monkeypatch.setattr(capture, "set_incremental", lambda value: None)
    monkeypatch.setattr(capture, "set_round_cache", lambda path: None)
    monkeypatch.setattr(capture, "shutdown_precompile", lambda: None)
    monkeypatch.setattr(builder, "_report_unit", lambda *a, **kw: 1)
    monkeypatch.setattr(dispatch, "lookup", lambda *a: "atomic")
    observed = []

    def run(*args, **kwargs):
        x = torch.empty(2, 768, dtype=torch.bfloat16)
        w = torch.empty(768, dtype=torch.float32)
        observed.append(compile_native._resolve_bwd_path(2, 768, x, x, w, w, w))
        assert settings.current().layernorm_dispatch == "off"
        assert settings.current().biasonly_dispatch == "off"
        return 1

    monkeypatch.setattr(builder, "run_case", run)
    argv = ["--case", "transition", "--mode", "train", "--shard", str(tmp_path / "s.json")]
    if pin:
        argv += ["--switch", "ln_bwd_path", "--value", pin]
    assert builder._child_main(argv) == 0
    assert observed == [pin or "persistent"]

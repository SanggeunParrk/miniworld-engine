"""`build all` with no flags must produce a complete cache, and one work list now is enough.

It was not, and the reason was a driver's WIDTH. `op_units` is declared coverage -- registry.csv x
level -- so every kernel with a driver gets tuned; but each was driven through its own harness, and
a harness's width constants were frozen at import while only its length could be overridden. The
sweep therefore reached one width per kernel, and every other width the model uses missed the cache
and fell back to the grid at runtime. Measured on an A6000: the resulting cache answers
`missing_pairs 0` to the declared question and misses 363 lookups the module matrix makes, across
42 of 91 ops.

The second pass -- `cases`, the module matrix -- existed to reach those, at the cost of a whole
second pass, and it reaches only 48 of the 91 triton kernels itself.

`driver_width` closes the hole at its source: a unit is (op, dtype, side, length, WIDTH), the
drivers read the base width from the environment exactly as they already read the length, and every
other width derives from it the way it does in the model (ND = n*D, NH = D//32, DC). So `build all`
is one pass again, and the module matrix is what `--per-module` asks for when you want to exercise
real dispatch paths -- not a requirement for coverage.
"""
from __future__ import annotations

import pytest

from miniworld_engine import cli


@pytest.fixture
def spy(monkeypatch, tmp_path):
    """Record every build_all call instead of running one."""
    calls = []

    def fake_build_all(selected, shard_dir, gpus, compile_jobs, **kw):
        calls.append({"n": len(selected), "kind": type(selected[0]).__name__,
                      "fill_gaps": kw.get("fill_gaps", False)})
        return [{"label": "u", "gpu": 0, "rc": 0, "ops": 1, "seconds": 1.0,
                 "shard": str(tmp_path / "s.json"), "log": ""}]

    from miniworld_engine.autotune import builder
    monkeypatch.setattr(builder, "build_all", fake_build_all)
    monkeypatch.setattr(builder, "device_sm", lambda: "sm_86")
    monkeypatch.setattr(cli, "_merge_built_shards", lambda args, results: 0)
    monkeypatch.setattr(cli, "_resolve_gpus", lambda g: [0])
    from miniworld_engine.autotune import derive, plan, preflight
    # These tests exercise scheduling and final certification. Refresh/dependency
    # execution has its own tests; a simulated stale certificate must not launch
    # thousands of real derivation units or import a machine-specific Flash backend.
    monkeypatch.setattr(plan, "ensure", lambda *a, **kw: tmp_path / "plan.csv")
    monkeypatch.setattr(preflight, "native_dependencies", lambda arch: None)
    monkeypatch.setattr(plan, "load", lambda *a: {"complete": True})
    monkeypatch.setattr(derive, "coverage", lambda *a: {"missing": []})
    monkeypatch.setattr(derive, "uncovered_kernels",
                        lambda arch: {"gated_projection_gate_triton"})
    return calls


def _args(shards, *extra, case="all"):
    """Parsed by the REAL parser, so the test cannot drift from the command it is about.

    ``shards`` is always the test's own tmp dir. It defaults to ``~/.cache/miniworld-build``, and
    `cmd_build` now refuses to re-bench units that directory already holds -- so a developer with a
    real half-finished build on the machine made these tests fail for a reason that had nothing to
    do with them.
    """
    return cli.build_parser().parse_args(["build", case, "--shards", str(shards), *extra])


def _run(args):
    rc = cli.cmd_build(args)
    assert rc == 0, rc


def test_the_default_is_the_module_sweep_alone(spy, tmp_path) -> None:
    """One pass, and it is the MODULE pass.

    It was the op pass, on the argument that the op sweep's coverage is DECLARED (registry.csv x
    a width ladder) while the module pass reaches only what some module dispatches to. The
    declaration was the problem: the ladders were hand-written in `op_units` and drifted from the
    shapes `cases()` actually runs -- token lengths stopped at 512 while the sweep ran to 1024,
    MSA widths held 64 while the config declares 64 and 128 -- and each drift was a bucket
    production reaches with no entry, found only by a replay on a card (146 of them).

    The module sweep is enumerated from `registry_module.csv` now, and `dev derive` runs that same
    enumeration on fake tensors to write `registry_kernel.csv`. So the coverage of this pass is
    not an argument any more: it is a file, and `dev coverage` diffs it against the cache with no
    GPU at all."""
    _run(_args(tmp_path))
    assert spy, "`build all` ran no pass at all"
    assert spy[0]["kind"] == "Case", spy
    # Building what is MISSING is the default. It was the reverse -- `--fill-gaps` was opt-in and
    # `--rebuild-cached` sat next to it -- and that arrangement cost 5h14m of an A6000 once,
    # re-timing 64 already-tuned units of layernorm_bwd_split to fill three missing keys.
    assert all(p["fill_gaps"] is True for p in spy), "a build should build what is missing"
    # A second pass is allowed, and only for the complement: the kernels `dev derive` shows no
    # module reaches. It is not a second statement of the same shapes -- that is what was wrong
    # with the old two-pass build -- it is the kernels the first pass provably cannot produce.
    assert len(spy) <= 2, f"`build all` ran {len(spy)} passes"
    if len(spy) == 2:
        assert spy[1]["kind"] == "OpUnit", spy


def test_the_op_sweep_drives_more_than_one_width(monkeypatch) -> None:
    """The whole reason one pass is enough. Without this the sweep tunes one width per kernel and
    the other widths the model uses fall back to the grid -- the 363 lookups the second pass
    existed to reach."""
    from miniworld_engine.autotune import builder
    from miniworld_engine.autotune.configs import config_set

    units = builder.op_units(config_dir=config_set("grid"))
    widths = {u.width for u in units}
    assert len(widths) > 1, f"the op sweep drives one width ({widths}); the module pass was for this"
    assert 0 not in widths, "a unit with no width leaves its driver at whatever it was frozen at"
    # and the width has to REACH the driver, or the unit list is a decoration
    one = next(u for u in units if u.width)
    assert one.env().get("MINIWORLD_DRIVER_WIDTH") == str(one.width), one.env()
    assert "--width" in one.cmd_args(), one.cmd_args()
    # two widths of the same (op, length) must be different units, or one overwrites the other
    stems = {u.stem for u in units}
    assert len(stems) == len(units), "two units share a shard stem; one would overwrite the other"


@pytest.mark.parametrize(("flag", "kind"), [("--per-op", "OpUnit"), ("--per-module", "Case")])
def test_an_explicit_flag_still_asks_for_one_pass(spy, tmp_path, flag, kind) -> None:
    _run(_args(tmp_path, flag))
    assert spy[0]["kind"] == kind
    if flag == "--per-module":
        # `all` still owes the drivers no module reaches on the target card.
        assert len(spy) == 2, spy
        assert spy[1]["kind"] == "OpUnit", spy
    else:
        assert len(spy) == 1, spy
    assert spy[0]["fill_gaps"] is True, "an explicit single pass still fills gaps by default"


def test_a_named_case_still_gets_its_single_module_pass(spy, tmp_path) -> None:
    """Two passes are what `build all` means, not what `build` means.

    `build <case>` names a module; `--per-op <kernel>` names a kernel. Running the op pass for a
    case name filters `op_units` by a name no kernel has, so it finds nothing and the command exits
    2 -- which is what `build gated_projection grid` did for one commit, having worked before it.
    """
    _run(_args(tmp_path, case="gated_projection"))
    assert len(spy) == 1, f"a named case ran {len(spy)} passes: {spy}"
    assert spy[0]["kind"] == "Case", spy
    assert spy[0]["fill_gaps"] is True, spy


def test_rebuild_is_the_only_way_to_remeasure_a_key_the_cache_answers(spy, tmp_path) -> None:
    """The other half of the default, and the reason it is safe to flip it.

    A default that only fills gaps is wrong if there is no way to say "measure it all again" --
    a new triton or a changed bench setting makes every committed number suspect. `--rebuild` is
    that word, and it is the only one: the old `--rebuild-cached` spelling maps onto it."""
    _run(_args(tmp_path, "--rebuild"))
    assert spy, "--rebuild ran no pass"
    assert all(p["fill_gaps"] is False for p in spy), spy
    spy.clear()
    _run(_args(tmp_path, "--rebuild-cached"))
    assert spy, "the old spelling ran no pass"
    assert all(p["fill_gaps"] is False for p in spy), "the old spelling stopped filling nothing"


def test_the_flag_reaches_the_child(tmp_path, monkeypatch) -> None:
    """The last untested link. `build_all(fill_gaps=True)` is checked above, and the child parses
    `--fill-gaps`; nothing checked that the runner in between puts it on the command line, and the
    two passes of `build all` differ by nothing else. A `blk64` smoke cannot catch it either: with
    one config per op, filling a gap and re-ranking a hit are the same work."""
    from miniworld_engine.autotune import builder

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        raise SystemExit(0)          # stop before anything launches

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    unit = builder.op_units({"gated_projection_gate_triton"})[0]
    for want in (True, False):
        seen.clear()
        shard_dir = tmp_path / f"s{want}"
        shard_dir.mkdir()                       # the runner claims the unit with O_EXCL in here
        with pytest.raises(SystemExit):
            builder._run_unit_subprocess(unit, 0, shard_dir, tmp_path, 1, fill_gaps=want)
        assert ("--fill-gaps" in seen["cmd"]) is want, seen["cmd"]


def test_the_driver_pass_runs_when_the_card_is_named(spy, tmp_path, monkeypatch) -> None:
    """`build all`'s second pass, exercised without a GPU.

    It was not exercised at all: `_driver_pass_for_uncovered` returns early when
    `builder.device_sm()` is None, which it is on every machine these tests run on, so the whole
    pass was dead code under test. It crashed on its first real launch -- `device_sm` spells the
    arch `sm_86` and both registries spell it `sm86`, so `uncovered_kernels` raised and all three
    build jobs exited without writing a shard. Naming the card is all it takes to cover it."""
    from miniworld_engine.autotune import builder as _builder

    monkeypatch.setattr(_builder, "device_sm", lambda: "sm_86")
    _run(_args(tmp_path))
    kinds = [p["kind"] for p in spy]
    assert kinds == ["Case", "OpUnit"], (
        f"`build all` on a named card ran {kinds}; it owes the module sweep and then the driver "
        f"sweep for the kernels no module reaches")


def test_successful_units_cannot_hide_missing_cache_keys(spy, tmp_path, monkeypatch):
    from miniworld_engine.autotune import builder, derive
    monkeypatch.setattr(builder, "device_sm", lambda: "sm_86")
    monkeypatch.setattr(derive, "coverage", lambda *a: {"missing": [("op", "float32|128")]})
    assert cli.cmd_build(_args(tmp_path)) == 1


def test_successful_units_cannot_certify_a_stale_plan(spy, tmp_path, monkeypatch):
    from miniworld_engine.autotune import builder, plan
    monkeypatch.setattr(builder, "device_sm", lambda: "sm_86")
    def stale(*args):
        raise ValueError("stale derivation")
    monkeypatch.setattr(plan, "load", stale)
    assert cli.cmd_build(_args(tmp_path)) == 1


@pytest.mark.parametrize(("rc", "ops", "skipped", "expected"), [
    (1, 1, False, 1), (0, 0, False, 1), (1, 0, True, 0),
])
def test_partial_driver_failure_is_not_hidden_by_module_coverage(
        spy, tmp_path, monkeypatch, rc, ops, skipped, expected):
    from miniworld_engine.autotune import builder, derive

    monkeypatch.setattr(derive, "uncovered_kernels", lambda arch: {"example_triton"})
    monkeypatch.setattr(builder, "op_units", lambda *a, **kw: [
        builder.OpUnit("example_triton", 128), builder.OpUnit("example_triton", 256)])
    merged = []

    def build(selected, *args, **kwargs):
        if isinstance(selected[0], builder.OpUnit):
            return [
                {"label": "example_triton[bfloat16] L=128", "rc": 0, "ops": 1, "log": ""},
                {"label": "example_triton[bfloat16] L=256", "rc": rc, "ops": ops,
                 "skipped": skipped, "log": "failed.log"},
            ]
        return [{"label": "module[miniworld/bfloat16]", "rc": 0, "ops": 1, "log": ""}]

    monkeypatch.setattr(builder, "build_all", build)
    monkeypatch.setattr(cli, "_merge_built_shards", lambda args, rows: merged.extend(rows) or 0)
    assert cli.cmd_build(_args(tmp_path)) == expected
    assert len(merged) == 3, "successful measurements must survive partial failures"


def test_claimed_elsewhere_is_pending_work():
    assert cli.is_bad_unit({"rc": 0, "ops": -1, "claimed_elsewhere": True})


@pytest.mark.parametrize("mixed", [False, True])
def test_build_summary_distinguishes_completed_skipped_and_held(
        spy, tmp_path, monkeypatch, capsys, mixed):
    from miniworld_engine.autotune import builder, derive

    held = {"label": "held[miniworld/bfloat16]", "rc": 0, "ops": -1,
            "claimed_elsewhere": True, "log": ""}
    rows = [held]
    if mixed:
        rows.extend([
            {"label": "ok[miniworld/bfloat16]", "rc": 0, "ops": 1, "log": ""},
            {"label": "empty[miniworld/bfloat16]", "rc": 0, "ops": 0, "log": "empty.log"},
            {"label": "failed[miniworld/bfloat16]", "rc": 1, "ops": 1, "log": "failed.log"},
            {"label": "skip[miniworld/bfloat16]", "rc": 1, "ops": 0,
             "skipped": True, "log": "skip.log"},
        ])
    monkeypatch.setattr(derive, "uncovered_kernels", lambda arch: set())
    monkeypatch.setattr(builder, "build_all", lambda *args, **kwargs: rows)
    assert cli.cmd_build(_args(tmp_path)) == 1
    output = capsys.readouterr()
    counts = "1 ok, 1 empty, 1 failed, 1 skipped" if mixed else "0 ok, 0 empty, 0 failed, 0 skipped"
    assert f"{counts}, 1 claimed elsewhere" in output.out
    assert "HELD  held[miniworld/bfloat16]" in output.out
    assert "1 units claimed elsewhere, with completion unverified" in output.err
    if mixed:
        assert "1 failed and 1 empty planned units" in output.err
    else:
        assert "failed or produced no measurements" not in output.err
        assert "failed and" not in output.err

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
    monkeypatch.setattr(cli, "_merge_built_shards", lambda args, results: 0)
    monkeypatch.setattr(cli, "_resolve_gpus", lambda g: [0])
    return calls


def _args(*extra, case="all"):
    """Parsed by the REAL parser, so the test cannot drift from the command it is about."""
    return cli.build_parser().parse_args(["build", case, *extra])


def _run(args):
    rc = cli.cmd_build(args)
    assert rc == 0, rc


def test_the_default_is_the_op_sweep_alone(spy) -> None:
    _run(_args())
    assert len(spy) == 1, f"`build all` ran {len(spy)} pass(es); the op sweep now covers the shape"
    assert spy[0]["kind"] == "OpUnit", spy
    assert spy[0]["fill_gaps"] is False, "the declared sweep must bench the full grid"


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
def test_an_explicit_flag_still_asks_for_one_pass(spy, flag, kind) -> None:
    _run(_args(flag))
    assert len(spy) == 1, spy
    assert spy[0]["kind"] == kind, spy
    assert spy[0]["fill_gaps"] is False, "an explicit single pass is the unmodified old behaviour"


def test_a_named_case_still_gets_its_single_module_pass(spy) -> None:
    """Two passes are what `build all` means, not what `build` means.

    `build <case>` names a module; `--per-op <kernel>` names a kernel. Running the op pass for a
    case name filters `op_units` by a name no kernel has, so it finds nothing and the command exits
    2 -- which is what `build gated_projection grid` did for one commit, having worked before it.
    """
    _run(_args(case="gated_projection"))
    assert len(spy) == 1, f"a named case ran {len(spy)} passes: {spy}"
    assert spy[0]["kind"] == "Case", spy
    assert spy[0]["fill_gaps"] is False, spy


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

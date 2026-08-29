"""`trunk` and `diffusion` overlap, and the overlap must be measured once, not twice.

registry.csv's `stack` column says which half of the model launches a kernel, and a kernel BOTH
halves launch says `both`. `op_units(stack=...)` includes those in either sweep -- deliberately: a
kernel the trunk launches has to be tuned for a trunk build whether or not the diffusion side
launches it too. Measured on the shipped registry: trunk 1269 units, diffusion 1353, union 1713,
so 909 units -- 53% of the two sweeps added together -- are in both.

A unit's identity is (op, dtype, side, length, width). No stack. So those 909 write the SAME shard
file whichever sweep produced them, and the second command re-benches every one of them unless
`--resume` filters them out first. Nothing said so: the build printed "1353 units" and ran them.

`cmd_build` now refuses instead. Not because it knows the shards are still good -- it cannot, and
that judgement is the operator's -- but so that making it takes a flag rather than forgetting one.
"""
from __future__ import annotations

import json

import pytest

from miniworld_engine import cli
from miniworld_engine.autotune import builder
from miniworld_engine.autotune.configs import config_set


@pytest.fixture
def spy(monkeypatch, tmp_path):
    calls = []

    def fake_build_all(selected, shard_dir, gpus, compile_jobs, **kw):
        calls.append(len(selected))
        return [{"label": "u", "gpu": 0, "rc": 0, "ops": 1, "seconds": 1.0,
                 "shard": str(tmp_path / "s.json"), "log": ""}]

    monkeypatch.setattr(builder, "build_all", fake_build_all)
    monkeypatch.setattr(cli, "_merge_built_shards", lambda args, results: 0)
    monkeypatch.setattr(cli, "_resolve_gpus", lambda g: [0])
    return calls


def _args(shards, case, *extra):
    return cli.build_parser().parse_args(["build", case, "--shards", str(shards), *extra])


def test_the_two_stacks_really_do_share_most_of_their_units() -> None:
    """The premise. If it stopped being true the guard below would be guarding nothing."""
    cd = config_set("grid")
    trunk = {u.stem for u in builder.op_units(config_dir=cd, stack="trunk")}
    diff = {u.stem for u in builder.op_units(config_dir=cd, stack="diffusion")}
    both = {u.stem for u in builder.op_units(config_dir=cd)}
    assert trunk & diff, "the stacks no longer overlap; this whole file is about the overlap"
    assert trunk | diff == both, (
        "trunk + diffusion is no longer the same work as `all`; one half now reaches a unit the "
        "full sweep does not, or the reverse")
    # Not a threshold on a number that may drift -- the claim is that the overlap is a big fraction
    # of either sweep, which is what makes re-benching it expensive rather than untidy.
    assert len(trunk & diff) > len(trunk) // 2


def _finish(shard_dir, unit) -> None:
    """Write the shard a completed unit leaves behind: entries, not just a file."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"{unit.stem}.json").write_text(json.dumps(
        {unit.op: {"entries": [{"config": {}, "ms": 1.0}], "op_id": 0}}))


def test_a_finished_unit_stops_the_second_stack(spy, tmp_path) -> None:
    cd = config_set("grid")
    shared = next(iter({u.stem for u in builder.op_units(config_dir=cd, stack="trunk")}
                       & {u.stem for u in builder.op_units(config_dir=cd, stack="diffusion")}))
    unit = next(u for u in builder.op_units(config_dir=cd, stack="diffusion") if u.stem == shared)
    _finish(tmp_path, unit)

    assert cli.cmd_build(_args(tmp_path, "diffusion")) == 2, (
        "the second stack re-benched a unit the first had already finished")
    assert not spy, "it should not have reached build_all at all"


def test_resume_is_how_you_say_skip_them(spy, tmp_path) -> None:
    cd = config_set("grid")
    units = builder.op_units(config_dir=cd, stack="diffusion")
    _finish(tmp_path, units[0])
    assert cli.cmd_build(_args(tmp_path, "diffusion", "--resume")) == 0
    assert spy, "--resume must let the build through; builder.build_all does the filtering"


def test_an_empty_shard_dir_is_not_a_refusal(spy, tmp_path) -> None:
    """A first build must not need a flag to say it is the first."""
    assert cli.cmd_build(_args(tmp_path, "trunk")) == 0
    assert spy == [len(builder.op_units(config_dir=config_set("grid"), stack="trunk"))]


def test_a_shard_with_no_entries_does_not_count_as_finished(spy, tmp_path) -> None:
    """`dump_shard` writes a file even when the unit measured nothing -- an unsupported shape, or a
    kernel that died before the first config. Those are exactly the units a restart must re-run, so
    they must not trigger the refusal either."""
    units = builder.op_units(config_dir=config_set("grid"), stack="trunk")
    (tmp_path / f"{units[0].stem}.json").write_text("{}")
    assert cli.cmd_build(_args(tmp_path, "trunk")) == 0
    assert spy

"""`trunk` and `diffusion` overlap, and the overlap must be measured once, not twice.

registry.csv's `stack` column says which half of the model launches a kernel, and a kernel BOTH
halves launch says `both`. `op_units(stack=...)` includes those in either sweep -- deliberately: a
kernel the trunk launches has to be tuned for a trunk build whether or not the diffusion side
launches it too. Measured on the shipped registry: trunk 1269 units, diffusion 1353, union 1713,
so 909 units -- 53% of the two sweeps added together -- are in both.

A unit's identity is (op, dtype, side, length, width). No stack. So those 909 write the SAME shard
file whichever sweep produced them, and the second command re-benches every one of them unless
`--resume` filters them out first. Nothing said so: the build printed "1353 units" and ran them.

`cmd_build` used to REFUSE instead (exit 2 unless `--resume` was passed), so that re-benching took
a flag rather than forgetting one. `--resume` is the DEFAULT now, which serves the same end better:
forgetting a flag skips the finished work rather than redoing it, and the flag you have to remember
(`--no-resume`) is the one that costs GPU-hours. What a default cannot decide -- whether the shards
are still good, which a kernel edit invalidates -- is printed, not assumed.
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
    # `mpnn` is a third stack and it IS disjoint from the other two. `both` means both STRUCTURE-model
    # halves -- a kernel those two share is launched by neither ProteinMPNN nor anything else --
    # so the sharing rule stops at the model boundary. It did not, and `build mpnn` spent its first
    # minute building gated_projection and layernorm at pair shapes.
    mpnn = {u.stem for u in builder.op_units(stack="mpnn", config_dir=cd)}
    assert trunk & diff, "the stacks no longer overlap; this whole file is about the overlap"
    assert mpnn, "the mpnn stack reaches no unit at all"
    assert not (mpnn & (trunk | diff)), (
        "an mpnn unit is reachable from a structure-model half; the two models share no kernel, "
        "and a `both` row belongs to that model's two halves only")
    assert trunk | diff | mpnn == both, (
        "the halves are no longer the same work as `all`; one half now reaches a unit the "
        "full sweep does not, or the reverse")
    # Not a threshold on a number that may drift -- the claim is that the overlap is a big fraction
    # of either sweep, which is what makes re-benching it expensive rather than untidy.
    assert len(trunk & diff) > len(trunk) // 2


def _finish(shard_dir, unit) -> None:
    """Write the shard a completed unit leaves behind: entries, not just a file."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"{unit.stem}.json").write_text(json.dumps(
        {unit.op: {"entries": [{"config": {}, "ms": 1.0}], "op_id": 0}}))


def test_a_finished_unit_is_skipped_by_the_second_stack(spy, tmp_path, capsys) -> None:
    cd = config_set("grid")
    shared = next(iter({u.stem for u in builder.op_units(config_dir=cd, stack="trunk")}
                       & {u.stem for u in builder.op_units(config_dir=cd, stack="diffusion")}))
    unit = next(u for u in builder.op_units(config_dir=cd, stack="diffusion") if u.stem == shared)
    _finish(tmp_path, unit)

    assert cli.cmd_build(_args(tmp_path, "diffusion")) == 0
    assert cli.build_parser().parse_args(
        ["build", "diffusion", "--shards", str(tmp_path)]).resume is True, (
        "the plain command stopped resuming -- it is the one run after a build is killed")
    assert "SKIPPING 1 of" in capsys.readouterr().err, (
        "the operator was not told which finished units this sweep is standing on; a kernel edited "
        "since they were written makes them measurements of code that no longer exists")


def test_no_resume_is_how_you_say_measure_it_all_again(spy, tmp_path, capsys) -> None:
    cd = config_set("grid")
    units = builder.op_units(config_dir=cd, stack="diffusion")
    _finish(tmp_path, units[0])
    assert cli.cmd_build(_args(tmp_path, "diffusion", "--no-resume")) == 0
    assert spy, "--no-resume must let the build through; builder.build_all does the filtering"
    assert "RE-RUNNING 1 of" in capsys.readouterr().err


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


def test_the_report_does_not_parse_every_shard(tmp_path) -> None:
    """It prints a count, and a sweep's shard directory is 2,079 files of megabytes on a shared
    filesystem. Parsing them cost 17 minutes of startup before a single unit ran -- twice, since
    `build_all` parses them again for the resume filter that actually decides anything."""
    import inspect

    src = inspect.getsource(cli._report_finished_units)
    assert "_shard_has_entries(" not in src, (
        "the startup count parses every shard again; the exact test belongs where it decides work")
    assert "st_size" in src
    assert "_EMPTY_SHARD_BYTES" in src


def test_the_cheap_test_still_separates_the_two_kinds_of_shard(tmp_path) -> None:
    """An empty shard is `{"_key_scheme": N}` and nothing else. The margin the size test relies on
    is four orders of magnitude; this pins that the two really are on opposite sides of it."""
    import json

    from miniworld_engine.autotune.builder import _shard_has_entries

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"_key_scheme": 3}))
    full = tmp_path / "full.json"
    full.write_text(json.dumps({"_key_scheme": 3, "op": {
        "grid": [], "op_id": "x",
        "entries": {"bfloat16|b1": [{"kwargs": {"BLOCK_M": 64}, "num_warps": 4,
                                     "num_stages": 2, "ms": 1.0}]}}}))
    assert empty.stat().st_size <= cli._EMPTY_SHARD_BYTES < full.stat().st_size
    assert not _shard_has_entries(empty)
    assert _shard_has_entries(full)

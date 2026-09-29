"""`--per-op` names kernels, and naming several must be one sweep rather than several commands.

It took exactly one name. Tuning a related set -- the twenty kernels that carry the tile visit-order
axis, say -- therefore meant twenty commands, and each one re-imports every kernel module before it
can even build its work list: minutes of triton compilation paid once per name, and a GPU pool that
drains and refills between them. The work list is also interleaved by op on purpose (see
`op_units`), so twenty separate lists lose the interleaving that stops eight cards compiling the
same keys at once.

A comma list is the whole change. One name is a list of one, so nothing about the old spelling moves.
"""
from __future__ import annotations

import pytest

from miniworld_engine import cli
from miniworld_engine.autotune import builder


@pytest.fixture
def spy(monkeypatch, tmp_path):
    calls = []

    def fake_build_all(selected, shard_dir, gpus, compile_jobs, **kw):
        calls.append(sorted({u.op for u in selected}))
        return [{"label": "u", "gpu": 0, "rc": 0, "ops": 1, "seconds": 1.0,
                 "shard": str(tmp_path / "s.json"), "log": ""}]

    monkeypatch.setattr(builder, "build_all", fake_build_all)
    monkeypatch.setattr(cli, "_merge_built_shards", lambda args, results: 0)
    monkeypatch.setattr(cli, "_resolve_gpus", lambda g: [0])
    return calls


def _args(shards, case, *extra):
    return cli.build_parser().parse_args(
        ["build", case, "--per-op", "--shards", str(shards), *extra])


TWO = "transition_fwd_b2b_triton,trimul_gemm_gate_triton"


def test_one_name_still_means_one_kernel(spy, tmp_path) -> None:
    assert cli.cmd_build(_args(tmp_path, "transition_fwd_b2b_triton")) == 0
    assert spy == [["transition_fwd_b2b_triton"]]


def test_a_comma_list_is_one_sweep_over_both(spy, tmp_path) -> None:
    assert cli.cmd_build(_args(tmp_path, TWO)) == 0
    assert spy == [sorted(TWO.split(","))], (
        "the two kernels must arrive in ONE build_all call; two calls is two pools")


def test_whitespace_around_a_name_is_not_a_new_kernel(spy, tmp_path) -> None:
    assert cli.cmd_build(_args(tmp_path, TWO.replace(",", ", "))) == 0
    assert spy == [sorted(TWO.split(","))]


def test_one_bad_name_in_a_list_stops_the_whole_command(spy, tmp_path, capsys) -> None:
    """Before anything imports, and naming the bad one. A typo in a twenty-name list must not cost
    the import of every kernel module to discover, nor build the nineteen and leave one out."""
    bad = f"{TWO},transition_fwd_b2b_tritn"
    assert cli.cmd_build(_args(tmp_path, bad)) == 2
    assert not spy
    said = capsys.readouterr().err
    assert "transition_fwd_b2b_tritn" in said, said
    assert "transition_fwd_b2b_triton" not in said.replace("transition_fwd_b2b_tritn", ""), (
        "the message should name the unknown kernel, not the ones that were fine")


def test_a_real_kernel_the_config_set_cannot_build_is_not_silently_dropped(spy, tmp_path,
                                                                          capsys) -> None:
    """`op_units` drops a kernel with no driver, or one this config set has no grid for. That is
    right for `all` and wrong for a named list: asking for two and getting one must not read as
    success."""
    import csv
    from pathlib import Path

    reg = Path(builder.__file__).resolve().parents[1] / "kernels" / "registry.csv"
    driverless = next(
        (r["kernel"] for r in csv.DictReader(reg.open())
         if r["backend"] == "triton" and not (r.get("driver") or "").strip()), None)
    if driverless is None:
        pytest.skip("every triton kernel has a driver now; nothing to drop")
    assert cli.cmd_build(_args(tmp_path, f"transition_fwd_b2b_triton,{driverless}")) == 2
    assert not spy
    assert driverless in capsys.readouterr().err

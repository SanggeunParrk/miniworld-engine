"""`build all` with no flags must produce a complete cache, and one work list is not enough.

Two lists exist and neither covers the other:

  * `op_units` -- declared coverage, registry.csv x level. Every kernel with a driver gets tuned,
    but each through its own driver, so the constexpr combinations a module's real dispatch
    produces (`SAVE_PREACT=1`, `ADD_RESIDUAL=0`, `H2=512,K=256`) never occur. Measured on an
    A6000: the resulting cache answers `missing_pairs 0` to the declared question and misses 363
    lookups the module matrix makes, across 42 of 91 ops.
  * `cases` -- the module matrix. Reaches those keys, reaches only 48 of the 91 triton kernels.

The default was the first list alone, so `build all` on a fresh card left the 363. Now it is both,
in order, with a merge between them, and the second runs with `fill_gaps` so a key the first pass
already tuned costs a 3-config re-rank rather than a full-grid sweep. That flag is not a detail:
without it the module pass re-benches everything the op sweep did, which is the 244 GPU-h that
made these an either/or.
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


def test_the_default_runs_the_op_sweep_then_the_module_matrix(spy) -> None:
    _run(_args())
    assert len(spy) == 2, f"`build all` ran {len(spy)} pass(es); one work list is not complete"
    assert spy[0]["kind"] == "OpUnit", spy
    assert spy[1]["kind"] == "Case", spy


def test_only_the_second_pass_fills_gaps(spy) -> None:
    """The op sweep must search the whole grid; the module pass must not repeat it."""
    _run(_args())
    assert spy[0]["fill_gaps"] is False, "the declared sweep must bench the full grid"
    assert spy[1]["fill_gaps"] is True, "without this the module pass re-benches 244 GPU-h"


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

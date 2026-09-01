"""`miniworld-engine build all` must work as typed, and must not throw away a run over one bad unit.

Two things made the documented entry point unusable:

  * ``config_type`` defaulted to the literal string "default", and ``configs/default`` has never
    existed in this repo -- so `build all` (and `bench module`, which hardcoded the same string)
    failed at argument resolution before doing any work;
  * one bad unit returned before ``merge_shards``, so a single OOMing shape or one kernel that
    hung its compiler discarded every good measurement in the run -- 526 of 527 units, unwritten.
"""
from __future__ import annotations

from pathlib import Path

from miniworld_engine import cli


def test_the_default_config_set_exists():
    """A default that resolves to nothing is not a default."""
    repo = Path(cli.__file__).resolve().parents[2]
    got = cli.resolve_config_dir(cli.DEFAULT_CONFIG_SET, repo)
    assert not isinstance(got, int), (
        f"the default config set {cli.DEFAULT_CONFIG_SET!r} does not resolve to a directory")
    assert got.is_dir(), f"{got} is not a directory"
    assert any(got.glob("*.csv")), f"{got} holds no <op>.csv"


def test_build_all_needs_no_second_word():
    """`build all` -- exactly as the goal states it -- must not require naming a config set.

    The parser supplies DEFAULT_CONFIG_SET for the optional second positional, so the check is
    that the name it supplies resolves to a real directory.
    """
    repo = Path(cli.__file__).resolve().parents[2]
    assert not isinstance(cli.resolve_config_dir(cli.DEFAULT_CONFIG_SET, repo), int)
    parsed = cli.build_parser().parse_args(["build", "all"])
    assert parsed.config_type == cli.DEFAULT_CONFIG_SET


def test_an_unknown_config_set_names_the_default():
    """The error a user actually hits should say what they could have typed."""
    repo = Path(cli.__file__).resolve().parents[2]
    assert cli.resolve_config_dir("no-such-set", repo) == 2


def _merge(monkeypatch, tmp_path, results, strict=False):
    """Drive the merge tail with a canned builder result, capturing whether it merged."""
    import argparse

    from miniworld_engine.autotune import capture
    merged: list = []
    (tmp_path / "u.json").write_text("{}")
    monkeypatch.setattr(capture, "merge_shards", lambda sh, **k: merged.append(list(sh)) or ["op"])
    # argparse.Namespace, not SimpleNamespace: _merge_built_shards is annotated for the
    # real thing, and a stand-in that merely walks like one hides a signature drift.
    args = argparse.Namespace(shards=str(tmp_path), strict=strict)
    rc = cli._merge_built_shards(args, results)
    return merged, rc


GOOD = {"rc": 0, "ops": 1, "label": "good", "log": "-"}
OOM = {"rc": 1, "ops": 0, "label": "oom-shape", "log": "-"}


def test_one_bad_unit_does_not_discard_the_whole_run(monkeypatch, tmp_path):
    """An OOMing shape must cost its own entry, not all 526 others."""
    merged, rc = _merge(monkeypatch, tmp_path, [GOOD, OOM])
    assert merged, "the good unit's shard must still be merged"
    assert rc == 0, "a skippable failure must not fail the build"


def test_strict_restores_all_or_nothing(monkeypatch, tmp_path):
    merged, rc = _merge(monkeypatch, tmp_path, [GOOD, OOM], strict=True)
    assert not merged
    assert rc == 1


def test_a_failing_run_does_not_unmake_earlier_shards(monkeypatch, tmp_path):
    """Superseded the old "every unit failed -> refuse to merge" rule, which is precisely what
    stranded 526 finished shards: with --resume, a late job's whole slice is leftovers."""
    merged, rc = _merge(monkeypatch, tmp_path, [OOM])
    assert merged, "shards already on disk are finished work"
    assert rc == 0


def test_a_clean_run_merges_and_succeeds(monkeypatch, tmp_path):
    merged, rc = _merge(monkeypatch, tmp_path, [GOOD, GOOD])
    assert merged
    assert rc == 0


# --------------------------------------------------------------------------- #
# a shape this card cannot hold is an answer, not a failure
#
# 52 units on the A6000 sweep ended with "OutOfMemoryError: Tried to allocate 16.00 GiB". Counting
# those as failures had two costs. A resumed job releases their claims, so every later job
# re-claimed the same OOMing units and produced nothing; and because such a job's units are ALL
# leftovers, it reported "0 ok, 9 failed" and refused to merge -- leaving 526 finished shards
# unwritten while the job exited rc=1.
# --------------------------------------------------------------------------- #

SKIP = {"rc": 1, "ops": 0, "label": "big-shape", "log": "-", "skipped": True}


def test_an_oom_skip_is_not_a_bad_unit(monkeypatch, tmp_path):
    merged, rc = _merge(monkeypatch, tmp_path, [GOOD, SKIP])
    reason = "a shape that cannot fit is a permanent answer, not a failure"
    assert merged, reason
    assert rc == 0, reason


def test_a_run_of_only_skips_still_merges_the_shards_on_disk(monkeypatch, tmp_path):
    """The exact case that stranded 526 shards: a resumed job whose whole slice OOMs."""
    merged, rc = _merge(monkeypatch, tmp_path, [SKIP, SKIP, SKIP])
    assert merged, "the shard dir decides what to merge, not this run's tally"
    assert rc == 0


def test_a_run_of_only_real_failures_still_merges_earlier_shards(monkeypatch, tmp_path):
    """Even genuine failures must not strand shards a previous run already produced."""
    merged, _rc = _merge(monkeypatch, tmp_path, [OOM, OOM])
    assert merged, "shards on disk are finished work; a later run's failures do not unmake them"


def test_no_shards_at_all_is_a_failure(monkeypatch, tmp_path):
    import argparse

    from miniworld_engine.autotune import capture
    called: list = []
    monkeypatch.setattr(capture, "merge_shards", lambda sh, **k: called.append(1) or [])
    args = argparse.Namespace(shards=str(tmp_path / "empty"), strict=False)
    assert cli._merge_built_shards(args, [OOM]) == 1
    assert not called


def test_no_build_skips_the_case_decomposition(monkeypatch, capsys):
    """`bench_module all` re-tuned 1,738 CASE units on top of a finished 922-unit `build all`.

    The two decompositions cover the same 91 ops; the case one is the older, redundant shape
    (more than half of its units re-tune a bucket another unit already covered). After a
    completed `build all` the pre-bench build is days of work for nothing, and there was no way
    to say so -- `--no-build` is that way.
    """
    import argparse

    from miniworld_engine.autotune import builder

    def explode(*a, **k):
        raise AssertionError("build_all ran despite --no-build")

    monkeypatch.setattr(builder, "build_all", explode)
    monkeypatch.setattr(cli, "apply_config_dir", lambda d: 0)
    repo = Path(cli.__file__).resolve().parents[2]
    args = argparse.Namespace(no_build=True, shards="/tmp/x", gpus="1", compile_jobs=1,
                              resume=False)
    rc = cli._bench_build_first(args, ("transition",), repo, {"transition": ("transition",)},
                                "MODULE_TARGETS")
    assert rc == 0
    assert "--no-build" in capsys.readouterr().out


def test_without_no_build_the_pre_bench_build_still_runs(monkeypatch):
    """The flag has to be opt-in: benching an untuned kernel measures a tuning run."""
    import argparse

    from miniworld_engine.autotune import builder

    called = []
    monkeypatch.setattr(builder, "build_all", lambda *a, **k: called.append(1) or [])
    monkeypatch.setattr(cli, "apply_config_dir", lambda d: 0)
    monkeypatch.setattr(cli, "_merge_built_shards", lambda a, r: 0)
    monkeypatch.setattr(builder, "cases", list)
    repo = Path(cli.__file__).resolve().parents[2]
    args = argparse.Namespace(no_build=False, shards="/tmp/x", gpus="1", compile_jobs=1,
                              resume=False)
    cli._bench_build_first(args, ("transition",), repo, {"transition": ("transition",)},
                           "MODULE_TARGETS")
    assert called, "build_all did not run without --no-build"


def test_the_sweep_axis_reaches_bench_py():
    """Without this the CLI could only ever sweep seq_len.

    bench.py's config defaults to `sweep_axis: seq_len` and the CLI never passed one, so the
    d_pair half of the matrix -- which docs/benchmarks.md and the README both call for, and where
    the width-dependent kernels separate -- was reachable only by invoking bench.py directly.
    """
    import argparse

    for axis in ("seq_len", "d_pair"):
        args = argparse.Namespace(impl="all", mode="inference", sweep_axis=axis)
        cmd, _ = cli._bench_cmd(args, "transition", None, level="module")
        assert f"sweep_axis={axis}" in cmd, cmd


def test_a_kernel_targets_mode_is_not_the_callers_to_choose():
    """`*_bwd` is training, everything else inference -- the name already says which."""
    import argparse

    args = argparse.Namespace(impl="all", mode="inference", sweep_axis="seq_len")
    bwd, _ = cli._bench_cmd(args, "layernorm_bwd", None, level="kernel")
    fwd, _ = cli._bench_cmd(args, "layernorm", None, level="kernel")
    assert "mode=training" in bwd
    assert "mode=inference" in fwd


def test_case_names_are_declared() -> None:
    """`CASE_NAMES` must be exactly what `cases()` builds, in order.

    It exists so `miniworld-engine build <typo>` can be rejected without importing anything:
    `cases()` constructs the production modules, which imports every kernel, so the old code spent
    minutes of triton compilation before printing "unknown case". A declared list is only safe if
    it cannot drift from the thing it stands in for, which is what this asserts.
    """
    from miniworld_engine.autotune.builder import CASE_NAMES, cases

    assert tuple(c.name for c in cases()) == CASE_NAMES


def test_build_rejects_an_unknown_case_without_importing_kernels(capsys) -> None:
    """The rejection has to come BEFORE the imports, or it is not a fast failure.

    `sys.modules` is the check: if resolving the name pulled in a kernel module, the guard ran too
    late and the user waited for it.
    """
    import argparse
    import sys

    repo = Path(cli.__file__).resolve().parents[2]
    before = set(sys.modules)
    rc = cli._reject_unknown_build_target(
        argparse.Namespace(case="triangle_multiplicaton", per_op=False), repo)
    assert rc == 2
    out = capsys.readouterr().err
    assert "unknown case" in out
    assert "triangle_multiplication" in out, "the message must list what IS valid"
    new_kernel_imports = [m for m in set(sys.modules) - before
                          if m.startswith("miniworld_engine.kernels")]
    assert not new_kernel_imports, new_kernel_imports


def test_build_accepts_every_declared_case_and_every_registered_op() -> None:
    """Both name spaces `build` takes, checked against their declarations."""
    import argparse
    import csv

    from miniworld_engine.autotune.builder import CASE_NAMES

    repo = Path(cli.__file__).resolve().parents[2]
    for name in CASE_NAMES:
        assert cli._reject_unknown_build_target(
            argparse.Namespace(case=name, per_op=False), repo) == 0, name
    registry = repo / "src" / "miniworld_engine" / "kernels" / "registry.csv"
    with registry.open() as fh:
        ops = [row["kernel"] for row in csv.DictReader(fh)]
    assert ops, "registry.csv is empty"
    for op in ops:
        assert cli._reject_unknown_build_target(
            argparse.Namespace(case=op, per_op=True), repo) == 0, op


def test_capture_argv_names_the_target_and_its_level() -> None:
    """`dev capture` builds bench.py's argv by hand, and bench.py needs BOTH halves of a target's
    identity: `target=` alone would be ambiguous the moment a kernel and a module share a name."""
    from pathlib import Path as _Path

    jobs = cli.build_jobs(("augmented_attention_atom", "conditioned_transition"),
                          _Path("/tmp/shards"), sweep_dispatch=True)
    atom = next(j for j in jobs if j.target == "augmented_attention_atom").bench_args("miniworld")
    assert "target=augmented_attention_atom" in atom
    assert "level=module" in atom
    # the atom ladder comes from SHAPES, not the token default
    assert "max_seq_len=384" in atom, atom
    cond = next(j for j in jobs if j.target == "conditioned_transition").bench_args("miniworld")
    # fp32 is this family's declared io dtype, so the target still pins it. d_single_token is NOT
    # pinned any more: the bench builds ConditionedTransition(768, 384) -- the model's `token_dit`,
    # AlphaFold-3's c_token conditioned on c_s -- and `d_single_token=384` forced that to 384/384,
    # a square combination no model config declares.
    assert "precision=32" in cond, cond
    assert not [a for a in cond if a.startswith("d_single_token=")], cond

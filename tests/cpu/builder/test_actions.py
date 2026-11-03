"""The release actions' own logic (``build.actions`` and their CLI): the parser has no way to widen a scope, ``--all`` needs a reason, ``wheel`` and
``release`` refuse without upstream records for this exact tree, the release gate's checks (bundle names, built files, run_all evidence), the
whole-bundle row audit, the reviewed prune, and the junit mapping. No GPU, no graph build."""
from __future__ import annotations

import argparse
import json
import types

import pytest

from miniworld_engine import cli
from miniworld_engine.build import action_records as rec
from miniworld_engine.build import actions, kernel_bundle

REPO = rec.repo_root()
PUBLIC = ("status", "test", "bundle", "verify", "derive", "bench", "wheel", "release")
WIDENING = {"only", "path", "paths", "coverage", "family", "families", "filter", "row", "kernel", "target", "arch", "out"}


def subparsers() -> dict[str, argparse.ArgumentParser]:
    parser = cli.build_parser()
    top = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return top.choices


@pytest.mark.parametrize("name", PUBLIC)
def test_every_action_is_a_command_and_none_takes_a_scope_argument(name):
    p = subparsers()[name]
    dests = {a.dest for a in p._actions}
    assert not dests & WIDENING, f"`{name}` must not let the caller pick a scope: {dests & WIDENING}"
    assert all(not a.option_strings or a.option_strings[0].startswith("-") for a in p._actions)
    assert not any(a.nargs in ("*", "+", argparse.REMAINDER) for a in p._actions), "no positional path list"


def test_a_full_run_needs_all_and_a_reason_through_the_parser():
    for name in ("test", "bundle", "verify", "derive", "bench"):
        args = cli.build_parser().parse_args([name, "--all", "--reason", "torch upgrade"])
        assert args.all
        assert args.reason == "torch upgrade"
        with pytest.raises(rec.Refused):
            actions.Ctx(cli.build_parser().parse_args([name, "--all"]), graph=False)
        with pytest.raises(rec.Refused):
            actions.Ctx(cli.build_parser().parse_args([name, "--reason", "x"]), graph=False)
    assert cli.main(["test", "--all"]) == 2             # refused through the command, before any graph or GPU work


def test_the_low_level_commands_are_still_there():
    dev = next(a for a in subparsers()["dev"]._actions if isinstance(a, argparse._SubParsersAction)).choices
    for name in ("bundle-kernels", "bundle-verify", "build-wheel", "derive", "code-graph"):
        assert name in dev


@pytest.fixture
def records(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIWORLD_ENGINE_RECORDS", str(tmp_path))
    return tmp_path


def _write(action, tree, result="pass", **extra):
    return rec.write_record(action, "sm90", tree, base="v", scope={}, result=result, version=rec.declared_version(REPO), root=None, extra=extra)


def test_wheel_refuses_without_bundle_and_verify_records_and_names_the_action(records, capsys):
    assert cli.main(["wheel"]) == 2
    assert "Run: miniworld-engine bundle" in capsys.readouterr().out
    tree = rec.tree_hash(REPO)
    _write("bundle", tree)
    assert cli.main(["wheel"]) == 2
    assert "Run: miniworld-engine verify" in capsys.readouterr().out
    _write("verify", "0" * 64)                                  # a verify of another tree does not count
    assert cli.main(["wheel"]) == 2
    assert "the code changed since" in capsys.readouterr().out


def test_wheel_refuses_a_bundle_that_changed_after_it_was_verified(records, capsys, monkeypatch):
    tree = rec.tree_hash(REPO)
    _write("bundle", tree)
    _write("verify", tree, bundle_digest="old")
    monkeypatch.setattr(actions, "prebuilt_manifests", lambda root, arch: [])
    assert cli.main(["wheel"]) == 2
    assert "bundle changed since it was verified" in capsys.readouterr().out


def test_release_refuses_without_a_wheel_record(records, capsys):
    assert cli.main(["release", "--confirm"]) == 2
    assert "Run: miniworld-engine wheel" in capsys.readouterr().out


def _release_setup(records, monkeypatch, tmp_path, names=None):
    tree, version = rec.tree_hash(REPO), rec.declared_version(REPO)
    prev = kernel_bundle.read_name_list(REPO / actions.PREV_NAMES)
    whl = tmp_path / "miniworld_engine-0-py3-none-any.whl"
    whl.write_bytes(b"wheel bytes")
    _write("bundle", tree, bundle_digest="D")
    _write("verify", tree, bundle_digest="D")
    _write("wheel", tree, bundle_digest="D", names=sorted(prev if names is None else names),
           files={whl.name: {"path": str(whl), "sha256": rec.file_digest(whl)}})
    monkeypatch.setattr(actions, "release_evidence", lambda repo, v: ["sm90"])
    return whl, version


def test_release_checks_everything_and_never_uploads_without_confirm(records, tmp_path, capsys, monkeypatch):
    _release_setup(records, monkeypatch, tmp_path)
    called = []
    monkeypatch.setattr(actions, "upload", lambda paths: called.append(paths) or 0)
    assert cli.main(["release"]) == 0
    out = capsys.readouterr().out
    assert "Nothing uploaded" in out
    assert not called


def test_release_refuses_a_file_that_changed_after_the_wheel_record(records, tmp_path, capsys, monkeypatch):
    whl, _ = _release_setup(records, monkeypatch, tmp_path)
    whl.write_bytes(b"tampered")
    assert cli.main(["release", "--confirm"]) == 2
    assert "changed or are gone" in capsys.readouterr().out


def test_release_refuses_a_dropped_name_the_allowlist_does_not_explain(records, tmp_path, capsys, monkeypatch):
    _release_setup(records, monkeypatch, tmp_path, names=[])
    assert cli.main(["release", "--confirm"]) == 2
    assert "dropped since the previous release" in capsys.readouterr().out


def test_release_refuses_a_wheel_not_built_from_the_verified_bundle(records, tmp_path, capsys, monkeypatch):
    _release_setup(records, monkeypatch, tmp_path)
    tree = rec.tree_hash(REPO)
    _write("verify", tree, bundle_digest="OTHER")
    assert cli.main(["release", "--confirm"]) == 2
    assert "not built from the verified bundle" in capsys.readouterr().out


def test_release_needs_the_run_all_evidence(records, tmp_path, capsys, monkeypatch):
    _release_setup(records, monkeypatch, tmp_path)
    monkeypatch.setattr(actions, "release_evidence", lambda repo, v: [])
    assert cli.main(["release", "--confirm"]) == 2
    assert "run_all" in capsys.readouterr().out


def test_release_uploads_only_with_confirm(records, tmp_path, capsys, monkeypatch):
    whl, _ = _release_setup(records, monkeypatch, tmp_path)
    uploaded = []
    monkeypatch.setattr(actions, "upload", lambda paths: uploaded.append(paths) or 0)
    assert cli.main(["release", "--confirm"]) == 0
    assert uploaded == [[str(whl)]]


def test_upload_is_twine(monkeypatch):
    cmds = []
    monkeypatch.setattr(actions.subprocess, "run", lambda cmd, **k: cmds.append(cmd) or types.SimpleNamespace(returncode=0))
    assert actions.upload(["a.whl"]) == 0
    assert cmds == [[actions.sys.executable, "-m", "twine", "upload", "a.whl"]]


def test_run_all_evidence_is_read_from_the_committed_manifests():
    version = rec.declared_version(REPO)
    assert isinstance(actions.release_evidence(REPO, version), list)
    assert actions.release_evidence(REPO, "0.0.0-none") == []


# ------------------------------------------------------------------------------------------------------------------ audit / prune / junit
def test_the_row_audit_reports_unrecorded_rows_and_missing_artifacts():
    rep = actions.audit_rows({"r1": ["a", "b"], "r2": ["c"], "old": ["z"]}, {"r1", "r2", "r3"}, {"a", "b"})
    assert rep["no_record"] == ["r3"]
    assert rep["missing"] == {"r2": ["c"]}
    assert rep["stale_record"] == ["old"]
    assert not rep["ok"]
    assert actions.audit_rows({"r1": ["a"]}, {"r1"}, {"a", "extra"})["ok"]


def test_a_clean_row_replaces_its_record_and_an_unclean_one_loses_it(tmp_path, monkeypatch):
    mf = tmp_path / "sm90-cu129" / "manifest.json"
    mf.parent.mkdir()
    mf.write_text(json.dumps({"artifacts": {"k": {"name": "ka", "sha256": "s1"}}}))
    monkeypatch.setattr(actions, "prebuilt_manifests", lambda root, arch: [mf])
    actions.merge_row_record("sm90", tmp_path, {"a/x/d=1": {"text": "t", "artifacts": ["ka"], "clean": True, "missing": []},
                                                "b/x/d=1": {"text": "t", "artifacts": [], "clean": False, "missing": ["kz"]}})
    got = actions.read_row_record("sm90", tmp_path)
    assert list(got) == ["a/x/d=1"]
    assert got["a/x/d=1"]["served"] == actions._fingerprint("sm90", tmp_path)("a/x/d=1")
    mf.write_text(json.dumps({"artifacts": {"k": {"name": "ka", "sha256": "s2"}}}))        # the bundle rebuilt ka: the row is owed a verify
    assert actions._fingerprint("sm90", tmp_path)("a/x/d=1") != got["a/x/d=1"]["served"]
    actions.merge_row_record("sm90", tmp_path, {"a/x/d=1": {"text": "t", "artifacts": ["ka"], "clean": False, "missing": ["ka"]}})
    assert actions.read_row_record("sm90", tmp_path) == {}


def test_prune_removes_only_the_named_artifacts():
    manifest = {"artifacts": {"k1": {"name": "a", "path": "artifacts/a.so", "sidecars": [".json"]}, "k2": {"name": "b", "path": "artifacts/b.so"}}}
    new, gone = actions.prune_names(manifest, {"a"})
    assert list(new["artifacts"]) == ["k2"]
    assert gone == ["artifacts/a.so", "artifacts/a.json"]


def test_prune_reviewed_refuses_a_name_nobody_reviewed(tmp_path, capsys):
    names = tmp_path / "n.txt"
    names.write_text("not_a_reviewed_name\n")
    assert actions.prune_reviewed([], names, REPO) == 2
    assert "not in the reviewed list" in capsys.readouterr().out


def test_name_diff_explains_drops_only_through_the_allowlist():
    prev = kernel_bundle.read_name_list(REPO / actions.PREV_NAMES)
    assert actions.name_diff(set(prev), REPO)["dropped"] == []
    d = actions.name_diff(set(), REPO)
    allowed = kernel_bundle.read_name_list(REPO / actions.ALLOWED_DROPS)
    assert set(d["dropped"]) == prev - allowed
    assert set(d["allowed"]) == prev & allowed


def test_junit_maps_results_to_files(tmp_path):
    xml = tmp_path / "r.xml"
    xml.write_text('<testsuites><testsuite><testcase classname="tests.cpu.a.test_x" name="t1"/>'
                   '<testcase classname="tests.cpu.a.test_x.TestC" name="t2"><failure/></testcase>'
                   '<testcase classname="tests.cpu.a.test_y" name="t3"/></testsuite></testsuites>')
    got = actions._junit_files(xml, ["tests/cpu/a/test_x.py", "tests/cpu/a/test_y.py", "tests/cpu/a/test_z.py"])
    assert got == {"tests/cpu/a/test_x.py": False, "tests/cpu/a/test_y.py": True, "tests/cpu/a/test_z.py": False}
    assert actions._junit_files(tmp_path / "missing.xml", ["tests/cpu/a/test_x.py"]) == {"tests/cpu/a/test_x.py": False}


# ------------------------------------------------------------------------------------------------------------------ the internals refuse outside an action
@pytest.mark.parametrize("argv", [["dev", "build-wheel"], ["dev", "derive", "--arch", "sm90"], ["bench_module", "transition"]])
def test_an_internal_refuses_outside_an_action(argv, monkeypatch, capsys):
    monkeypatch.delenv(cli.ACTION_ENV, raising=False)
    monkeypatch.delenv(cli.INTERNAL_ENV, raising=False)
    args = cli.build_parser().parse_args(argv)
    assert "internal of `miniworld-engine" in cli.internal_refusal(args)
    monkeypatch.setenv(cli.ACTION_ENV, "bundle")                                 # an action runs it
    assert cli.internal_refusal(args) is None
    monkeypatch.delenv(cli.ACTION_ENV)
    monkeypatch.setenv(cli.INTERNAL_ENV, "debugging one kernel")                 # a deliberate direct run, with its reason
    assert cli.internal_refusal(args) is None
    assert "debugging one kernel" in capsys.readouterr().err


def test_an_ordinary_command_is_not_an_internal(monkeypatch):
    monkeypatch.delenv(cli.ACTION_ENV, raising=False)
    monkeypatch.delenv(cli.INTERNAL_ENV, raising=False)
    assert cli.internal_refusal(cli.build_parser().parse_args(["dev", "code-graph", "--diff", "v0"])) is None


def test_pytest_refuses_a_directory_outside_an_action():
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("MINIWORLD_ENGINE_ACTION", "CI", "MINIWORLD_ENGINE_INTERNAL")}
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--collect-only", "tests/cpu/builder"], cwd=REPO,
                         env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 4, run.stdout[-500:] + run.stderr[-500:]
    assert "a directory is not a test run here" in run.stdout + run.stderr
    one = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--collect-only", "tests/cpu/builder/test_actions.py"],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert one.returncode == 0, one.stdout[-500:] + one.stderr[-500:]            # naming a file is always allowed


def test_an_unchanged_failure_is_not_rerun_and_not_forgotten(records):
    ctx = actions.Ctx(cli.build_parser().parse_args(["status"]), graph=False)
    ctx.action = "test"
    rec.save_state("test", ctx.arch, "T0", [], records, failing=["tests/cpu/a.py", "tests/cpu/b.py"])
    # this run reran a.py (a change reached it) and it passed; b.py was not rerun; c.py failed now
    assert ctx.still_failing(["tests/cpu/a.py", "tests/cpu/c.py"], ["tests/cpu/c.py"]) == ["tests/cpu/b.py", "tests/cpu/c.py"]
    ctx.start = "T1"
    path = ctx.finish("test", None, "pass", owed=[], failing=["tests/cpu/b.py"])
    assert json.loads(path.read_text())["result"] == "fail"                    # a failure nobody reran is still a failure
    assert rec.load_state("test", ctx.arch, records) == {**rec.load_state("test", ctx.arch, records), "tree": "T1", "failing": ["tests/cpu/b.py"]}


def test_retry_failing_reruns_the_recorded_failures_and_nothing_else(records):
    args = cli.build_parser().parse_args(["test", "--retry-failing"])
    assert args.retry_failing
    ctx = actions.Ctx(cli.build_parser().parse_args(["status"]), graph=False)
    ctx.action = "test"
    rec.save_state("test", ctx.arch, "T0", ["tests/cpu/notrun.py"], records, failing=["tests/cpu/f.py"])
    rec.save_ledger("test", ctx.arch, {"tests/cpu/f.py": "d", "tests/cpu/ok.py": "d"}, records)
    assert ctx.owed() == ["tests/cpu/notrun.py"]
    assert set(ctx.ledger("test")) == {"tests/cpu/f.py", "tests/cpu/ok.py"}
    ctx.retry = True
    assert ctx.owed() == ["tests/cpu/notrun.py", "tests/cpu/f.py"]
    assert set(ctx.ledger("test")) == {"tests/cpu/ok.py"}


def test_a_module_level_skip_is_not_a_failure(tmp_path):
    xml = tmp_path / "r.xml"
    xml.write_text('<testsuites><testsuite><testcase classname="" name="tests.cpu.a.test_skipped" time="0"><skipped message="no src"/>'
                   '</testcase></testsuite></testsuites>')
    assert actions._junit_files(xml, ["tests/cpu/a/test_skipped.py"]) == {"tests/cpu/a/test_skipped.py": True}


def test_a_failure_that_no_longer_exists_is_not_failing(records):
    ctx = actions.Ctx(cli.build_parser().parse_args(["status"]), graph=False)
    ctx.action = "bundle"
    rec.save_state("bundle", ctx.arch, "T0", [], records, failing=["row:gone/x/d=1/A=1|1", "row:kept/x/d=1/A=1|1"])
    assert ctx.still_failing([], [], alive=["row:kept/x/d=1/A=1|1"]) == ["row:kept/x/d=1/A=1|1"]
    assert ctx.still_failing([], ["row:new/x/d=1/A=1|1"], alive=[]) == ["row:new/x/d=1/A=1|1"]       # this run's own failure always counts


def test_the_audit_lists_the_rows_served_by_no_engine_kernel(monkeypatch, tmp_path):
    mf = tmp_path / "sm90-cu129" / "manifest.json"
    mf.parent.mkdir()
    mf.write_text(json.dumps({"artifacts": {"k": {"name": "ka", "sha256": "s"}}}))
    monkeypatch.setattr(actions, "prebuilt_manifests", lambda root, arch: [mf])
    monkeypatch.setattr(actions, "read_row_record", lambda arch, repo: {"a/x/d=1/A=1|1": {"text": "t", "artifacts": ["ka"]},
                                                                        "b/x/d=1/A=1|1": {"text": "t", "artifacts": []}})
    from miniworld_engine.build import registry_calls

    rows = [types.SimpleNamespace(key=k, text_digest="t", calls=[1]) for k in ("a/x/d=1/A=1|1", "b/x/d=1/A=1|1")]
    monkeypatch.setattr(registry_calls, "expand", lambda arch: rows)
    audit = actions.audit_bundle("sm90", tmp_path)
    assert audit["ok"]
    assert audit["no_engine_kernel"] == ["b/x/d=1/A=1|1"]


def test_derive_owes_a_new_rows_units_and_drops_a_removed_rows(tmp_path, monkeypatch):
    """The plan is judged against the units the registry asks for now, not only against the units it recorded."""
    from miniworld_engine.autotune import identity, plan

    monkeypatch.setattr(actions, "registry_units", lambda card: ["kept", "new"])
    monkeypatch.setattr(identity, "unit_safety_files", list)
    path = tmp_path / "units.json"
    path.write_text(json.dumps({"schema": plan.SCHEMA, "units": {"kept": {}, "gone": {}}, "file_index": [], "file_tokens": [],
                                "unit_files": {"kept": [], "gone": []}, "norm_scheme": identity.norm_scheme()}))
    assert actions.plan_stale("sm90", path) == {"new": ["<no recorded files>"], "gone": ["<row removed from registry_module.csv>"]}
    assert set(actions.plan_stale("sm90", tmp_path / "absent.json")) == {"kept", "new"}


def test_a_derived_cubin_is_replayed_with_its_base():
    base = {"key": "b", "name": "trimul_h100/src", "recipe": {"fn": "build"}}
    variant = {"key": "v", "name": "trimul_h100/src.pool120", "recipe": {"fn": "derived", "base_key": "b"}}
    recipes = {base["name"]: [base], variant["name"]: [variant]}
    assert actions.with_bases([variant], recipes) == [base, variant]
    assert actions.with_bases([base, variant], recipes) == [base, variant]      # a base already asked for is not added twice


def test_a_nothing_bundle_record_carries_the_bundle_digest_the_release_matches(records, tmp_path, capsys, monkeypatch):
    """Committing the run_all manifest makes a new tree on which `bundle` owes nothing; its "nothing" record must still name the bundle
    digest, or `release` cannot match the wheel with the bundle it was verified on."""
    ctx = actions.Ctx(cli.build_parser().parse_args(["bundle"]), graph=False)
    ctx.nothing("bundle", None, {"bundle_digest": "D"})
    got = rec.latest("bundle", ctx.arch, tree=ctx.tree, version=ctx.version, root=ctx.records)
    assert got["result"] == "nothing"
    assert got["bundle_digest"] == "D"
    _release_setup(records, monkeypatch, tmp_path)
    _write("bundle", rec.tree_hash(REPO), result="nothing", bundle_digest="D")
    assert cli.main(["release"]) == 0
    assert "every check passed" in capsys.readouterr().out

"""The records and ledgers the release actions share (``build.action_records``): a record round-trips, a downstream action refuses without an
upstream record for the exact tree and version and names the action to run, ``--all`` needs a reason, the tree hash follows the shipped bytes."""
from __future__ import annotations

import datetime
import subprocess

import pytest

from miniworld_engine.build import action_records as rec

T0 = datetime.datetime(2026, 11, 3, 12, 0, 0, tzinfo=datetime.timezone.utc)


def test_a_record_round_trips(tmp_path):
    path = rec.write_record("bundle", "sm90", "a" * 64, base="v2.4.3", scope={"items": 3}, result="pass", version="2.4.4", extra={"names": ["x"]}, root=tmp_path, now=T0)
    assert tmp_path in path.parents
    assert "aaaaaaaaaaaa" in path.name
    (back,) = rec.read_records("bundle", "sm90", tmp_path)
    assert (back["action"], back["arch"], back["tree"], back["base"], back["result"], back["version"]) == ("bundle", "sm90", "a" * 64, "v2.4.3", "pass", "2.4.4")
    assert back["scope"] == {"items": 3}
    assert back["names"] == ["x"]
    assert back["all"] is None
    assert "torch" in back["versions"]
    assert rec.read_records("bundle", "sm100", tmp_path) == []
    assert rec.read_records("verify", "sm90", tmp_path) == []


def test_two_records_in_one_second_do_not_overwrite(tmp_path):
    for _ in range(2):
        rec.write_record("test", "sm90", "b" * 64, base="v", scope={}, result="pass", version="1", root=tmp_path, now=T0)
    assert len(rec.read_records("test", "sm90", tmp_path)) == 2


def test_require_needs_the_exact_tree_and_version_and_names_the_action(tmp_path):
    with pytest.raises(rec.Refused, match=r"Run: miniworld-engine verify"):
        rec.require("verify", "sm90", "a" * 64, "2.4.4", tmp_path)
    rec.write_record("verify", "sm90", "a" * 64, base="v", scope={}, result="pass", version="2.4.4", root=tmp_path, now=T0)
    assert rec.require("verify", "sm90", "a" * 64, "2.4.4", tmp_path)["result"] == "pass"
    with pytest.raises(rec.Refused, match="the code changed since"):
        rec.require("verify", "sm90", "c" * 64, "2.4.4", tmp_path)          # another tree: the verified one is stale
    with pytest.raises(rec.Refused):
        rec.require("verify", "sm90", "a" * 64, "2.4.5", tmp_path)          # another version


def test_a_failed_or_partial_record_does_not_satisfy_a_downstream_action(tmp_path):
    for result in ("fail", "partial"):
        rec.write_record("bundle", "sm90", "a" * 64, base="v", scope={}, result=result, version="1", root=tmp_path, now=T0)
    with pytest.raises(rec.Refused):
        rec.require("bundle", "sm90", "a" * 64, "1", tmp_path)
    rec.write_record("bundle", "sm90", "a" * 64, base="v", scope={}, result="nothing", version="1", root=tmp_path, now=T0)   # nothing owed = satisfied
    assert rec.require("bundle", "sm90", "a" * 64, "1", tmp_path)["result"] == "nothing"


def test_all_needs_a_reason_and_a_reason_needs_all():
    assert rec.require_reason(False, "") is None
    assert rec.require_reason(True, "  torch upgrade ") == "torch upgrade"
    with pytest.raises(rec.Refused, match="--all needs --reason"):
        rec.require_reason(True, "")
    with pytest.raises(rec.Refused, match="--all needs --reason"):
        rec.require_reason(True, "   ")
    with pytest.raises(rec.Refused, match="only goes with --all"):
        rec.require_reason(False, "because")


def test_the_ledger_remembers_digests_and_merges(tmp_path):
    assert rec.load_ledger("test", "sm90", tmp_path) == {}
    rec.save_ledger("test", "sm90", {"a": "1"}, tmp_path)
    rec.save_ledger("test", "sm90", {"b": "2", "a": "3"}, tmp_path)
    assert rec.load_ledger("test", "sm90", tmp_path) == {"a": "3", "b": "2"}
    assert rec.load_ledger("test", "sm100", tmp_path) == {}


def test_content_digest_follows_the_bytes_and_a_missing_file(tmp_path):
    (tmp_path / "a.py").write_text("x")
    d1 = rec.content_digest(["a.py", "gone.py"], tmp_path)
    assert d1 == rec.content_digest(["gone.py", "a.py", "a.py"], tmp_path)
    (tmp_path / "a.py").write_text("y")
    assert rec.content_digest(["a.py", "gone.py"], tmp_path) != d1
    assert rec.content_digest(["a.py"], tmp_path, read=lambda p: "fixed") == rec.content_digest(["a.py"], tmp_path / "elsewhere", read=lambda p: "fixed")


def test_the_tree_hash_follows_shipped_files_and_ignores_the_rest(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "pyproject.toml").write_text('version = "1"\n')
    (tmp_path / "src" / "a.py").write_text("a")
    h = rec.tree_hash(tmp_path)
    (tmp_path / "docs" / "n.md").write_text("notes")                         # not shipped
    assert rec.tree_hash(tmp_path) == h
    (tmp_path / "src" / "b.py").write_text("untracked source counts")          # shipped, even untracked
    h2 = rec.tree_hash(tmp_path)
    assert h2 != h
    (tmp_path / "src" / "b.py").write_text("changed")
    assert rec.tree_hash(tmp_path) != h2
    assert rec.declared_version(tmp_path) == "1"


def _git(repo, *args):
    import subprocess

    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def test_a_snapshot_is_the_checkout_as_it_is_and_leaves_the_index_alone(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-qm", "c")
    (repo / "a.py").write_text("x = 2\n")
    (repo / "new.py").write_text("y = 1\n")                                   # untracked, not ignored: part of the snapshot
    tree = rec.snapshot_tree(repo)
    assert rec.tree_exists(tree, repo)
    assert _git(repo, "show", f"{tree}:a.py") == "x = 2\n"                    # the snapshot IS the working tree, edits included
    assert _git(repo, "show", f"{tree}:new.py") == "y = 1\n"
    assert _git(repo, "diff", "--cached", "--name-only") == ""                  # the checkout's index is untouched
    (repo / "a.py").write_text("x = 3\n")
    assert _git(repo, "show", f"{tree}:a.py") == "x = 2\n"                    # a later edit does not move it: the next run diffs from it


def test_the_state_says_where_an_action_left_off(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "tree_exists", lambda tree, repo=None: tree == "T1")
    monkeypatch.setattr(rec, "release_base", lambda repo=None: "v9")
    assert rec.action_base("test", "sm90", root=tmp_path) == ("v9", "the last release (no earlier run of this action)")
    rec.save_state("test", "sm90", "T1", ["tests/cpu/b.py", "tests/cpu/a.py"], tmp_path)
    assert rec.load_state("test", "sm90", tmp_path)["owed"] == ["tests/cpu/a.py", "tests/cpu/b.py"]
    assert rec.action_base("test", "sm90", root=tmp_path) == ("T1", "its last run")
    rec.save_state("test", "sm90", "PRUNED", [], tmp_path)                       # a snapshot git gc removed: back to the release
    assert rec.action_base("test", "sm90", root=tmp_path)[0] == "v9"

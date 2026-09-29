"""Committed-but-unpushed looks exactly like done, and once cost ten hours.

Seventy commits sat on one machine while a second cluster ran a month-old tree; the failure being
diagnosed there had been fixed for days. `git status` says nothing about it -- it is clean the
moment you commit -- so the state that matters is invisible at exactly the moment you stop looking.

`.githooks/post-commit` prints what the remote does not have. It cannot be a gate: refusing a
commit because it is unpushed is nonsense, and refusing a push is the opposite of what is wanted.
So the check here is that the hook exists, is executable, is reachable through `core.hooksPath`,
and actually prints when there is something to print -- verified by running it against a
constructed repository rather than by reading it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
HOOK = REPO / ".githooks" / "post-commit"


def test_the_hook_exists_and_is_executable() -> None:
    assert HOOK.is_file(), f"{HOOK.relative_to(REPO)} is missing"
    import os
    assert os.access(HOOK, os.X_OK), f"{HOOK.relative_to(REPO)} is not executable"


def test_the_hooks_directory_is_the_one_git_uses() -> None:
    """A hook in a directory git does not read is a file, not a hook.

    `core.hooksPath` is per-clone local config -- git deliberately does not let a repository wire
    its own hooks -- so a fresh clone has none and CONTRIBUTING says so ("once per clone"). This
    asserted it unconditionally and therefore could not pass anywhere but on the author's machine;
    the clean-clone job is what caught it. What is left is the failure that is actually a defect:
    hooks wired at some OTHER directory, where the files in `.githooks/` silently never run.
    """
    got = subprocess.run(["git", "config", "core.hooksPath"], cwd=REPO,
                         capture_output=True, text=True, check=False).stdout.strip()
    if not got:
        pytest.skip("hooks not wired in this checkout; CONTRIBUTING: git config core.hooksPath "
                    ".githooks")
    assert got == ".githooks", (
        f"core.hooksPath is {got!r}, so the hooks in .githooks/ never run. "
        f"`git config core.hooksPath .githooks`")


def test_it_reports_when_there_is_something_to_report(tmp_path) -> None:
    """Built rather than mocked: a bare remote, a clone, one commit that is not pushed."""
    bare, work = tmp_path / "remote.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    subprocess.run(["git", "clone", "-q", str(bare), str(work)], check=True)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e", "PATH": "/usr/bin:/bin"}
    def git(*a):
        subprocess.run(["git", *a], cwd=work, check=True, env=env, capture_output=True)
    (work / "a").write_text("1")
    git("add", "a"); git("commit", "-qm", "first"); git("push", "-q", "origin", "HEAD:master")
    git("branch", "--set-upstream-to=origin/master")

    out = subprocess.run(["bash", str(HOOK)], cwd=work, capture_output=True, text=True,
                         check=False).stdout
    assert out.strip() == "", f"hook spoke when everything was pushed:\n{out}"

    (work / "b").write_text("2")
    git("add", "b"); git("commit", "-qm", "second unpushed")
    out = subprocess.run(["bash", str(HOOK)], cwd=work, capture_output=True, text=True,
                         check=False).stdout
    assert "1 commit(s) not on" in out, f"hook stayed silent on an unpushed commit:\n{out}"
    assert "second unpushed" in out, f"hook did not name the commit:\n{out}"

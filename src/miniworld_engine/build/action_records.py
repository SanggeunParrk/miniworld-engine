"""Records of the release actions (``miniworld-engine test | bundle | verify | derive | bench | wheel | release``).

Every action writes one JSON record when it finishes; a downstream action (``wheel``, ``release``) reads the upstream records instead of
recomputing anything, and refuses when one is missing or belongs to another tree. Records live OUTSIDE the checkout, under
``<engine_jit_root>/records/<action>/`` (``MINIWORLD_ENGINE_RECORDS`` overrides the root), so a record can never be committed by accident and
a fresh worktree starts with none.

A record says: which action, which arch, the **tree hash** (the content of everything that ships: ``src/`` and ``pyproject.toml``, tracked
and untracked, ``_prebuilt`` excluded because git ignores it), the base the graph was taken against, the scope the action derived, the
result (``pass`` | ``nothing`` | ``partial`` | ``fail``), the versions, and an ``all`` reason when a deliberate full run was asked for.

Besides records there is one **ledger** per (action, arch): ``item -> input digest`` of what already succeeded. An item whose digest is
unchanged is not run again, so repeating an action on the same edits does nothing.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

#: results a downstream action accepts as "this step is satisfied for that tree"
OK = ("pass", "nothing")
SHIPPED = ("src", "pyproject.toml")


class Refused(Exception):
    """A downstream action's precondition is not met; the message names the action to run."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def records_root(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("MINIWORLD_ENGINE_RECORDS")
    if env:
        return Path(env).expanduser()
    from miniworld_engine import settings

    return settings.engine_jit_root() / "records"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def release_base(repo: Path | None = None) -> str:
    """The last uploaded release: the newest ``v*`` tag reachable from HEAD."""
    try:
        return _git(repo or repo_root(), "describe", "--tags", "--abbrev=0", "--match", "v*").strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise Refused("no v* tag is reachable from HEAD: pass --base <revision> (the last uploaded release)") from exc


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return "absent"
    return h.hexdigest()


def tree_hash(repo: Path | None = None, paths: tuple[str, ...] = SHIPPED) -> str:
    """Content hash of what ships: every tracked or untracked-unignored file under ``paths`` (names and bytes)."""
    repo = repo or repo_root()
    listing = _git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *paths)
    h = hashlib.sha256()
    for rel in sorted({p for p in listing.split("\0") if p}):
        if (repo / rel).is_file():
            h.update(rel.encode() + b"\0" + file_digest(repo / rel).encode() + b"\n")
    return h.hexdigest()


def content_digest(paths, repo: Path | None = None, read=None) -> str:
    """One digest of the bytes of ``paths`` (repo-relative); a missing file is part of the digest as ``absent``."""
    repo = repo or repo_root()
    read = read or (lambda rel: file_digest(repo / rel))
    h = hashlib.sha256()
    for rel in sorted(set(paths)):
        h.update(rel.encode() + b"\0" + read(rel).encode() + b"\n")
    return h.hexdigest()


def _version(dist: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return ""


def versions() -> dict[str, str]:
    return {dist: _version(dist) for dist in ("miniworld-engine", "torch", "triton")}


def declared_version(repo: Path | None = None) -> str:
    m = re.search(r'^version = "([^"]+)"', ((repo or repo_root()) / "pyproject.toml").read_text(), re.MULTILINE)
    if not m:
        raise Refused("no top-level version in pyproject.toml")
    return m.group(1)


# ------------------------------------------------------------------------------------------------------------------ records
def write_record(action: str, arch: str, tree: str, *, base: str, scope: dict, result: str, version: str, extra: dict | None = None,
                 all_reason: str | None = None, root: Path | None = None, now: datetime.datetime | None = None) -> Path:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    rec = {"action": action, "arch": arch, "tree": tree, "base": base, "version": version, "scope": scope, "result": result,
           "all": all_reason, "versions": versions(), "time": now.isoformat(timespec="seconds"), **(extra or {})}
    d = records_root(root) / action
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    while (path := d / f"{now:%Y%m%dT%H%M%S}-{n:02d}-{arch}-{tree[:12]}.json").exists():   # the counter sorts same-second records in order
        n += 1
    path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return path


def read_records(action: str, arch: str = "", root: Path | None = None) -> list[dict]:
    """Every record of ``action`` (``arch``), newest first."""
    out = []
    for p in sorted((records_root(root) / action).glob("*.json"), reverse=True):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not arch or rec.get("arch") == arch:
            rec["_path"] = str(p)
            out.append(rec)
    return out


def latest(action: str, arch: str, *, tree: str | None = None, version: str | None = None, results=OK, root: Path | None = None) -> dict | None:
    for rec in read_records(action, arch, root):
        if tree is not None and rec.get("tree") != tree:
            continue
        if version is not None and rec.get("version") != version:
            continue
        if rec.get("result") in results:
            return rec
    return None


def require(action: str, arch: str, tree: str, version: str, root: Path | None = None) -> dict:
    """The passing record of ``action`` for exactly this tree and version, or :class:`Refused` naming the action to run."""
    rec = latest(action, arch, tree=tree, version=version, root=root)
    if rec is None:
        other = latest(action, arch, version=version, root=root)
        why = (f"the newest passing one is for tree {other['tree'][:12]}, this tree is {tree[:12]} (the code changed since)"
               if other else "there is none")
        raise Refused(f"no passing `{action}` record for tree {tree[:12]} version {version} on {arch}: {why}. Run: miniworld-engine {action}")
    return rec


# ------------------------------------------------------------------------------------------------------------------ state: where an action left off
def snapshot_tree(repo: Path | None = None) -> str:
    """A git tree object of the checkout as it is now (tracked and untracked-unignored files), written to the object store. It is the base the
    action's NEXT run diffs against (``git diff <tree>`` / ``git show <tree>:<path>``): no commit, no ref, and the checkout's own index is not
    touched (a temporary index file)."""
    import tempfile

    repo = repo or repo_root()
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        for args in (("read-tree", "HEAD"), ("add", "-A", ".")):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)
        return subprocess.run(["git", "-C", str(repo), "write-tree"], check=True, capture_output=True, text=True, env=env).stdout.strip()


def tree_exists(tree: str, repo: Path | None = None) -> bool:
    """Whether the object store still holds ``tree`` (an unreferenced snapshot is pruned by ``git gc`` after a while)."""
    return subprocess.run(["git", "-C", str(repo or repo_root()), "cat-file", "-e", f"{tree}^{{tree}}"], capture_output=True).returncode == 0


def _state_path(action: str, arch: str, root: Path | None) -> Path:
    return records_root(root) / "state" / f"{action}-{arch}.json"


def load_state(action: str, arch: str, root: Path | None = None) -> dict:
    """``{"tree": the snapshot the last run started from, "owed": items it did not run, "failing": items that failed and were not rerun
    since}``, or {} before the first run. A failing item runs again when a change reaches it (the diff since ``tree``), not before: the same
    inputs fail the same way."""
    try:
        return json.loads(_state_path(action, arch, root).read_text())
    except (OSError, ValueError):
        return {}


def save_state(action: str, arch: str, tree: str, owed: list[str], root: Path | None = None, failing: list[str] = ()) -> None:
    path = _state_path(action, arch, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"tree": tree, "owed": sorted(set(owed)), "failing": sorted(set(failing)),
                               "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}, indent=1) + "\n")
    tmp.replace(path)


def action_base(action: str, arch: str, repo: Path | None = None, root: Path | None = None) -> tuple[str, str]:
    """``(base, why)``: the snapshot the action's last run started from, when the object store still has it; else the last uploaded release."""
    tree = load_state(action, arch, root).get("tree")
    if tree and tree_exists(tree, repo):
        return tree, "its last run"
    return release_base(repo), "the last release (no earlier run of this action)"


# ------------------------------------------------------------------------------------------------------------------ ledger
def _ledger_path(action: str, arch: str, root: Path | None) -> Path:
    return records_root(root) / "ledger" / f"{action}-{arch}.json"


def load_ledger(action: str, arch: str, root: Path | None = None) -> dict[str, str]:
    try:
        return json.loads(_ledger_path(action, arch, root).read_text())
    except (OSError, ValueError):
        return {}


def save_ledger(action: str, arch: str, done: dict[str, str], root: Path | None = None) -> None:
    """Merge ``done`` (item -> digest of its inputs) into the ledger."""
    cur = load_ledger(action, arch, root)
    cur.update(done)
    path = _ledger_path(action, arch, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=0, sort_keys=True) + "\n")
    tmp.replace(path)


def require_reason(all_: bool, reason: str | None) -> str | None:
    """``--all`` is a deliberate full run and needs ``--reason``; without ``--all`` a reason is meaningless."""
    if all_ and not (reason or "").strip():
        raise Refused('--all needs --reason "<text>": a full run is deliberate, and the reason is printed and recorded')
    if reason and not all_:
        raise Refused("--reason only goes with --all")
    return reason.strip() if all_ and reason else None

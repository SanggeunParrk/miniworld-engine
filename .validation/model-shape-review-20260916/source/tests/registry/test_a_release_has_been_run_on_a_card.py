"""A released version must have been run on at least one GPU, by someone, at that version.

Both CI jobs are `ubuntu-latest`; the step that mentions the GPU proves the tests can be
*collected*. 103 kernels, zero executed by anything automatic. Every correctness claim rests on a
person remembering to run `run_all` and remembering what came back.

A self-hosted runner is not available, so the gate is the artifact rather than the runner.
`run_all` already writes `autotune/manifests/<card>.csv`, it is committed, and its `#provenance`
row records the version, commit, tree state and date. This asserts that the version in
`pyproject.toml` appears in one of them.

Scoped to the RELEASE on purpose. A freshness gate -- "the newest manifest must match HEAD" --
goes red on every ordinary commit until someone finds a card, and a gate that is red by default is
a gate someone switches off. An earlier attempt at this was cut for exactly that. The version only
moves at a release (`test_version_is_released.py` ties it to a changelog entry), which is the one
moment where "nobody has run this" should stop the line.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
MANIFESTS = REPO / "src" / "miniworld_engine" / "autotune" / "manifests"
PROVENANCE = "#provenance"


def declared_version() -> str:
    m = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.MULTILINE)
    assert m, "no top-level version in pyproject.toml"
    return m.group(1)


def provenances() -> dict[str, dict[str, str]]:
    out = {}
    for f in sorted(MANIFESTS.glob("*.csv")):
        with f.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["kernel"] == PROVENANCE:
                    out[f.stem] = {"version": r["backend"], "commit": r["family"],
                                   "tree": r["file"], "date": r["status"]}
                break
    return out


def test_there_are_manifests_to_check() -> None:
    """Without this, deleting the directory would make the gate below vacuous rather than loud."""
    found = sorted(MANIFESTS.glob("*.csv"))
    assert found, f"no manifests in {MANIFESTS.relative_to(REPO)}; the release has no evidence"


def test_the_released_version_has_run_on_a_card() -> None:
    version, have = declared_version(), provenances()
    at_version = {card: p for card, p in have.items() if p["version"] == version}
    assert at_version, (
        f"no manifest records version {version}. Nothing has run this release on a GPU, or the "
        f"run was not committed. On a card:\n"
        f"    python -m miniworld_engine.autotune.run_all\n"
        f"then commit the manifest it writes. Present: "
        f"{ {c: p['version'] or '(no provenance row)' for c, p in have.items()} or 'none'}")


def test_the_evidence_describes_a_commit_and_not_a_working_tree() -> None:
    """A manifest produced from a dirty tree describes whatever was uncommitted at the time. It is
    useful to its author and it is not evidence for anyone else."""
    version = declared_version()
    dirty = [f"{card} (commit {p['commit']}, {p['date']})"
             for card, p in provenances().items()
             if p["version"] == version and p["tree"] != "clean"]
    assert not dirty, (
        "the only evidence for this release was produced from a dirty working tree:\n  "
        + "\n  ".join(dirty)
        + "\n  Re-run on a clean checkout so the manifest names something reproducible.")

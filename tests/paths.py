"""Where the things under test live, and the two files every other test opens by hand.

Nine spellings of the same path had accumulated -- `parents[2]`, a `pyproject.toml` walk, a
`Path(cache.__file__).parents[1]`, and six more -- across 27 files. Three of them were pinned to a
directory depth, so moving a test one level would have broken it silently. One import instead.

These are PATHS and plain readers, not fixtures and not wrappers around the package. That is
deliberate: many of these tests exist to check that a declaration matches the source, and they have
to read the file rather than import the thing it declares, or the import does the checking for them
and the test passes on a broken repository.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
PKG = ROOT / "src" / "miniworld_engine"
REGISTRY = PKG / "kernels" / "registry.csv"
CONFIGS = PKG / "autotune" / "configs"
DATA = PKG / "autotune" / "data"
#: The set `build all` uses. The others pin whole configs rather than declaring ladders.
GRID = CONFIGS / "grid"


def registry_rows() -> list[dict[str, str]]:
    """Every row of registry.csv, in file order."""
    with REGISTRY.open(newline="") as fh:
        return list(csv.DictReader(fh))


def ladders(path: Path) -> dict[str, list[int]]:
    """A grid-spec config set's `axis,values` rows as {axis: [value, ...]}.

    Returns {} for a MATERIALISED set (one row = one config), which declares no ladders.
    """
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows or rows[0][:1] != ["axis"]:
        return {}
    return {r[0]: [int(x) for x in r[1].split()] for r in rows[1:] if len(r) >= 2}


def cache_entries(op: str) -> list[tuple[str, str, list[dict]]]:
    """(card, bucket key, ranked configs) for every tuned entry `op` has.

    Ranked fastest-first, five deep. An entry stores `num_warps` and `num_stages` at the top level
    and the tile axes inside `kwargs` -- a split that has caught out more than one reader.
    """
    out = []
    d = DATA / op
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for key, ranked in (data.get("entries") or {}).items():
            if isinstance(ranked, list) and ranked:
                out.append((f.stem, key, ranked))
    return out


def tracked_subdirectories(root: Path) -> set[str]:
    """Subdirectories of `root` holding source files tracked or not ignored by git.

    `benchmarks/` results are gitignored, so a working checkout accumulates output under target
    names that no longer exist and `iterdir()` sees them. That happened: fourteen directories under
    the retired short vocabulary, plus 257 MB of another branch's `mpnn` output, turned this test
    red locally while CI -- a fresh clone with only tracked files -- stayed green. A check that
    depends on what a previous experiment left on disk is not checking the repository.

    Raises rather than falling back to `iterdir()`: a silent fallback restores the behaviour this
    replaces, and an empty result would make the caller pass vacuously.
    """
    rel = root.relative_to(ROOT)
    proc = subprocess.run(["git", "ls-files", "--cached", "--others",
                           "--exclude-standard", "-z", "--", str(rel)],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed for {rel}: {proc.stderr.strip()}")
    names = {Path(e).relative_to(rel).parts[0]
             for e in proc.stdout.split("\0") if e and len(Path(e).relative_to(rel).parts) > 1}
    assert names, f"no tracked files under {rel}; this check would pass vacuously"
    return names

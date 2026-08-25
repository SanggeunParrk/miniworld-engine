"""Code that runs may not name a path that exists on one machine.

Three CUDA transition extensions carried `-I/home/<user>/mathdx_dl/...`, six times across three
build functions -- unbuildable by anyone else by construction, and by the time it was found
unbuildable here too: the directory was gone and nothing noticed. A build path rots invisibly, so
it gets a guard rather than a convention.

Scope is `src/` and `tools/`, the two trees whose contents are executed. `benchmarks/` and
`profiles/` are excluded on purpose -- a recorded measurement naming the file it came from is
provenance, and stripping it would destroy the thing those files exist to carry.

Absolute paths as such are fine: `/usr/local/cuda`, `/proc`, `/tmp` are portable facts about a
Linux box. What is banned is a path into one account or one site's storage.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Trees whose files are executed. Assembled from parts so this file does not match its own rule.
SCANNED = ("src", "tools")
SUFFIXES = (".py", ".cu", ".cuh", ".csv", ".json", ".toml", ".cfg", ".sh", ".sbatch")
_ACCOUNT = ("home", "users", "Users")
_SITE = ("public_data", "public_data02", "scratch", "lustre", "gpfs")
BANNED = re.compile("|".join(rf"/{root}/[A-Za-z0-9_.-]+" for root in (*_ACCOUNT, *_SITE)))


def scanned_files() -> list[Path]:
    # kernels/<family>/notes/ is the family's optimization log: records of what was run on one
    # machine at one time. A path in there is part of the record, not a build input.
    return sorted(p for root in SCANNED for p in (REPO / root).rglob("*")
                  if p.is_file() and p.suffix in SUFFIXES and "notes" not in p.parts)


def test_there_are_files_to_check() -> None:
    """A change to SCANNED or SUFFIXES must not turn the test below into a green no-op."""
    found = scanned_files()
    assert len(found) > 100, f"only {len(found)} files found under {SCANNED}; the scan is broken"


def test_no_executed_file_names_a_personal_or_site_path() -> None:
    offenders: dict[str, list[str]] = {}
    for path in scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            hit = BANNED.search(line)
            if hit:
                offenders.setdefault(str(path.relative_to(REPO)), []).append(f"{lineno}: {hit[0]}")
    assert not offenders, (
        "a path that exists on one machine, in code that runs:\n"
        + "\n".join(f"  {f}\n" + "\n".join(f"      {h}" for h in hits)
                    for f, hits in offenders.items())
        + "\n  Resolve it at run time (see `_nvcc.mathdx_includes`) and name the variable to set.")

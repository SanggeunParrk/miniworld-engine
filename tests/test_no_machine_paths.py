"""Shipped source may not name a filesystem location that exists on one machine.

The failure this prevents is not hypothetical and not subtle. Three CUDA transition extensions
carried `-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/...` -- six occurrences across three
build functions. No other user could ever have built them. By the time it was found the path
had also stopped existing on the author's own machine, and nothing noticed, because those
extensions are absent from `registry.csv`, their build is lazy, and no test or CI job asks for
them. A build path is exactly the kind of thing that rots invisibly, so it gets a guard rather
than a convention.

What is banned is an absolute path *into somebody's account or a site-specific data mount*.
Absolute paths as such are fine -- `/usr/local/cuda`, `/proc`, `/tmp` are portable facts about a
Linux box, and `Path(__file__)`-relative resolution is the correct way to find package data.
Comments are scanned too: a real path in a comment is a path someone will copy.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

#: Roots that belong to one account or one site. `/home/<user>` and the cluster's data mounts.
BANNED = re.compile(
    r"/(?:home|users|Users)/[A-Za-z0-9_.-]+"          # a specific account
    r"|/public_data\d*/[A-Za-z0-9_.-]+"               # this cluster's per-user data volumes
    r"|/scratch/[A-Za-z0-9_.-]+"
    r"|/lustre/[A-Za-z0-9_.-]+"
    r"|/gpfs/[A-Za-z0-9_.-]+")

#: Source kinds that end up in the wheel and can carry a build or data path.
SUFFIXES = (".py", ".cu", ".cuh", ".csv", ".json", ".toml", ".cfg")


def shipped_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*") if p.is_file() and p.suffix in SUFFIXES)


def test_there_are_files_to_check() -> None:
    """Without this, a change to SUFFIXES or SRC turns the test below into a green no-op --
    the failure mode `tests/test_lazy_import_targets.py` exists for."""
    files = shipped_files()
    assert len(files) > 100, f"only {len(files)} shipped files found under {SRC}; the scan is broken"


def test_no_shipped_file_names_a_personal_or_site_path() -> None:
    offenders: dict[str, list[str]] = {}
    for path in shipped_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            match = BANNED.search(line)
            if match:
                rel = str(path.relative_to(REPO))
                offenders.setdefault(rel, []).append(f"{lineno}: {match.group(0)}")
    assert not offenders, (
        "shipped source names a path that exists on one machine:\n"
        + "\n".join(f"  {f}\n" + "\n".join(f"      {h}" for h in hits)
                    for f, hits in offenders.items())
        + "\n  Resolve it at run time instead (see `_nvcc.mathdx_includes`), and make the "
          "failure name the variable to set.")

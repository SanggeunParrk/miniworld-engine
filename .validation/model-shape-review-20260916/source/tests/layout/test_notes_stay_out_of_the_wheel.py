"""A kernel's optimization log lives beside it, and ships with nothing.

`kernels/<family>/notes/` is where a family's per-round write-ups and the scratch measured with
them now live -- beside the kernel they describe, because that is the only thing they are about.
Putting them under `src/` has one hazard: `[tool.setuptools.package-data]` sweeps `**/*.csv`,
`**/*.json` and the CUDA sources out of the package tree recursively, and those globs do not care
that `notes/` is not a package. It shipped 20 files into the wheel before this was noticed.

A consumer installs a kernel, not its history. This asserts the exclusion holds, against the
package-data declaration rather than against a built wheel -- building one here would put a
minute of setuptools into every test run.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
KERNELS = REPO / "src" / "miniworld_engine" / "kernels"


def test_there_are_notes_to_exclude() -> None:
    """If every notes/ tree disappears, the exclusion below is guarding nothing and the reasoning
    in this file should be revisited rather than left as decoration."""
    trees = sorted(d for d in KERNELS.glob("*/notes") if d.is_dir())
    assert trees, "no kernels/*/notes/ tree exists; is the exclusion still needed?"


def test_notes_are_excluded_from_package_data() -> None:
    text = (REPO / "pyproject.toml").read_text()
    section = re.search(r"^\[tool\.setuptools\.exclude-package-data\]\n(?P<body>(?:.*\n)*?)(?=^\[|\Z)",
                        text, re.MULTILINE)
    assert section, "no [tool.setuptools.exclude-package-data]; the notes would ship"
    body = section.group("body")
    # Depth-explicit, not `**`: setuptools' glob does not match across directories here, and the
    # `**` form silently excluded nothing -- the wheel still carried all 20 files.
    patterns = re.findall(r'"kernels/\*/notes/(\*(?:/\*)*)"', body)
    assert patterns, f"nothing excludes kernels/*/notes in:\n{body}"
    # Each pattern covers ONE depth: setuptools' glob does not match across directories, so `**`
    # excluded nothing at all and the wheel still carried every file. The deepest pattern is
    # therefore not enough either -- a gap at depth 2 ships depth 2, whatever depth 3 says.
    covered = {p.count("*") for p in patterns}
    actual = max((len(f.relative_to(d).parts) for d in KERNELS.glob("*/notes")
                  for f in d.rglob("*") if f.is_file()), default=0)
    missing = sorted(set(range(1, actual + 1)) - covered)
    assert not missing, (
        f"notes reach {actual} levels deep and the exclusion covers {sorted(covered)}; files at "
        f"depth {missing} would ship. Add: "
        + ", ".join(f'"kernels/*/notes/{"/".join("*" * d)}"' for d in missing))


def test_notes_are_not_importable() -> None:
    """No __init__.py anywhere under notes/ -- otherwise packages.find would pick it up and the
    data exclusion would not be the only thing standing between a lab notebook and the wheel."""
    packages = [str(p.relative_to(REPO)) for d in KERNELS.glob("*/notes")
                for p in d.rglob("__init__.py")]
    assert not packages, f"notes/ contains a package: {packages}"

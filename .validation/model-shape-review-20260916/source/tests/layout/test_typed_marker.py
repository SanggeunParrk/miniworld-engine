"""The package ships its types, or the type gate protects only this repo.

`ty check src tests benchmarks tools` runs at zero, and every public signature is annotated. None
of that reaches a consumer without a PEP 561 marker: absent `py.typed`, a type checker treats every
symbol imported from `miniworld_engine` as `Any`, silently.

Two halves, and either alone ships nothing: the file has to exist, and `package-data` has to carry
it into the wheel. `[tool.setuptools.package-data]` here is explicit rather than
`include-package-data`, so a new asset type is opt-in -- which is why the second half needs a test.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
PKG = REPO / "src" / "miniworld_engine"


def test_the_marker_file_exists() -> None:
    marker = PKG / "py.typed"
    assert marker.is_file(), f"missing {marker.relative_to(REPO)} (PEP 561)"
    # PEP 561 allows content, but an empty marker is the convention and says "all of this package".
    assert marker.read_text() == "", "py.typed should be empty; a partial-stub spec is not intended"


def test_the_marker_is_shipped() -> None:
    """A wheel built without this entry contains only `.py` and the assets listed, so the marker
    would exist in the source tree and not in the artifact -- the failure this test exists for."""
    # Read as text, not with `tomllib`: this package declares `requires-python = ">=3.10"` and
    # tomllib is 3.11+, so a tomllib import here would make the test suite need a newer Python
    # than the library does.
    text = (REPO / "pyproject.toml").read_text()
    match = re.search(r"^miniworld_engine = \[(?P<patterns>[^\]]*)\]", text, re.MULTILINE)
    assert match, "no `miniworld_engine = [...]` entry under [tool.setuptools.package-data]"
    patterns = [p.strip().strip('"') for p in match.group("patterns").split(",")]
    assert "py.typed" in patterns, (
        f"py.typed is not in [tool.setuptools.package-data].miniworld_engine ({patterns}); "
        f"the marker exists in src/ but no wheel would carry it")

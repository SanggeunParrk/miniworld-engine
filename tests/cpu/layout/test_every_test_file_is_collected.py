"""A test file pytest never opens is worse than a deleted one: it still looks like coverage.

Grouping the suite into subdirectories put seven files under `tests/build/`, and the whole
directory vanished -- 65 tests, no error, no warning, a green run. `build` is in pytest's default
`norecursedirs`, alongside `dist`, `node_modules` and `venv`. The only signal was the pass count
dropping from 1201 to 1136, which nobody is obliged to notice.

So the rule is checked rather than remembered.
"""
from __future__ import annotations

from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
TESTS = REPO / "tests"

#: pytest's built-in default, copied here because it is not exposed as a value. The test below
#: fails if the copy stops matching pytest's own source, so it cannot drift silently.
IGNORED = {"*.egg", ".*", "_darcs", "build", "CVS", "dist", "node_modules", "venv", "{arch}"}


def test_the_copied_ignore_list_still_matches_pytest() -> None:
    import re

    import _pytest.main as main_mod
    src = Path(main_mod.__file__).read_text()
    match = re.search(r'"norecursedirs",\s*\n?\s*.*?default=\[([^\]]*)\]', src, re.DOTALL)
    assert match, "could not find pytest's norecursedirs default; re-derive IGNORED by hand"
    upstream = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert upstream == IGNORED, (
        f"pytest's defaults changed: it now skips {sorted(upstream)}, this file assumes "
        f"{sorted(IGNORED)}. Update IGNORED, then re-check the tree against it.")


def test_no_test_file_hides_in_a_directory_pytest_skips() -> None:
    hidden = [
        f"{f.relative_to(REPO)}  (directory {part!r})"
        for f in sorted(TESTS.rglob("test_*.py"))
        for part in f.relative_to(TESTS).parts[:-1]
        if part in IGNORED or part.startswith(".")
    ]
    assert not hidden, (
        "pytest will not collect these -- they sit inside a directory in its default "
        "norecursedirs, so they pass by never running:\n  " + "\n  ".join(hidden))


def test_there_are_test_files_to_check() -> None:
    found = list(TESTS.rglob("test_*.py"))
    assert len(found) > 40, f"only {len(found)} test files found under {TESTS}; the walk is broken"

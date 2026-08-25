"""A version number that never moves cannot distinguish two packages.

`version = "0.1.0"` stood across 191 commits *including* a rename of the distribution from
`miniworld-kernels` to `miniworld-engine`. The one consumer's pinned tree declares
`name = "miniworld-kernels", version = "0.1.0"`; main declared `name = "miniworld-engine",
version = "0.1.0"`. Two packages that cannot both be imported, identical in every declared
field a resolver looks at. That is the failure these tests exist to make impossible to repeat.

The rule is not "bump the version often". It is that the version and the changelog cannot
disagree: whatever `pyproject.toml` claims to be must have a section in `CHANGELOG.md`
describing it. A bump with no entry fails here, and so does a changelog that has accumulated
everything under `[Unreleased]` while the version stands still.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Read as text, not with `tomllib`: `requires-python = ">=3.10"` and tomllib is 3.11+, so
# importing it here would make the suite need a newer Python than the library does.
_VERSION = re.compile(r'^version = "(?P<v>[^"]+)"', re.MULTILINE)
_HEADING = re.compile(r"^## \[(?P<v>[^\]]+)\]", re.MULTILINE)


def declared_version() -> str:
    match = _VERSION.search((REPO / "pyproject.toml").read_text())
    assert match, "no top-level `version = ...` in pyproject.toml"
    return match.group("v")


def changelog_sections() -> list[str]:
    return _HEADING.findall((REPO / "CHANGELOG.md").read_text())


def test_the_declared_version_has_a_changelog_section() -> None:
    version, sections = declared_version(), changelog_sections()
    assert version in sections, (
        f"pyproject declares version {version!r} but CHANGELOG.md has no `## [{version}]` "
        f"section (it has {sections}). Add the section, or the release is a number with no "
        f"description of what is in it.")


def test_unreleased_is_not_the_only_section() -> None:
    """The state this repo was in: a well-kept changelog for a package that never released."""
    sections = changelog_sections()
    released = [s for s in sections if s.lower() != "unreleased"]
    assert released, (
        "CHANGELOG.md has only an [Unreleased] section. Every change ever made is filed as "
        "unreleased, so no consumer can learn what is in the version they have.")


def test_the_current_version_is_not_unreleased() -> None:
    """`[Unreleased]` must sit ABOVE the declared version, not be it."""
    sections = changelog_sections()
    assert sections, "CHANGELOG.md has no `## [...]` sections at all"
    version = declared_version()
    if sections[0].lower() == "unreleased":
        assert len(sections) > 1, "CHANGELOG.md has [Unreleased] and nothing released below it"
        assert sections[1] == version, (
            f"the newest released section is {sections[1]!r} but pyproject says {version!r}; "
            f"the version and the changelog disagree about what was last released.")
    else:
        assert sections[0] == version, (
            f"the newest changelog section is {sections[0]!r} but pyproject says {version!r}")


def test_a_major_bump_records_what_broke() -> None:
    """1.0.0 exists because the import name changed. A major version with no `### Breaking`
    section is a break the consumer has to discover by running it."""
    version = declared_version()
    major, _, rest = version.partition(".")
    if not (major.isdigit() and int(major) > 0 and rest.startswith("0.0")):
        return  # only x.0.0 carries this obligation
    text = (REPO / "CHANGELOG.md").read_text()
    start = text.index(f"## [{version}]")
    end = text.find("\n## [", start + 1)
    body = text[start : end if end != -1 else len(text)]
    assert re.search(r"^### Breaking", body, re.MULTILINE), (
        f"`## [{version}]` is a major release with no `### Breaking` section. If nothing broke, "
        f"it should not be a major bump; if something did, it has to be written down.")

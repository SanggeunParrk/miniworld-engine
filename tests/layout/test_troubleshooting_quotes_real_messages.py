"""A troubleshooting page that quotes a message the code no longer prints is worse than none.

The reader greps their log for the string, finds nothing, and concludes the page is not about
their problem. So every section headed by a message we own names the literal that must still be
in `src/`, and a section whose message comes from argparse, torch or triton says whose it is.

The anchors are written here rather than derived from the heading: the messages are f-strings, so
the heading shows an interpolated example (`unknown config set 'configs/grid'`) while the source
holds the template (`unknown config set {config_type!r}`). Guessing the overlap is how this test
first failed on four correct sections.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
PAGE = REPO / "docs" / "troubleshooting.md"

#: heading -> the literal that must still exist in src/, or None when the message is not ours.
ANCHORS: dict[str, str | None] = {
    "dynamic_func() missing 2 required positional arguments: 'BLOCK_M1' and 'BLOCK_K'": None,
    "unknown config set 'configs/grid'": "unknown config set ",
    "cuBLASDx headers not found": "cuBLASDx headers not found",
    "no tuned autotune cache entry for this shape": "no tuned autotune cache entry for this shape",
    "miniworld-engine: error: unrecognized arguments: --bench-budget": None,
    "import miniworld_kernels` fails": None,
    # Headings that describe a SYMPTOM rather than quote a message. They still belong on the page
    # -- "the count is wrong" and "a kernel was skipped" are what the reader arrives with -- but
    # there is no literal to keep alive, so they anchor on the source of the behaviour instead.
    "build all` reports far fewer units than expected": None,
    "A kernel is `skipped (this card is smXX)": "skipped (this card is ",
}


def headings() -> list[str]:
    return [ln[3:].strip().strip("`") for ln in PAGE.read_text().splitlines()
            if ln.startswith("## ") and "`" in ln]


def test_every_message_heading_is_accounted_for() -> None:
    """A new section must declare its anchor or declare the message foreign. Without this the
    check below silently stops covering whatever was added."""
    undeclared = [h for h in headings() if h not in ANCHORS]
    assert not undeclared, (
        f"message-shaped sections with no entry in ANCHORS: {undeclared}. Add the literal that "
        f"must exist in src/, or None if the message belongs to argparse/torch/triton.")


def test_every_message_we_own_is_still_emitted() -> None:
    missing = []
    for heading, anchor in ANCHORS.items():
        if anchor is None or heading not in headings():
            continue
        found = subprocess.run(["git", "grep", "-qF", "--", anchor, "--", "src"],
                               cwd=REPO, capture_output=True, check=False)
        if found.returncode != 0:
            missing.append(f"{heading!r}: src/ no longer contains {anchor!r}")
    assert not missing, (
        "docs/troubleshooting.md quotes a message src/ no longer emits. Either it was reworded -- "
        "update the page and the anchor -- or the failure is gone and the section should go with "
        "it:\n  " + "\n  ".join(missing))


def test_the_anchor_list_does_not_outlive_the_page() -> None:
    """The other direction: an anchor for a section nobody kept."""
    stale = [h for h in ANCHORS if h not in headings()]
    assert not stale, f"ANCHORS names sections {PAGE.name} no longer has: {stale}"

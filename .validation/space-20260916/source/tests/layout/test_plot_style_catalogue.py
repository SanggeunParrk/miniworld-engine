"""Every implementation label a figure can show must have a pinned identity.

``viz/style.py`` maps each backend label to an identity, a display name and a colour, and its own
comment says why: "so sibling variants that share a substring the canonical heuristics collapse
(`cute`/`miniworld`) never overwrite each other in a single figure". The mapping is keyed by the
label with separators stripped, so a label rename on the bench side silently orphans its entry —
and then `canonical()` falls through to the substring heuristics.

That is not hypothetical. Renaming the implementation labels (`triton_tri_attn` ->
`triton_triangle_attention` and four siblings) orphaned five entries, and one of them collided:
`triton_triangle_attention_miniworld` stopped matching its alias, matched the "miniworld"
heuristic instead, and became the same identity as the real `miniworld` series. Two lines, one
colour, one legend entry, in the figures comparing exactly those two.

The labels come from two places, and both matter:
  * what ``bench.py`` can produce, read out of its `implementation == "..."` chains — a future
    figure's series;
  * what the committed tables under ``benchmarks/**/results/`` already contain — an existing
    figure re-rendered from its own data must not change colour.
"""
from __future__ import annotations

import ast
import collections
import csv
from pathlib import Path

import pytest

from miniworld_engine.viz import style

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
BENCH = REPO / "benchmarks" / "runners" / "bench.py"


def _labels_bench_can_produce() -> set[str]:
    """Read from the source, like `bench.py::target_impls` does, so it cannot drift."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(BENCH.read_text())):
        if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id == "implementation"):
            continue
        for cmp in node.comparators:
            values = [cmp] if isinstance(cmp, ast.Constant) else list(getattr(cmp, "elts", []))
            out |= {v.value for v in values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)}
    return out


def _labels_in_committed_tables() -> set[str]:
    out: set[str] = set()
    for table in REPO.glob("benchmarks/*/*/results/*/tables/*.csv"):
        with table.open(newline="") as fh:
            for row in csv.DictReader(fh):
                out |= {row[c] for c in ("implementation", "implementation_type") if row.get(c)}
    return out


PRODUCIBLE = _labels_bench_can_produce()
COMMITTED = _labels_in_committed_tables()
ALL_LABELS = PRODUCIBLE | COMMITTED


def test_there_are_labels_to_check() -> None:
    """Both readers must find something, or the checks below are no-ops."""
    assert len(PRODUCIBLE) > 20, PRODUCIBLE
    assert len(COMMITTED) > 20, sorted(COMMITTED)


@pytest.mark.parametrize("label", sorted(ALL_LABELS))
def test_every_label_has_a_pinned_identity(label: str) -> None:
    """Uncatalogued means the colour and legend text come from a substring guess."""
    assert style._norm(label) in style._ALIASES, (
        f"{label!r} has no entry in viz/style.py, so `canonical()` guesses: it resolves to "
        f"{style.canonical(label)!r} and is drawn as {style.label_for(label)!r}. Add it to "
        f"_KERNEL_VARIANTS (or _ALIASES) with a colour from the scheme stated there.")


def test_no_two_labels_share_one_identity() -> None:
    """The collision the table exists to prevent, asserted directly."""
    by_identity = collections.defaultdict(list)
    for label in ALL_LABELS:
        by_identity[style.canonical(label)].append(label)
    clashes = {ident: sorted(v) for ident, v in by_identity.items() if len(v) > 1}
    assert not clashes, (
        f"these labels collapse to one identity, so they share a colour and a legend entry: "
        f"{clashes}")


def test_no_style_entry_is_unreachable() -> None:
    """A key no label can reach is a rename applied on one side only — which is exactly how the
    five attention entries were orphaned. Historical labels count as reachable: a committed table
    still has to re-render with the colour it was published with."""
    keys = {style._norm(label) for label in ALL_LABELS}
    orphaned = sorted(k for k in style._KERNEL_VARIANTS if k not in keys)
    assert not orphaned, (
        f"no implementation label reaches these style entries: {orphaned}. Either a label was "
        f"renamed without re-keying its entry, or the entry is dead and should go.")

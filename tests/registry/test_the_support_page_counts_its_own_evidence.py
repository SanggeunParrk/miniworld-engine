"""`docs/supported.md` is a support claim, and every number in it must come from a manifest.

The page opens by saying a claim that outruns its measurements is how a consumer books time on
hardware the library has never touched. It then said `driven 94, ok 94, failed 0, skipped 9` for an
A6000 whose manifest held 83 ok and 6 skipped, over a registry of 91 kernels rather than the 103 the
numbers were counted from. Nothing tied the two together, so the page aged and the file moved.

Counted per PRECISION, because that is what a run is: a kernel is launched at the precisions its
row declares, and 89 of 91 kernels declare bf16 while 42 declare fp32.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
PAGE = ROOT / "docs" / "supported.md"
MANIFESTS = ROOT / "src" / "miniworld_engine" / "autotune" / "manifests"

#: `card (arch)` in the page's first column -> the manifest file that is its evidence. Declared,
#: so a new card added to the page without a manifest is a failure here rather than a claim.
EVIDENCE = {
    "RTX A6000 (sm86)": "NVIDIA RTX A6000 (sm86).csv",
    "RTX A5000 (sm86)": "NVIDIA RTX A5000 (sm86).csv",
}


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["kernel"] != "#provenance"]


def _claims() -> list[tuple[str, str, dict[str, int]]]:
    """(card, precision, {status: count}) for every table row that states counts."""
    out = []
    for line in PAGE.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0] not in EVIDENCE:
            continue
        counts = {k: int(v) for k, v in re.findall(r"(\w+) (\d+)", " ".join(cells))}
        if counts:
            out.append((cells[0], cells[1], counts))
    return out


def test_the_page_still_states_some_counts() -> None:
    """Guard the guard: a reworded table would make every check below vacuous."""
    claims = _claims()
    assert claims, f"{PAGE.name} states no `ok N` counts at all; this file would pass on anything"
    assert {c for c, _, _ in claims} == set(EVIDENCE), (
        f"the page names cards {sorted({c for c, _, _ in claims})}, EVIDENCE knows "
        f"{sorted(EVIDENCE)}")


@pytest.mark.parametrize(("card", "precision", "counts"), _claims(),
                         ids=[f"{c}-{p}" for c, p, _ in _claims()])
def test_each_stated_count_matches_the_manifest(card: str, precision: str,
                                                counts: dict[str, int]) -> None:
    rows = [r for r in _rows(MANIFESTS / EVIDENCE[card])
            if ((r.get("dtype") or "").strip() or "bf16") == precision]
    assert rows, f"{EVIDENCE[card]} has no {precision} rows, but the page states {counts}"
    have = {"ok": 0, "failed": 0, "skipped": 0, "untested": 0}
    for r in rows:
        have[r["status"]] = have.get(r["status"], 0) + 1
    # `driven` is the run's own word for "launched and judged" -- ok plus failed.
    have["driven"] = have["ok"] + have["failed"]
    for name, claimed in counts.items():
        if name not in have:
            continue
        assert have[name] == claimed, (
            f"{PAGE.name} claims {name} {claimed} for {card} at {precision}; "
            f"{EVIDENCE[card]} holds {name} {have[name]}")


def test_the_arch_counts_match_the_registry() -> None:
    """The "GPU that has NOT been run" table counts declared archs. It said sm100 7 when the
    registry declares 4 -- counted before nine rows left the registry, and never recounted."""
    reg = list(csv.DictReader((ROOT / "src/miniworld_engine/kernels/registry.csv").open(newline="")))
    want = {}
    for r in reg:
        want[(r.get("arch") or "").strip() or "sm80"] = want.get(
            (r.get("arch") or "").strip() or "sm80", 0) + 1
    text = PAGE.read_text()
    for arch, n in sorted(want.items()):
        row = re.search(rf"^\| {re.escape(arch)} \| (\d+) \|", text, re.MULTILINE)
        assert row, f"{PAGE.name} has no row for declared arch {arch} ({n} kernels)"
        assert int(row.group(1)) == n, (
            f"{PAGE.name} says {arch} has {row.group(1)} kernels; registry.csv declares {n}")

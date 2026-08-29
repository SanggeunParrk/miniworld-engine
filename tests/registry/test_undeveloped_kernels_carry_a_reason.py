"""A kernel held out of `build all` has to say why, in writing.

`developed=no` in registry.csv removes a kernel from the sweep, which is a decision about where
tuning time goes -- hours of GPU, and a cache that will not have an entry for it. The judgement is
HAND-MAINTAINED on purpose: bias_only_attention loses to torch on time on every committed card and
uses half the memory at every length, so no rule reading one of those numbers alone gets it right.

What a test can hold is that the judgement was written down and that the two files agree.
"""
from __future__ import annotations

import csv
from pathlib import Path

REG = Path(__file__).resolve().parents[2] / "src/miniworld_engine/kernels/registry.csv"
UNDEV = REG.parent / "undeveloped.csv"


def _rows(p: Path) -> list[dict]:
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_undeveloped_kernel_names_a_reason() -> None:
    marked = {r["kernel"] for r in _rows(REG)
              if r["backend"] == "triton" and (r.get("developed") or "").strip() == "no"}
    reasons = {r["kernel"]: (r.get("reason") or "").strip() for r in _rows(UNDEV)}
    missing = sorted(marked - set(reasons))
    assert not missing, (
        "held out of `build all` with no reason recorded in kernels/undeveloped.csv: "
        + ", ".join(missing))
    thin = sorted(k for k in marked if len(reasons[k]) < 40)
    assert not thin, f"the reason is too short to be one: {thin}"


def test_no_reason_is_recorded_for_a_kernel_that_is_still_built() -> None:
    """The other direction, so the file cannot rot into a list of things that were undone."""
    built = {r["kernel"] for r in _rows(REG)
             if r["backend"] == "triton" and (r.get("developed") or "yes").strip() != "no"}
    stale = sorted(built & {r["kernel"] for r in _rows(UNDEV)})
    assert not stale, ("kernels/undeveloped.csv explains kernels that ARE built; "
                       f"delete the row or set developed=no: {stale}")


def test_the_column_says_yes_or_no_and_nothing_else() -> None:
    bad = {(r["kernel"], r.get("developed")) for r in _rows(REG)
           if r["backend"] == "triton" and (r.get("developed") or "").strip() not in ("yes", "no")}
    assert not bad, f"developed must be yes or no: {sorted(bad)}"

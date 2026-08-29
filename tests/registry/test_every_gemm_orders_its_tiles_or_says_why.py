"""Every GEMM kernel either tunes its tile visit order or records why it cannot.

`kernels/_tiles.py` explains what the order buys and when. The list of kernels that DON'T take it
is the part that rots: a kernel added later, or one whose grid changes shape, silently keeps a
fixed order and nobody notices because nothing fails. So the exemptions are declared, with a
reason, and this test holds the two lists complementary.

Two things exempt a kernel today, and both are structural rather than a judgement call:

  * a 1-D grid over M alone, with the column extent looped inside the program -- one program owns
    a whole row block across every column, so there is no second tile axis to order at all
  * attention, where the grid's other axes are batch, head and row, not output columns; the
    key/value axis is looped in-kernel, and axis 0 varying fastest already keeps one head's K/V hot
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "src/miniworld_engine/kernels/registry.csv"
EXEMPT = ROOT / "src/miniworld_engine/kernels/tile_order_exempt.csv"


def _gemms() -> list[dict]:
    with REG.open(newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if r["backend"] == "triton" and r["kind"] == "gemm"]


def _uses_tile_order(row: dict) -> bool | None:
    path = ROOT / "src" / row["file"]
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == row["symbol"]), None)
    if fn is None:
        return None
    return "tile_order(" in (ast.get_source_segment(path.read_text(), fn) or "")


def _exempt() -> dict[str, str]:
    with EXEMPT.open(newline="") as fh:
        return {r["kernel"]: r["reason"].strip() for r in csv.DictReader(fh)}


def test_every_gemm_is_on_the_axis_or_exempt_with_a_reason() -> None:
    exempt = _exempt()
    missing = []
    for r in _gemms():
        uses = _uses_tile_order(r)
        if uses is None or uses:
            continue
        if r["kernel"] not in exempt:
            missing.append(r["kernel"])
    assert not missing, (
        "GEMM kernels with a fixed tile visit order and no recorded reason -- either call "
        "tile_order (see kernels/_tiles.py) or add a row to kernels/tile_order_exempt.csv:\n  "
        + "\n  ".join(sorted(missing)))


def test_no_exemption_outlives_its_kernel() -> None:
    """An exemption for a kernel that now tunes its order, or no longer exists, is stale."""
    gemms = {r["kernel"]: r for r in _gemms()}
    stale = []
    for kernel in _exempt():
        if kernel not in gemms:
            stale.append(f"{kernel}: not a triton gemm in registry.csv")
        elif _uses_tile_order(gemms[kernel]):
            stale.append(f"{kernel}: calls tile_order, so the exemption is spent")
    assert not stale, "stale rows in kernels/tile_order_exempt.csv:\n  " + "\n  ".join(stale)


def test_every_reason_says_something() -> None:
    thin = sorted(k for k, why in _exempt().items() if len(why) < 40)
    assert not thin, f"exemptions whose reason is too short to be one: {thin}"


def test_the_ladder_reaches_both_ends() -> None:
    """Whatever a kernel did BEFORE the axis existed has to stay reachable.

    The two ends are the two things a kernel could already have been doing, and the repo had both:
    a 2-D grid pins column-first, which is any GROUP_M >= n_m; a hand-written
    `pid_m = pid // num_pid_n` pins row-first, which is GROUP_M = 1. Seven of the twenty converted
    kernels were the second kind, and the first ladder written for them held 4, 16 and 65536 -- so
    the claim that "the tuner can always choose today's behaviour" was false for those seven, in
    the direction that matters: if 1 was already their best, tuning could only make them worse.
    """

    cfg = ROOT / "src/miniworld_engine/autotune/configs"
    live = {r["kernel"] for r in _gemms()}
    bad = []
    for f in sorted(cfg.rglob("*.csv")):
        if f.stem not in live:
            continue          # a config file for a kernel the registry no longer declares
        with f.open(newline="") as fh:
            if not fh.readline().startswith("axis,"):
                continue
        for line in f.read_text().splitlines():
            if not line.startswith("GROUP_M,"):
                continue
            values = {int(v) for v in line.split(",", 1)[1].split()}
            if 1 not in values:
                bad.append(f"{f.parent.name}/{f.name}: no 1 (row-first)")
            if not any(v >= 4096 for v in values):
                bad.append(f"{f.parent.name}/{f.name}: nothing >= 4096 (column-first)")
    assert not bad, ("GROUP_M ladders that cannot reproduce a kernel's pre-axis behaviour:\n  "
                     + "\n  ".join(bad))


def test_the_materialised_sets_pin_one_value() -> None:
    """A set that lists whole configs is not searching this axis; it must state which end it is at."""
    import csv as _csv

    cfg = ROOT / "src/miniworld_engine/autotune/configs"
    live = {r["kernel"] for r in _gemms()}
    bad = []
    for f in sorted(cfg.rglob("*.csv")):
        if f.stem not in live:
            continue
        with f.open(newline="") as fh:
            header = fh.readline().strip()
        if header.startswith("axis,") or "GROUP_M" not in header:
            continue
        i = header.split(",").index("GROUP_M")
        with f.open(newline="") as fh:
            rows = list(_csv.reader(fh))[1:]
        seen = {r[i] for r in rows if r}
        if seen - {"1", "65536"}:
            bad.append(f"{f.parent.name}/{f.name}: {sorted(seen)}")
    assert not bad, ("materialised sets whose GROUP_M is neither end -- they do not tune it, so it "
                     "should say which behaviour they mean:\n  " + "\n  ".join(bad))

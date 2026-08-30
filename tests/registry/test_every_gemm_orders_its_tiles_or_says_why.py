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


def test_the_ladder_keeps_the_end_that_wins() -> None:
    """`GROUP_M = 1` -- columns first -- must stay in every ladder, because it is what wins.

    This test used to demand BOTH ends: 1, and a rung at or above 65536 that `tile_order` clamps to
    n_m, which is the row-first order every kernel had before the axis existed. Keeping the old
    behaviour reachable was the right instinct and the measurement retired the row-first half of it.

    Across 144 tuned units on an A6000 -- every unit of the visit-order sweep plus the 2/8 probe --
    `GROUP_M = 65536` won ALONE zero times. It tied for first in 35, always with 1, 4 or 16 tying
    too, and removing it costs 1.0000x: in all 144 some smaller rung is at least as fast.

    Why it loses is a property of the operands, not of the card. Row-first makes consecutive
    programs share a WEIGHT column strip; column-first makes them share an ACTIVATION row strip.
    Here the weights are (K, N) with K and N at most 1536, so 128 KB at a typical shape and never
    more than a few MB -- they sit in L2 whatever the order. The activation is (M, K) with M = L*L
    on the pair side: 4 MB at L=128 and 64 MB at L=512, against an A6000's 6 MB of L2. So row-first
    optimises reuse of the operand that is cached anyway and evicts the one that is not, and
    column-first does the reverse.

    That argument is about the RATIO, so it moves with the card -- a B200's 126 MB L2 holds the
    activation up to about L=718. What it does not do is reverse: the weights are small on every
    card, so there is no card on which keeping them resident is the scarce thing. That is why this
    is a removal and not an A6000 tuning choice.

    The other end of that question -- whether a rung BETWEEN 1 and 16 helps -- was benched
    separately, because 2 and 8 had never been measured at all: 18 units over four GEMMs at
    4,608 configs each, with the other axes narrowed to values both sweeps show winning. Across all
    154 buckets that carry more than one GROUP_M, 2 wins alone once and 8 wins alone never, and
    keeping only 1/4/16 costs 1.0000x median and 1.0064x worst. So the ladder is three rungs
    because three is what the measurement supports, not because nobody looked between them.

    16 came out too, and the head-to-head is why. "16 wins alone in 12 buckets" is a count, not a
    size: over the 130 buckets that belong to kernels still in the registry and carry more than one
    GROUP_M, 16 beats 4 by more than the 1.059x noise floor ZERO times, while 4 beats 16 above it
    three times, by up to 1.134x. More than half the buckets (83 of 154 before the dead kernels are
    dropped) are within 0.5% of each other. So 16 is not a rung the tuner needs when 4 is there.

    The ladder `1 4` costs 1.0000x median and 1.0500x worst, and four of 130 buckets lose more than
    1%. That worst case is one bucket where 16 and 65536 tie and both 1 and 4 sit 5% back -- still
    inside the noise floor, and the price of it is 23% of the whole build (171 -> 132 GPU-hours at
    0.24 s a config). Two of the five worst buckets for this ladder belonged to
    trimul_outproj_gemm_sigmoid, a kernel since removed for being unreachable, which is why the
    live-kernel figure is the one quoted.
    """
    cfg = ROOT / "src/miniworld_engine/autotune/configs"
    live = {r["kernel"] for r in _gemms()}
    bad = []
    for f in sorted(cfg.rglob("*.csv")):
        if f.stem not in live:
            continue
        with f.open(newline="") as fh:
            if not fh.readline().startswith("axis,"):
                continue
        for line in f.read_text().splitlines():
            if not line.startswith("GROUP_M,"):
                continue
            values = {int(v) for v in line.split(",", 1)[1].split()}
            if 1 not in values:
                bad.append(f"{f.parent.name}/{f.name}: no 1 (columns first) -- the end that wins")
            if any(v >= 65536 for v in values):
                bad.append(f"{f.parent.name}/{f.name}: still carries a row-first rung, which won "
                           f"alone in 0 of 144 units and costs 1.0000x to drop")
    assert not bad, ("GROUP_M ladders that no longer say what the measurement says:\n  "
                     + "\n  ".join(bad))


def test_the_materialised_sets_pin_one_value() -> None:
    """A set that lists whole configs is not searching this axis, so its one value has to be one
    the tuner would have picked. 1 and 4 are those: over 144 units, pinning to 1 costs 1.000x
    median and 1.067x worst, pinning to 4 costs 1.000x and 1.050x. 16 is worse at the tail (1.134x)
    and 65536 is gone from the ladder entirely -- the thirteen sets that pinned it were repointed
    to 4, which ties or beats it in every unit measured."""
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
        if seen - {"1", "4"}:
            bad.append(f"{f.parent.name}/{f.name}: {sorted(seen)}")
    assert not bad, ("materialised sets pinning a GROUP_M the measurement does not support -- they "
                     "do not tune it, so the one value has to be one that wins:\n  "
                     + "\n  ".join(bad))

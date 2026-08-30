"""adaln, conditioned_transition and augmented_attention are driven per side, like `level=both`.

These seventeen rows say `level=atom` because `atom_key` is the key function they call. That is
about the KEY, and it was read as a statement about the SHAPES: they were driven as one list of
atom lengths crossed with every width the single stream has. Both halves of that are wrong, and a
build shows it in opposite directions.

  * It built shapes the model never presents. An atom count of 8192 at d=768 is an atom activation
    with a token width -- eight of the eighteen units per precision, and the largest ones.
  * It missed the shapes the model does present. `atom_key` started at 256, so a token count of
    128 and one of 384 both floored into the 256 bucket -- and krystal builds one
    DiffusionTransformer block class 24 times on the token side (d_single=768, d_cond=384) against
    3 times on the atom side (128/128). The side that is 24 of 27 blocks had no bucket of its own.

So each is two work lists: token counts 256/384/512/768 at d 384/512/768, and atom counts
1024..8192 at d 128 and only 128. The ranges are disjoint on purpose -- that is what lets one
floor-clamp key both sides, where `level=both` needs `both_key`'s row count because a pair L and an
atom A of the same value are different launches.
"""
from __future__ import annotations

import csv
from pathlib import Path

from miniworld_engine.autotune.builder import op_units
from miniworld_engine.autotune.shape_key import (
    ATOM_KEY_BUCKETS,
    DIT_ATOM_LENGTHS,
    DIT_TOKEN_LENGTHS,
    atom_key,
    unpack_base,
)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REG = ROOT / "src/miniworld_engine/kernels/registry.csv"
ATOM_WIDTH = 128
TOKEN_WIDTHS = (384, 512, 768)


def _rows() -> list[dict]:
    with REG.open(newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if r["level"] == "atom" and (r["width"] or "").strip() == "single"
                and r["backend"] == "triton" and (r["driver"] or "").strip()]


def test_the_column_pair_still_selects_the_dit_families() -> None:
    """`level=atom, width=single` is the predicate this split is derived from, so it has to keep
    naming the three families and nothing else. A fourth family arriving here is not a failure --
    it is a row to look at, because it would silently inherit a two-sided work list."""
    fams = {r["family"] for r in _rows()}
    assert fams == {"adaln", "conditioned_transition", "augmented_attention"}, (
        f"level=atom + width=single now selects {sorted(fams)}. Either a family was added to the "
        f"DiT split without meaning to, or one left it and the columns no longer say so.")


def test_each_is_driven_from_both_streams_with_that_streams_widths() -> None:
    by_op: dict[str, list] = {}
    for u in op_units():
        by_op.setdefault(u.op, []).append(u)
    bad = []
    for r in _rows():
        us = by_op.get(r["kernel"], [])
        for dtype in {u.dtype for u in us}:
            mine = [u for u in us if u.dtype == dtype]
            token = {(u.length, u.width) for u in mine if u.side == "token"}
            atom = {(u.length, u.width) for u in mine if u.side == "atom"}
            want_t = {(L, w) for L in DIT_TOKEN_LENGTHS for w in TOKEN_WIDTHS}
            want_a = {(A, ATOM_WIDTH) for A in DIT_ATOM_LENGTHS}
            if token != want_t:
                bad.append(f"{r['kernel']} [{dtype}] token side: {sorted(token)}")
            if atom != want_a:
                bad.append(f"{r['kernel']} [{dtype}] atom side: {sorted(atom)}")
            if {u.side for u in mine} != {"token", "atom"}:
                bad.append(f"{r['kernel']} [{dtype}] sides: {sorted({u.side for u in mine})}")
    assert not bad, "\n  ".join(["a DiT family is not driven per side:", *bad])


def test_no_unit_pairs_an_atom_length_with_a_token_width() -> None:
    """The shape that motivated the split. It is worth its own name because it is the expensive
    half: an atom count of 8192 at d=768 is the biggest activation the old work list built and the
    one the model never asks for."""
    ops = {r["kernel"] for r in _rows()}
    bad = sorted({(u.op, u.length, u.width) for u in op_units()
                  if u.op in ops and u.length in DIT_ATOM_LENGTHS and u.width != ATOM_WIDTH})
    assert not bad, f"atom counts built at a token width: {bad}"


def test_a_token_length_gets_its_own_bucket() -> None:
    """The other half. Before, `atom_key` floored 128 and 384 into 256 -- so the token side shared
    a bucket with an atom count, and 384 (d_cond, 24 of 27 blocks) had none at all."""
    for L in DIT_TOKEN_LENGTHS:
        assert unpack_base(atom_key(L), 0) == L, (
            f"token length {L} does not key to itself; ATOM_KEY_BUCKETS={ATOM_KEY_BUCKETS}")
    assert not (set(DIT_TOKEN_LENGTHS) & set(DIT_ATOM_LENGTHS)), (
        "the two sides share a length, so one floor-clamp cannot tell them apart and the key "
        "would have to move to row counts the way level=both did")


def test_the_token_side_is_not_built_as_a_pair() -> None:
    """`drivers.both_level_is_pair` used to infer the side from the length, and every token unit at 512
    or under would have taken that path: (B, L, L, D) is 262,144 rows where the model hands 512."""
    import os

    from miniworld_engine.kernels import drivers

    prev = os.environ.get("MINIWORLD_DRIVER_SIDE")
    try:
        for L in DIT_TOKEN_LENGTHS:
            os.environ["MINIWORLD_DRIVER_SIDE"] = "token"
            assert drivers.both_level_is_pair(L) is False, f"token L={L} built as a pair activation"
    finally:
        if prev is None:
            os.environ.pop("MINIWORLD_DRIVER_SIDE", None)
        else:
            os.environ["MINIWORLD_DRIVER_SIDE"] = prev

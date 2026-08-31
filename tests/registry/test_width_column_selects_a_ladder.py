"""Every `width` value has to name a ladder the builder knows.

The width column decides which channel widths a kernel is tuned at. `op_units` looks the value up
with `LADDER.get(klass, LADDER["both"])`, so a typo does not fail -- it silently hands back the
UNION ladder, which builds widths the kernel never sees and costs build time no one asked for.
The default is there for rows that legitimately leave the column blank; it should never be reached
by a value that was meant to say something.

The column is also the ONLY thing that knows which stream a `level=token`/`level=atom` kernel sees.
`level` picks the key function, not the width: adaln, conditioned_transition and augmented_attention
all key on `atom_key` and all run at d_single_token=768 in krystal's 24 `token_dit` blocks.
"""
from __future__ import annotations

import csv

from paths import REGISTRY as REG

#: The four the builder's LADDER defines. `atom` is the fixed atom-stream width (128); `pair` and
#: `single` are the two streams' ladders; `both` is the union, for a kernel that meets both.
KNOWN = {"atom", "pair", "single", "both"}


def _rows() -> list[dict]:
    with REG.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_width_value_names_a_known_ladder() -> None:
    bad = sorted({(r["kernel"], r["width"]) for r in _rows()
                  if (r.get("width") or "").strip() and r["width"].strip() not in KNOWN})
    assert not bad, (
        "width values with no ladder in autotune/builder.py::op_units -- these fall through to the "
        f"union ladder and are tuned at widths they never see: {bad}"
    )


def test_the_builders_ladder_defines_exactly_these() -> None:
    """The test's vocabulary and the builder's must not drift apart."""
    src = (REG.parent.parent / "autotune/builder.py").read_text()
    body = src.split("LADDER = {", 1)[1].split("}", 1)[0]
    defined = {line.split('"')[1] for line in body.splitlines() if '"' in line}
    assert defined == KNOWN, f"builder defines {sorted(defined)}, this test knows {sorted(KNOWN)}"


def test_a_triton_row_declares_a_width() -> None:
    """A blank column is the union ladder by default, which is a decision, not an omission."""
    blank = sorted(r["kernel"] for r in _rows()
                   if r["backend"] == "triton" and not (r.get("width") or "").strip())
    assert not blank, f"triton rows with no width declared: {blank}"


def test_a_both_level_row_declares_the_union_and_nothing_else() -> None:
    """`level=both` is the one case where the column cannot decide anything -- and must still agree.

    A `both` row is driven once per SIDE (`op_units` splits it into pair units and atom units), and
    the side names the stream outright, so `_widths` takes the ladder from the side and never looks
    at the column. The value is therefore inert on these 27 rows, which is how a review found it.

    Inert is not the same as free to be wrong. A row saying `level=both,width=pair` would be a
    contradiction -- it claims to meet only one stream while its own level says it meets two -- and
    nothing would have caught it, because nothing reads the cell. Pinning the biconditional turns
    the dead value into a consistency check: `width=both` exactly when `level=both`.
    """
    wrong = sorted(f"{r['kernel']}: level={r['level']} width={(r.get('width') or '').strip()}"
                   for r in _rows() if r["backend"] == "triton"
                   and (r["level"] == "both") != ((r.get("width") or "").strip() == "both"))
    assert not wrong, (
        "level and width disagree about whether the kernel meets both streams. A `level=both` row "
        "is driven once per side and its ladder comes from the side, so the column can only say "
        "`both`; and a row that says `both` while its level names one stream is claiming a ladder "
        "it will never be given:\n  " + "\n  ".join(wrong))

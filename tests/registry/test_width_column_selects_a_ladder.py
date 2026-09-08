"""Every `width` value has to name a ladder the builder knows.

The width column decides which channel widths a kernel is tuned at. `op_units` looks the value up
with `LADDER.get(klass, LADDER["both"])`, so a typo does not fail -- it silently hands back the
UNION ladder, which builds widths the kernel never sees and costs build time no one asked for.
The default is there for rows that legitimately leave the column blank; it should never be reached
by a value that was meant to say something.

The column is also the ONLY thing that knows which stream a `level=token`/`level=atom` kernel sees.
`level` picks the key function, not the width: adaln, conditioned_transition and augmented_attention
all key on `atom_key` and all run at d_single_token=768 in the model's 24 `token_dit` blocks.
"""
from __future__ import annotations

from paths import REGISTRY as REG
from paths import registry_rows

#: The four the builder's LADDER defines. `atom` is the fixed atom-stream width (128); `pair` and
#: `single` are the two streams' ladders; `both` is the union, for a kernel that meets both.
#: `head_dim`, `pair_bidir` and `expand_nd` are DERIVED classes: the number in those kernels'
#: buckets is `d_hidden // n_head`, `2 * d_pair` and `n * d_hidden`, none of which a stream
#: ladder produces.
KNOWN = {"atom", "pair", "single", "both", "head_dim", "pair_bidir", "expand_nd"}

#: The derived subset of KNOWN -- the classes whose value is computed from a stream
#: width rather than being one. `_widths` returns these before it dispatches on side.
DERIVED = {"head_dim", "pair_bidir", "expand_nd"}


def _rows() -> list[dict]:
    return registry_rows()


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
    defined = set()
    # TWO declarations, because a width class is one of two kinds. `LADDER` holds the STREAM
    # classes, whose rungs are a stream's channel width. `DERIVED_WIDTHS` holds the ones whose
    # number is computed from a stream width and is therefore on no stream ladder --
    # `d_hidden // n_head`, `2 * d_pair`. Reading only the first is how a row could say
    # `width=head_dim`, fall through to the union, and be tuned at widths it never sees.
    for name in ("LADDER = {", "DERIVED_WIDTHS = {"):
        body = src.split(name, 1)[1].split("}", 1)[0]
        defined |= {line.split('"')[1] for line in body.splitlines() if '"' in line}
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
    # ...or a DERIVED class. `_widths` reads DERIVED_WIDTHS before it looks at the side, so a
    # derived value IS read on a `level=both` row and is not a stream claim at all: it names a
    # computed quantity (`n * d_hidden`, `d_hidden // n_head`) that no side's ladder produces.
    # `transition_expand_swiglu_triton` is the case that forced this. It is `level=both` -- the
    # kernel really does meet both streams -- and it folds ND into its key, so the side ladders
    # could only ever build the one ND its driver happened to pin, which is what `--replay`
    # measured: 1024 and 1536 asked for, 512 built, five times over.
    # Two directions, not one equality: a `level=both` row must say `both` or a derived class,
    # and `both` may only appear on a `level=both` row. A DERIVED class is free at any level --
    # `triangle_attention` is `level=token, width=head_dim` and that is the whole point of it.
    allowed = {"both", *DERIVED}
    wrong = sorted(
        f"{r['kernel']}: level={r['level']} width={(r.get('width') or '').strip()}"
        for r in _rows() if r["backend"] == "triton"
        and ((r["level"] == "both" and (r.get("width") or "").strip() not in allowed)
             or ((r.get("width") or "").strip() == "both" and r["level"] != "both")))
    assert not wrong, (
        "level and width disagree about whether the kernel meets both streams. A `level=both` row "
        "is driven once per side and its ladder comes from the side, so the column can only say "
        "`both` or name a derived class; and a row that says `both` while its level names one "
        "stream is claiming a ladder it will never be given:\n  " + "\n  ".join(wrong))

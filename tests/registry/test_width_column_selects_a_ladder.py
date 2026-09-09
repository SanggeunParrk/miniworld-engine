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
#: A row whose bucket carries something that is not a stream width says so in `key_axis` instead,
#: naming the kernel's own constexpr.
KNOWN = {"atom", "pair", "single", "both"}

#: The axis names a row may put in `key_axis` -- the constexpr its bucket actually carries, as the
#: KERNEL spells it in `pack(...)`. These are not width classes and never appear in `width`: the
#: two columns answer different questions, which stream the kernel sees and which axis its key
#: folds in. They were one column for a while, with `head_dim` / `pair_bidir` / `expand_nd` as
#: invented values, and a reader had no way to tell those from the stream names beside them.
AXES = {"HEAD_DIM", "H", "ND"}


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
    body = src.split("LADDER = {", 1)[1].split("}", 1)[0]
    defined |= {line.split('"')[1] for line in body.splitlines() if '"' in line}
    assert defined == KNOWN, f"builder defines {sorted(defined)}, this test knows {sorted(KNOWN)}"
    # and the axis ladders, under the names the kernels use
    body = src.split("AXIS_LADDERS = {", 1)[1].split("}", 1)[0]
    axes = {line.split('"')[1] for line in body.splitlines() if '"' in line}
    assert axes == AXES, f"builder drives axes {sorted(axes)}, this test knows {sorted(AXES)}"


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
    # `level=both` says `both`, and `both` appears nowhere else. The column is a stream name
    # again: a row whose key carries something other than a stream width says that in `key_axis`,
    # which is read independently of this.
    allowed = {"both"}
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


def test_key_axis_names_an_axis_the_kernel_actually_packs() -> None:
    """`key_axis` is the kernel's own constexpr, checked against the source, not a label.

    The column exists because `width` was carrying two different questions. A kernel whose bucket
    folds in `d_hidden // n_head` is still a PAIR-stream kernel; saying `width=head_dim` answered
    "which axis" in the cell that answers "which stream", and `head_dim` / `pair_bidir` /
    `expand_nd` were names this repository invented for `HEAD_DIM`, `H` and `ND` -- so a reader
    could not tell them from `atom` and `single` beside them, nor find them by grepping the kernels.

    Now the cell names what `pack(...)` names. This is what keeps it true.
    """
    import ast

    from paths import PKG

    bad = []
    for r in _rows():
        axes = [a for a in (r.get("key_axis") or "").split("|") if a]
        if not axes:
            continue
        path = PKG.parent / r["file"]
        try:
            tree = ast.parse(path.read_text())
        except OSError:
            bad.append(f"{r['kernel']}: {r['file']} cannot be read")
            continue
        packed: set[str] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) in ("pack", "token_key", "atom_key",
                                                           "both_key")):
                packed |= {kw.arg for kw in node.keywords if kw.arg}
        missing = [a for a in axes if a not in packed]
        if missing:
            bad.append(f"{r['kernel']}: key_axis names {missing}, and {r['file']} packs "
                       f"{sorted(packed) or 'nothing'}")
    assert not bad, (
        "a row's key_axis names an axis its kernel does not pack. The cell has to be the name the "
        "launcher uses, or it is a label that drifts:\n  " + "\n  ".join(bad))


def test_a_row_that_packs_a_derived_axis_declares_it() -> None:
    """The other direction: a kernel folding an axis no stream ladder produces must SAY so.

    Without this the column is optional, and an unset cell reads the same as "this kernel's bucket
    is a plain stream width" -- which is exactly the state that had `triangle_attention` tuned at
    d_pair's rungs while its key carried head dims, for as long as the column had no way to say
    otherwise.
    """
    derived = {"HEAD_DIM", "H", "ND"}
    bad = []
    for r in _rows():
        if r["backend"] != "triton" or (r.get("developed") or "yes").strip() == "no":
            continue
        axes = {a for a in (r.get("key_axis") or "").split("|") if a}
        import ast

        from paths import PKG
        try:
            tree = ast.parse((PKG.parent / r["file"]).read_text())
        except OSError:
            continue
        packed: set[str] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) in ("pack", "token_key", "atom_key",
                                                           "both_key")):
                packed |= {kw.arg for kw in node.keywords if kw.arg}
        undeclared = (packed & derived) - axes
        if undeclared:
            bad.append(f"{r['kernel']}: {r['file']} packs {sorted(undeclared)} and key_axis is "
                       f"{r.get('key_axis')!r}")
    assert not bad, (
        "a kernel folds a derived axis into its bucket and its row does not declare it, so the "
        "build tunes it at whatever the stream ladder happens to carry:\n  " + "\n  ".join(bad))

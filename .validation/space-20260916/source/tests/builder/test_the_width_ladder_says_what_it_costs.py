"""The build sweeps widths the model runs, plus declared headroom, and the headroom has a price.

The point of this file has not changed: headroom -- a width kept so a config that widens d_pair
finds a tuned cache instead of a miss -- must be a legible decision, not a number that quietly
grows in a tuple. What changed is where the decision lives.

It used to be two literals inside `op_units` (`PRESENTED` and `HEADROOM_PAIR`), read out of the
source by this file. That arrangement is what these tests were guarding against, and it lost: the
literals were the build's second, independent statement of the shapes, and they drifted from what
`cases()` actually runs -- token lengths stopped at 512 while the sweep ran to 1024, MSA widths
held 64 while the config declares 64 and 128. 146 replay misses came out of the gap, and no test
could see it because both sides were hand-written and neither was the model.

`registry_module.csv` is the single statement now, and every row says where its numbers come from
in a `source` column. So headroom is not a separate literal any more -- it is a row whose `source`
says "headroom", and its cost is the units those rows contribute. That is what these tests pin.
"""
from __future__ import annotations

from miniworld_engine.autotune import builder
from miniworld_engine.autotune.module_registry import module_rows

ROWS = module_rows()
#: A row is headroom when its own `source` column says so. Not inferred from the width: 256 is
#: headroom for d_pair and production for d_hidden_tri_multi, and a rule that guessed from the
#: number would call one of them wrong.
HEADROOM = [r for r in ROWS if "headroom" in r.source.lower()]
PRODUCTION = [r for r in ROWS if "headroom" not in r.source.lower()]


def test_every_row_says_where_its_numbers_came_from() -> None:
    """The check the old literals could not make: a width with no stated origin.

    `PRESENTED` and `HEADROOM_PAIR` said which half a number was in and nothing about why it was
    that number. Half of them turned out to be neither -- 384 was in `cases()` and in no other
    declaration, so it was swept by the module pass and never by the op pass, forever."""
    for row in ROWS:
        assert row.source.strip(), f"{row.module} {row.dims}: no source"


def test_the_model_widths_are_the_models_own() -> None:
    """AlphaFold-3's c_atom 128, c_s 384, c_token 768, and MiniWorld's d_pair 128.

    Same assertion the old `test_the_presented_widths_are_the_models_own` made, against the file
    the model's own config was copied into rather than against a literal in the builder."""
    widths = {v for r in PRODUCTION for v in r.dims.values()}
    for want, why in ((128, "c_atom / d_pair"), (384, "c_s / d_single"), (768, "c_token")):
        assert want in widths, f"no production row runs at {want} ({why})"


def test_the_headroom_is_declared_and_not_the_majority() -> None:
    """Headroom is a decision about the future and it is allowed to cost something -- but a build
    that spends most of itself on shapes nothing asks for today is a different decision, and it
    should not arrive by accident."""
    assert HEADROOM, "no row is marked headroom; the split has become invisible again"
    units = builder.units(builder.cases())
    by_module_dims = {(r.module, tuple(sorted(r.dims.items()))) for r in HEADROOM}
    cases = {c.name: c for c in builder.cases()}
    cost = sum(1 for u in units
               if (u.case, tuple(sorted(cases[u.case].dims[u.dim_index].items())))
               in by_module_dims)
    share = cost / len(units)
    assert share < 0.5, (
        f"{cost} of {len(units)} units ({share:.0%}) are headroom widths nothing runs today. "
        f"That is most of a build; either the headroom or this bound is the wrong call, but it "
        f"has to be made on purpose.")


#: Widths the model FIXES: AlphaFold-3's c_atom / c_s / c_token and the MSA hidden width. Every
#: config -- debug, small, medium, large -- sets these to the same numbers and differs in block
#: counts, so a value outside the model's own is not headroom, it is a shape nothing will present.
FIXED_AXES = ("d_single", "d_cond", "d_model", "d_single_atom", "d_hidden_msa")


def test_the_headroom_is_on_a_pair_axis() -> None:
    """The single ladder carried 256 and 512 once and they went when someone checked that no
    config presents them. Nothing has re-added them.

    The axis, not the stream: `attention_pair_bias` runs on `token_single` and its headroom row
    widens `d_pair` to 256, which is pair headroom on a single-stream module. Reading the stream
    instead of the dims name calls that a violation, and it is not one."""
    for row in HEADROOM:
        widened = [a for a in row.dims if a in FIXED_AXES
                   and row.dims[a] not in (128, 384, 768, 32)]
        assert not widened, (
            f"{row.module} declares headroom on {widened}; those widths are fixed by the model "
            f"(c_atom 128 / c_s 384 / c_token 768) and headroom there tunes nothing")

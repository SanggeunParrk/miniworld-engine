"""The build sweeps widths the model runs, plus declared headroom, and the headroom has a price.

`op_units` drives each kernel at a ladder of channel widths. Three of them are the model's own --
AlphaFold-3's c_atom (128), c_s (384) and c_token (768), with d_pair at 128 -- and two, 256 and 512,
are headroom: kept so a config that widens d_pair finds a tuned cache instead of a miss.

The ladder used to be one literal, so the headroom was invisible and free-looking. It is neither:
674 of `build all`'s 1,827 units are at 256 or 512, which is 37% of a full build's GPU time spent on
shapes nothing asks for today. That is a decision about the future, and a decision has to be
legible. This file pins the split and the cost, so dropping or extending the headroom is a visible
edit rather than a number quietly changing in a tuple.

The single ladder was already cut this way, for this reason: it carried 256 and 512 too, and they
went when someone checked that no config presents them.
"""
from __future__ import annotations

from miniworld_engine.autotune import builder
from miniworld_engine.autotune.configs import config_set


def _ladders():
    """The literals `op_units` defines, read out of its source rather than re-declared here."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(builder.op_units))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("PRESENTED", "HEADROOM_PAIR", "ATOM_WIDTH"):
                found[name] = ast.literal_eval(node.value)
    return found


def test_the_two_halves_are_still_declared_separately() -> None:
    """Guard the guard: folding them back into one literal would make every check below vacuous."""
    got = _ladders()
    assert set(got) == {"PRESENTED", "HEADROOM_PAIR", "ATOM_WIDTH"}, (
        f"op_units no longer names the halves separately (found {sorted(got)}); the headroom's "
        f"cost becomes invisible again")


def test_the_presented_widths_are_the_models_own() -> None:
    """128 / 384 / 768 are c_atom, c_s and c_token. A fourth would mean the model changed, and the
    ladder has to change with it -- silently missing one is what `driver_width` exists to stop."""
    got = _ladders()
    assert got["ATOM_WIDTH"] == 128
    assert got["PRESENTED"]["atom"] == (128,)
    assert got["PRESENTED"]["pair"] == (128,)
    assert got["PRESENTED"]["single"] == (128, 384, 768)


def test_the_headroom_is_pair_only_and_named() -> None:
    """Headroom on the single side was removed once already, having cost a third of every build for
    widths no config presents. It must not come back by being added to a shared tuple."""
    got = _ladders()
    assert got["HEADROOM_PAIR"] == (256, 512)
    presented = {w for ws in got["PRESENTED"].values() for w in ws}
    assert not (set(got["HEADROOM_PAIR"]) & presented), (
        "a width cannot be both presented and headroom; the split would say nothing")


def test_the_headroom_still_costs_what_the_comment_says() -> None:
    """The number in the comment is the whole argument for making this a decision. If it drifts,
    the decision is being made against a stale price."""
    units = builder.op_units(config_dir=config_set("grid"))
    got = _ladders()
    head = set(got["HEADROOM_PAIR"])
    extra = [u for u in units if u.width in head]
    share = len(extra) / len(units)
    assert 0.25 < share < 0.50, (
        f"headroom is now {len(extra)} of {len(units)} units ({share:.0%}); op_units' comment says "
        f"674 of 1,827 (37%). Re-measure and update the comment, or the cost written down is not "
        f"the cost being paid.")
    without = [u for u in units if u.width not in head]
    assert without, "dropping the headroom would leave no units at all"

"""Every side `op_units` emits must be one the build subprocess will accept.

The builder runs each unit as a child process and passes the side through `--side`. Parent and
child therefore have to agree on the vocabulary, and they silently did not: "token" was added to
`op_units` when the DiT families and the `level=both` rows were split by stream, while the child's
`choices` tuple stayed ("", "pair", "atom"). Every token unit then died in argparse -- 3 seconds,
0 ops, "invalid choice" -- and the build reported it as a plain FAIL alongside real ones. 52 of
the first 129 units in the next build went that way before anyone read a unit log.

Nothing else catches this: the parser lives in `__main__` and no test had run a unit through it.
"""
from __future__ import annotations

import ast

from paths import ROOT

from miniworld_engine.autotune.builder import op_units

BUILDER = ROOT / "src" / "miniworld_engine" / "autotune" / "builder.py"


def _side_choices() -> set[str]:
    """The `choices=` tuple on the child's `--side`, read from the source rather than imported:
    it is built inside `__main__`, which importing this module does not run."""
    tree = ast.parse(BUILDER.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "add_argument":
            continue
        if not (node.args and getattr(node.args[0], "value", None) == "--side"):
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                return {getattr(e, "value", None) for e in kw.value.elts}
    msg = "no `--side` argument with a `choices=` tuple in builder.py"
    raise AssertionError(msg)


def test_every_side_the_planner_emits_is_a_side_the_child_accepts() -> None:
    emitted = {u.side for u in op_units()}
    accepted = _side_choices()
    missing = sorted(emitted - accepted)
    assert not missing, (
        f"op_units emits side(s) {missing} that builder.py's --side refuses "
        f"(choices={sorted(accepted)}). Every unit with one of those sides fails in argparse "
        f"before it reaches a kernel.")


def test_the_child_accepts_no_side_the_planner_cannot_emit() -> None:
    """The other direction is not a crash, but a choice nothing produces is a claim that some
    shape is built when it is not -- the same way an exempt list outlives its kernel."""
    stale = sorted(_side_choices() - {u.side for u in op_units()})
    assert not stale, (
        f"builder.py's --side accepts {stale}, which `op_units` never emits. Either a work list "
        f"stopped driving that side or the choice is left over from one that did.")

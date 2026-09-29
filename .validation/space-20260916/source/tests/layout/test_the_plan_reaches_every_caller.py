"""A kernel has to be built at the widths and lengths of every case that CALLS it.

`registry.csv` says which family a kernel is in. That is not who calls it, and for the shared
kernels the two differ: `layernorm_fwd_saveact_triton` is reached by the transition, the trimul and
the DiT, each handing it its own width. Nothing tied a case's declared dims to the kernels that
case dispatches into, so "does the plan cover production" could only be answered by RUNNING
production -- `dev audit --replay`, on a card, against a finished cache -- and it answered with a
list of missed keys rather than a rule.

Every miss this repository has recorded came through that gap. The registry rows were right and
the ladders beside them were not:

    TOKEN_SHAPES stopped at 512 while cases() runs to 1024
    DIT_TOKEN_LENGTHS stopped at 768, same
    MSA_WIDTHS held 64 while cases() declares d_msa 64 and 128
    the pair side had no 384, which the shared layernorms are handed by the DiT

`dev callers` records the link by running each case once and noting which kernels fired. With it
this is arithmetic, at import time, for free.
"""
from __future__ import annotations

import json

import pytest
from paths import PKG

from miniworld_engine.autotune import builder

CALLERS = PKG / "kernels" / "case_callers.json"

#: dims keys that are not a width: head COUNTS (a driver turns one into a head dim) and the
#: bookkeeping fields a case carries alongside its shape.
NOT_A_WIDTH = frozenset({"n_head", "n_heads", "mask_prob", "n_layers", "batch_size"})


def _callers() -> dict:
    if not CALLERS.is_file():
        pytest.skip(f"{CALLERS.name} has not been recorded yet -- run `dev callers` on a card")
    return json.loads(CALLERS.read_text())


def test_every_caller_length_is_built_for_the_kernels_it_reaches() -> None:
    """The half that `TOKEN_SHAPES` and `DIT_TOKEN_LENGTHS` got wrong."""
    plan: dict[str, set[int]] = {}
    for u in builder.op_units(None):
        plan.setdefault(u.op, set()).add(u.length)
    bad = []
    for name, rec in sorted(_callers().items()):
        for op in rec["ops"]:
            if op not in plan:
                continue          # not a triton row with a driver; a different test owns that
            missing = sorted(set(rec["lengths"]) - plan[op])
            if missing:
                bad.append(f"{op} is called by {name} at lengths {missing}, and the sweep drives "
                           f"{sorted(plan[op])}")
    assert not bad, ("a kernel is not built at a length the case calling it runs:\n  "
                     + "\n  ".join(bad))


def test_every_caller_width_is_built_for_the_kernels_it_reaches() -> None:
    """The half the side ladders got wrong -- a shared kernel handed a width from another stream."""
    plan: dict[str, set[int]] = {}
    for u in builder.op_units(None):
        s = plan.setdefault(u.op, set())
        if u.width:
            s.add(u.width)
        if u.heads:
            s.add(u.heads)
    bad = []
    for name, rec in sorted(_callers().items()):
        want = {v for d in rec["dims"] for k, v in d.items()
                if k not in NOT_A_WIDTH and isinstance(v, int)}
        for op in rec["ops"]:
            if op not in plan:
                continue
            missing = sorted(want - plan[op])
            if missing:
                bad.append(f"{op} is called by {name} at widths {missing}, and the sweep drives "
                           f"{sorted(plan[op])}")
    assert not bad, ("a kernel is not built at a width the case calling it declares:\n  "
                     + "\n  ".join(bad)
                     + "\n\nA width here is a number the case's own dims carry. A kernel may take "
                       "it through a derivation -- the transition expands by n, a head dim is "
                       "d_hidden // n_head -- so a miss can mean either the ladder or the "
                       "derivation; `dev buckets` says which.")


def test_the_record_is_not_stale() -> None:
    """Every case in `cases()` has to be in the file, or the checks above skip it silently."""
    rec = _callers()
    missing = sorted({c.name for c in builder.cases()} - set(rec))
    assert not missing, (f"cases with no record: {missing}. Re-run `dev callers`; a case added "
                         f"after the last recording is unchecked, which is the state this file "
                         f"exists to end.")

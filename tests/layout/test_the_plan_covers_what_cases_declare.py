"""Every value `cases()` declares has to be reachable by the build plan.

`registry.csv` says which AXIS a kernel's bucket carries. It does not say which VALUES that axis
takes -- those come from ladders written by hand in `autotune/builder.py` and `autotune/shape_key.py`
-- and that is the seam every miss in this repository's history has come through. The registry rows
were right; the tuples beside them were not:

    TOKEN_SHAPES        (128, 256, 384, 512)      cases() runs 256..1024      768 and 1024 missing
    DIT_TOKEN_LENGTHS   (128, 256, 384, 512, 768) cases() runs 256..1024      1024 missing
    MSA_WIDTHS          (64,)                     cases() declares d_msa 64 AND 128

Each was found by `dev audit --replay`, which needs a card, a finished cache and half an hour, and
each was patched by appending another hand-written tuple. This asks the same question for free, at
import time, before anything is built: does the plan reach every number production declares?

It cannot check the PAIRING -- which op sees which width -- because that needs the driver to run
and is what replay is for. It checks the weaker, cheap thing: no declared value is unreachable by
every unit. That is exactly what the three lines above got wrong.
"""
from __future__ import annotations

import collections

import pytest

from miniworld_engine.autotune import builder

#: dims keys that are not a channel width or a length: head COUNTS, which a driver turns into a
#: head dim, and the per-family aliases for a width that is already covered by its own key.
NOT_A_VALUE = frozenset({"n_head", "n_heads", "mask_prob", "n_layers", "batch_size"})

#: Values a case declares that no unit drives, each with the reason. An entry here is a claim that
#: production never reaches a kernel at that number -- not that the number is unimportant -- and
#: the evidence for one is a clean `dev audit --replay`.
NOT_DRIVEN: dict[int, str] = {}


def _declared() -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """(width value -> who declares it, length -> who declares it)."""
    widths: dict[int, list[str]] = collections.defaultdict(list)
    lengths: dict[int, list[str]] = collections.defaultdict(list)
    for case in builder.cases():
        for length in case.lengths:
            lengths[int(length)].append(case.name)
        for dims in case.dims:
            for name, value in dims.items():
                if name in NOT_A_VALUE or not isinstance(value, int):
                    continue
                widths[value].append(f"{case.name}.{name}")
    return dict(widths), dict(lengths)


def _planned() -> tuple[set[int], set[int]]:
    units = builder.op_units(None)
    if not units:
        pytest.skip("no op units on this checkout")
    return ({u.width for u in units if u.width} | {u.heads for u in units if u.heads},
            {u.length for u in units})


def test_every_width_a_case_declares_is_driven() -> None:
    widths, _ = _declared()
    planned, _lengths = _planned()
    missing = {v: sorted(set(who)) for v, who in widths.items()
               if v not in planned and v not in NOT_DRIVEN}
    assert not missing, (
        "widths `cases()` declares that no unit drives:\n  "
        + "\n  ".join(f"{v}: {', '.join(who)}" for v, who in sorted(missing.items()))
        + f"\n\nplanned widths: {sorted(planned)}\n"
        "The ladders in `op_units` are hand-written; `cases()` is the declaration of what "
        "production runs. When they disagree the build tunes a bucket nothing asks for and misses "
        "the one it does -- and only `dev audit --replay`, on a card against a finished cache, "
        "says so. Derive the rung from `cases()` rather than adding it here.")


def test_every_length_a_case_declares_is_driven() -> None:
    _, lengths = _declared()
    _planned_widths, planned = _planned()
    missing = {v: sorted(set(who)) for v, who in lengths.items() if v not in planned}
    assert not missing, (
        "lengths `cases()` declares that no unit drives:\n  "
        + "\n  ".join(f"{v}: {', '.join(who)}" for v, who in sorted(missing.items()))
        + f"\n\nplanned lengths: {sorted(planned)}\n"
        "`TOKEN_SHAPES` and `DIT_TOKEN_LENGTHS` are KEY sets -- what `atom_key` floor-clamps into, "
        "kept disjoint so one clamp serves both sides -- and they were reused as WORK lists. A "
        "length production runs and the sweep does not is a bucket that can only ever miss.")


def test_the_declaration_is_not_empty() -> None:
    """Guard the guard: a renamed dims key or lengths field would make both checks vacuous."""
    widths, lengths = _declared()
    assert widths and lengths, "builder.cases() declares no dims or no lengths"

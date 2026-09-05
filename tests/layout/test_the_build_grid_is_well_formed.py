"""What `build all` will actually build, checked as a whole. No GPU, no cache, no measurement.

`op_units()` is a pure function of `registry.csv`, `shape_key.py` and `builder.py`, so "will the
build ask for sane work" is answerable here rather than by launching it and reading warnings an
hour later. Every property below is one that has been violated in this repository:

  * an op silently dropping to zero units (a `developed` flip, a missing grid CSV);
  * a unit pairing an atom length with a token width, or a `width=atom` row being handed token
    widths -- 90 units that built and stored nothing, and one kernel (`rope_fwd_triton`) that does
    not COMPILE at 384;
  * a declared dtype that never becomes a unit, so half a kernel's registry row is fiction;
  * a width no launcher presents, or one a case presents that no ladder drives.

These are structural. Whether the buckets are the ones production ASKS for is a different question
and only `dev audit --replay` can answer it; this is the half that does not need a card.
"""
from __future__ import annotations

import collections
import csv

import pytest
from paths import REGISTRY

from miniworld_engine.autotune.builder import op_units
from miniworld_engine.autotune.shape_key import (
    ATOM_SHAPES,
    DIT_ATOM_LENGTHS,
    DIT_TOKEN_LENGTHS,
    TOKEN_SHAPES,
)

ATOM_WIDTH = 128


@pytest.fixture(scope="module")
def units():
    u = op_units(None)
    assert u, "op_units() produced nothing at all"
    return u


@pytest.fixture(scope="module")
def rows():
    with REGISTRY.open(newline="") as fh:
        return [r for r in csv.DictReader(fh)]


def test_every_declared_triton_kernel_with_a_driver_gets_units(units, rows):
    have = {u.op for u in units}
    want = {r["kernel"] for r in rows
            if r["backend"] == "triton"
            and (r.get("developed") or "yes").strip() != "no"
            and (r.get("driver") or "").strip()}
    missing = sorted(want - have)
    assert not missing, (
        f"declared, developed, driven -- and the build asks for nothing: {missing}. Either the op "
        f"has no config grid in this set, or a `level`/`width`/`sides` cell no longer resolves.")


def test_no_unit_is_a_shape_its_own_registry_row_denies(units, rows):
    """The `width` column is a claim about which stream a kernel sees; a unit must honour it."""
    width_of = {r["kernel"]: (r.get("width") or "both").strip() for r in rows}
    bad = [(u.op, u.side, u.width) for u in units
           if width_of.get(u.op) == "atom" and u.width != ATOM_WIDTH]
    assert not bad, (
        f"`width=atom` rows handed a width they never see: {sorted(set(bad))}. This built 90 units "
        f"that stored nothing, and `rope_fwd_triton` does not compile at 384.")


def test_the_non_pair_side_carries_only_the_atom_and_msa_widths(units, rows):
    """`side="atom"` means "the non-pair side", not "the atom stream".

    For a `level=atom` row it is the atom stream and its width is `ATOM_WIDTH` and nothing else.
    For a `level=both` row it is also where the MSA stack lands: an MSA activation is
    (B, n_msa, n_token, d_msa) whose ROW COUNT falls in these buckets while its width is `d_msa`.
    `dev audit --replay` measured that directly -- `layernorm_fwd_saveact_triton` asking for
    (rows=2048, N=64) and (4096, 64), which no ladder carried until `MSA_WIDTHS` was added.

    So the invariant is not "one width" but "one width per level", and the sharp half is that a
    `level=atom` row must never see 64.
    """
    from miniworld_engine.autotune.builder import op_units as _u  # noqa: F401  (documents source)
    level_of = {r["kernel"]: r["level"] for r in rows}
    MSA_WIDTHS = {64}
    bad_atom = sorted({(u.op, u.width) for u in units
                       if u.side == "atom" and level_of.get(u.op) == "atom"
                       and u.width != ATOM_WIDTH})
    assert not bad_atom, (
        f"`level=atom` rows on the atom side at a width that is not {ATOM_WIDTH}: {bad_atom}")
    bad_both = sorted({(u.op, u.width) for u in units
                       if u.side == "atom" and level_of.get(u.op) == "both"
                       and u.width not in ({ATOM_WIDTH} | MSA_WIDTHS)})
    assert not bad_both, (
        f"`level=both` rows on the non-pair side at a width that is neither the atom width nor an "
        f"MSA width: {bad_both}")


def test_no_atom_level_row_pairs_an_atom_length_with_a_token_width(units, rows):
    level_of = {r["kernel"]: r["level"] for r in rows}
    atom_lengths = set(ATOM_SHAPES) | set(DIT_ATOM_LENGTHS)
    bad = sorted({(u.op, u.length, u.width) for u in units
                  if level_of.get(u.op) == "atom" and u.side == "atom"
                  and u.length in atom_lengths and u.width != ATOM_WIDTH})
    assert not bad, (
        f"atom lengths paired with a token width on a `level=atom` row: {bad}. This is the shape "
        f"`rope_fwd_triton` does not compile at.")


def test_every_declared_dtype_becomes_a_unit(units, rows):
    alias = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}
    by_op = collections.defaultdict(set)
    for u in units:
        by_op[u.op].add(u.dtype)
    bad = []
    for r in rows:
        if r["backend"] != "triton" or (r.get("developed") or "yes").strip() == "no":
            continue
        if r["kernel"] not in by_op:
            continue
        want = {alias.get(t.strip(), t.strip()) for t in (r.get("dtypes") or "bf16").split("|") if t.strip()}
        if want - by_op[r["kernel"]]:
            bad.append((r["kernel"], sorted(want - by_op[r["kernel"]])))
    assert not bad, (
        f"registry declares a precision the build never drives: {bad}. Either the column is "
        f"fiction or a unit filter is dropping it -- both were true here at once.")


def test_lengths_come_from_a_declared_ladder(units):
    known = set(TOKEN_SHAPES) | set(ATOM_SHAPES) | set(DIT_TOKEN_LENGTHS) | set(DIT_ATOM_LENGTHS)
    known |= {length * length for length in TOKEN_SHAPES}      # pair rows
    bad = sorted({(u.op, u.length) for u in units if u.length not in known})
    assert not bad, f"units at a length on no ladder: {bad}"


def test_the_unit_count_is_not_quietly_collapsing(units):
    """A guard on the guard: a filter bug that empties the sweep would pass every test above."""
    ops = {u.op for u in units}
    assert len(units) > 1500, f"only {len(units)} units -- the sweep has collapsed"
    assert len(ops) > 70, f"only {len(ops)} ops have units"
    thin = sorted(op for op in ops if sum(1 for u in units if u.op == op) < 2)
    assert not thin, f"ops reduced to a single unit: {thin}"

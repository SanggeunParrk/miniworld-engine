"""The ``axis,values`` grid spec must expand to exactly the materialised grid it replaces.

A search space written one row per config does not scale: the generated sweep is 205,266 configs
over 91 ops, and its largest op alone is 15,552 rows restating the same six value sets. The spec
form says them once. That is only safe if the two forms are interchangeable, and if a shard's
``slice`` keeps naming the same configs -- expansion order is part of the contract, not an
implementation detail, because reordering the rows of a spec would silently re-cut every shard.
"""
from __future__ import annotations

import itertools

import pytest

from miniworld_engine.autotune.configs import _read

SPEC = """axis,values
BLOCK_M1,32 64 128
BLOCK_N,16 32
num_warps,4 8
num_stages,1 2 3
"""


def _sig(c):
    return (tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages)


def _write(tmp_path, text, name="op.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_spec_expands_to_the_full_product(tmp_path):
    got = _read(_write(tmp_path, SPEC))
    assert len(got) == 3 * 2 * 2 * 3


def test_spec_and_materialised_forms_are_interchangeable(tmp_path):
    """Same configs from both files -- the point of the format, so it is pinned as equality."""
    spec = _read(_write(tmp_path, SPEC))
    rows = ["BLOCK_M1,BLOCK_N,num_warps,num_stages"]
    rows += [f"{m},{n},{w},{s}" for m, n, w, s
             in itertools.product((32, 64, 128), (16, 32), (4, 8), (1, 2, 3))]
    mat = _read(_write(tmp_path, "\n".join(rows) + "\n", "op2.csv"))
    assert {_sig(c) for c in spec} == {_sig(c) for c in mat}


def test_expansion_order_is_product_over_file_order(tmp_path):
    """A shard's slice names positions in this sequence, so the order is part of the contract."""
    got = _read(_write(tmp_path, SPEC))
    want = [((("BLOCK_M1", m), ("BLOCK_N", n)), w, s) for m, n, w, s
            in itertools.product((32, 64, 128), (16, 32), (4, 8), (1, 2, 3))]
    assert [_sig(c) for c in got] == want


def test_slice_takes_a_half_open_range_of_the_product(tmp_path):
    full = _read(_write(tmp_path, SPEC))
    part = _read(_write(tmp_path, SPEC + "slice,5-11\n", "op3.csv"))
    assert [_sig(c) for c in part] == [_sig(c) for c in full[5:11]]


def test_slices_of_one_spec_reconstruct_it_exactly(tmp_path):
    """What config-set sharding depends on: the shards must union back to the whole grid."""
    full = _read(_write(tmp_path, SPEC))
    parts = []
    for i, (a, b) in enumerate([(0, 13), (13, 26), (26, 36)]):
        parts += _read(_write(tmp_path, SPEC + f"slice,{a}-{b}\n", f"s{i}.csv"))
    assert [_sig(c) for c in parts] == [_sig(c) for c in full]


@pytest.mark.parametrize("bad,why", [
    ("axis,values\nBLOCK_M1,32 64\nnum_warps,4\n", "no num_stages row"),
    ("axis,values\nnum_warps,4\nnum_stages,2\n", "no tile-axis row"),
    ("axis,values\nBLOCK_M1,\nnum_warps,4\nnum_stages,2\n", "axis with no values"),
    ("axis,values\nBLOCK_M1,32\nBLOCK_M1,64\nnum_warps,4\nnum_stages,2\n", "axis declared twice"),
    ("axis,values\nBLOCK_M1,32 x\nnum_warps,4\nnum_stages,2\n", "non-integer value"),
])
def test_a_malformed_spec_is_rejected_at_read(tmp_path, bad, why):
    """These must fail where the file is read, naming it. A spec that silently loses an axis
    produces configs missing a constexpr, and Triton reports that far away as
    ``dynamic_func() missing 1 required positional argument``."""
    with pytest.raises(ValueError):
        _read(_write(tmp_path, bad, f"bad_{abs(hash(why))}.csv"))


def test_a_slice_selecting_nothing_is_an_error(tmp_path):
    """Silently empty means Triton substitutes its own Config({}) and the kernel dies at launch."""
    with pytest.raises(ValueError, match="selects nothing"):
        _read(_write(tmp_path, SPEC + "slice,900-999\n", "empty.csv"))

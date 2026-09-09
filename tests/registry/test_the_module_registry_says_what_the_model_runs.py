"""`registry_module.csv` is the source of the build's SHAPES, so it is the thing worth checking.

The defect class these tests exist for: the values a kernel is tuned at used to live in
hand-written tuples in `builder`, keyed off a column in `registry.csv` that only named the AXIS.
The two drifted -- token lengths stopped at 512 while the sweep ran to 1024, `MSA_WIDTHS` held 64
while the config declares 64 and 128 -- and every drift is a bucket production reaches with no
cache entry, found only by a replay on a card. Nothing checked the tuples because there was
nothing to check them against.

There is now: one file states the shapes, and these tests are what "점검하는 코드" means for it.
They check the file against its own stated vocabulary and against the modules it names -- both
cheap, both at import time, neither needing a GPU.
"""

from __future__ import annotations

import pytest

from miniworld_engine.autotune import derive
from miniworld_engine.autotune.builder import CASE_NAMES, SWITCH_SETTINGS

ROWS = derive.module_rows()


def test_the_file_is_not_empty():
    assert ROWS, "registry_module.csv has no rows; the build would have nothing to do"


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r.module}:{r.stream}:{r.dims}")
def test_every_row_names_a_module_the_builder_can_construct(row):
    """A row for a module with no case is a shape nobody can ever build.

    Silent before: the plan was built from the ladders, so a module named only here contributed
    nothing and said nothing.
    """
    assert row.module in CASE_NAMES, (
        f"registry_module.csv row {row.module!r} is not in CASE_NAMES, so no factory builds it")


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r.module}:{r.stream}:{r.dims}")
def test_every_row_uses_a_declared_stream(row):
    assert row.stream in derive.STREAM_LADDERS, (
        f"{row.module}: stream {row.stream!r} has no length ladder; "
        f"known streams are {sorted(derive.STREAM_LADDERS)}")


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r.module}:{r.stream}:{r.dims}")
def test_a_rows_lengths_are_its_streams_lengths(row):
    """The stream name has to MEAN its ladder, or it is decoration.

    This is the exact shape of the bug that produced the atom-side holes: `conditioned_transition`
    at d_hidden=128 is the atom DiT, so its lengths are atom counts (1024..8192) and not the token
    counts every other row carries. When the ladder lived in Python and the name lived in the
    viewer, nothing connected them and the atom rows were swept at token lengths forever.
    """
    ladder = derive.STREAM_LADDERS[row.stream]
    unknown = sorted(set(row.lengths) - set(ladder))
    assert not unknown, (
        f"{row.module} declares stream {row.stream} but runs at {unknown}, "
        f"which is not in that stream's ladder {ladder}")


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r.module}:{r.stream}:{r.dims}")
def test_every_option_can_actually_be_pinned(row):
    """An option nothing knows how to apply is a unit that silently duplicates the default.

    `p_drop` is the one option that is a constructor argument rather than a settings pin; every
    other one has to appear in `SWITCH_SETTINGS` or `derive` cannot set it.
    """
    for name, _value in row.options:
        assert name == "p_drop" or name in SWITCH_SETTINGS, (
            f"{row.module}: option {name!r} is neither p_drop nor a pinnable switch")


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r.module}:{r.stream}:{r.dims}")
def test_every_row_declares_what_it_varies(row):
    assert row.lengths, f"{row.module}: no lengths"
    assert row.dims, f"{row.module}: no dims"
    assert row.impls, f"{row.module}: no impls"
    assert row.dtypes, f"{row.module}: no dtypes"
    assert set(row.modes) <= {"eval", "train"}, f"{row.module}: bad modes {row.modes}"
    assert row.source, (
        f"{row.module}: no source. Every number here comes from somewhere -- a MiniWorld config "
        f"key, an AF3 constant, or an explicit headroom decision -- and saying which is what "
        f"keeps the next person from re-deriving it wrong.")


def test_no_two_rows_declare_the_same_shape_twice():
    """Two rows with the same (module, stream, dims) build the same units twice."""
    seen: dict = {}
    for row in ROWS:
        key = (row.module, row.stream, tuple(sorted(row.dims.items())))
        assert key not in seen, f"duplicate row for {key}"
        seen[key] = row


def test_the_streams_partition_by_length_scale():
    """Atom streams and token streams must not overlap, or the vocabulary carries no information.

    An atom count and a token count are different quantities -- MiniWorld buckets tokens to
    multiples of 128 and atoms to multiples of 1024 -- and a kernel keyed on rows sees them as
    different buckets. A ladder that served both would be the old undifferentiated `lengths`.
    """
    token = set(derive.STREAM_LADDERS["token_pair"]) | set(derive.STREAM_LADDERS["token_single"])
    atom = set(derive.STREAM_LADDERS["atom_single"])
    assert not (token & atom), f"token and atom ladders overlap at {sorted(token & atom)}"

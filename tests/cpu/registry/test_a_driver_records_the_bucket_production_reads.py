"""A driver must record into the bucket production will look up, not a bucket only it can produce.

An OUTER entry point takes its autotune key from the activation's pre-flatten shape --
`atom_key(length_of(x.shape))`, and `length_of` is `shape[-2]`. So the shape a driver hands it is
not cosmetic: it decides which bucket the measurement lands in.

`adaln_fwd`'s driver handed `(1, _M, D)`. That says "one sequence of _M". Since `76daae51` made
`_M = max(_L, 8192)` -- the ROWS to tune at, deliberately larger than the length -- every unit of
`adaln_fwd_triton`, at every L the sweep drives, recorded the single bucket `atom_key(8192)` and
overwrote the one before it, while the buckets production looks up (128 through 4096) stayed empty.
`dev audit` on the A6000 reported it as 16 missing (dtype, bucket) pairs, the largest hole on the
card, and no amount of rebuilding would have closed it.

`(M // L, L, D)` gives the entry point the length it keys on AND the row count the tuning needs --
and is what production hands it anyway: a batch of sequences, not one sequence of 8,192.
"""
from __future__ import annotations

from miniworld_engine.autotune.shape_key import atom_key, length_of
from miniworld_engine.kernels.drivers import conditioned_transition as CT


def _batched_shape(m: int, nx: int) -> tuple[int, ...]:
    """The shape `_adaln_args(batched=True)` builds, without allocating on a GPU."""
    length = min(CT._L, m)
    return (max(1, m // length), length, nx)


def test_the_outer_entry_point_lands_in_the_bucket_the_inner_drivers_use() -> None:
    shape = _batched_shape(CT._M, CT._D)
    assert atom_key(length_of(shape)) == CT._SHAPE_KEY, (
        f"the batched activation {shape} keys on atom_key({length_of(shape)}), while every inner "
        f"driver of the same family records {CT._SHAPE_KEY}. The two must be the same bucket or "
        f"the outer op's measurements land where nothing reads them.")


def test_the_row_count_is_not_traded_away_for_the_bucket() -> None:
    """The rows are what makes the measurement right: `_ROWS_SATURATE` exists because tuning at
    512 rows picked a config that costs production 1.53x. Fixing the bucket must not undo that."""
    shape = _batched_shape(CT._M, CT._D)
    assert shape[0] * shape[1] == CT._M, (
        f"{shape} is {shape[0] * shape[1]} rows, not the {CT._M} the driver tunes at")


def test_the_shape_is_the_one_production_hands_the_entry_point() -> None:
    """(B, L, D) -- a batch of sequences. `(1, M, D)` is a single sequence of M, which production
    never produces and which is what made the bucket wrong."""
    shape = _batched_shape(CT._M, CT._D)
    assert len(shape) == 3
    assert shape[-1] == CT._D
    assert shape[1] == CT._L, "shape[-2] must be the LENGTH; it is what the key is taken from"


def test_a_one_sequence_shape_would_still_be_caught() -> None:
    """The regression, stated directly: if the driver goes back to `(1, M, D)` this fails."""
    bad = (1, CT._M, CT._D)
    assert atom_key(length_of(bad)) != CT._SHAPE_KEY, (
        "either _M == _L again, in which case this test has nothing to guard, or the key stopped "
        "coming from shape[-2]")


def test_no_other_driver_hands_an_outer_entry_point_a_single_sequence() -> None:
    """The same defect elsewhere. `batched=True` is how a driver says 'this is an outer entry
    point'; every such call has to build the shape through `_adaln_args`, which now splits it."""
    import inspect

    from miniworld_engine.kernels.drivers import adaln

    src = inspect.getsource(adaln)
    assert "lead = (1,) if batched else ()" not in src, (
        "the single-sequence shape is back")
    assert src.count("batched=True") == src.count("_adaln_args(batched=True)"), (
        "a driver builds a batched activation by hand instead of through `_adaln_args`, so the "
        "split that fixes the bucket does not apply to it")

"""Autotune shape keys: one definition of "shape", one bucket set per kernel level.

WHAT SHAPE MEANS
----------------
``shape`` is L -- the number of TOKENS or ATOMS -- never the row count a kernel happens to receive.
A pair kernel iterating M = L*L rows and a linear kernel iterating M = L rows are at the same shape
when they are at the same L, and they should share a bucket space.

That was not previously true. Three edge sets were in use (squared on L*L, linear on L, and their
union) against two different base quantities (L at the attention families, M elsewhere, plus a flat
element count at two sites). Each call site was internally consistent with its own set, so nothing
failed -- but the same physical L landed in a different bucket space depending on the family, and a
``both``-level kernel bucketing raw M could not tell a linear M=384 from a pair M=384 that came from
L=20. The point of this module is that there is now one answer.

THE BUCKETS
-----------
Dimension (channel width) is an exact set: a kernel is built for one of these widths and no other,
so there is nothing to clamp.

    dim     64  128  256  384  512  768

Shape depends on where the model uses the kernel -- the ``level`` column of
``kernels/registry.csv``:

    token   128  256  384  512
    atom    256  512  1024  2048  4096  8192
    both    the union: 128 256 384 512 1024 2048 4096 8192

MAPPING
-------
Shape is mapped by FLOOR, with clamping at both ends:

    below the smallest bucket -> the smallest        (token 64  -> 128)
    above the largest         -> the largest         (atom 99999 -> 8192)
    anything between          -> the largest bucket <= it        (token 192 -> 128)

Floor rather than nearest or ceiling: a config tuned at 128 is being asked to run a 192-wide
problem, which it can (it tiles more), whereas a config tuned at 256 asked to run 192 was tuned for
a tile count it will not see. Rounding down keeps the tuned config a lower bound on the work.

The bucket VALUE is returned, not an index: it is what lands in the cache key and in axes.csv, and
`384` says what it means where `2` does not.
"""

from __future__ import annotations

#: Channel width. Exact -- a kernel is compiled for one of these and no other.
DIM_BUCKETS: tuple[int, ...] = (64, 128, 256, 384, 512, 768)

#: Token/pair-level sequence lengths.
TOKEN_SHAPES: tuple[int, ...] = (128, 256, 384, 512)

#: Atom-level counts.
ATOM_SHAPES: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192)

#: Kernels used at both levels bucket against the union, so a call from either side lands where it
#: would have landed in its own set (a token 192 still floors to 128, an atom 300 still to 256).
BOTH_SHAPES: tuple[int, ...] = tuple(sorted(set(TOKEN_SHAPES) | set(ATOM_SHAPES)))

SHAPES_BY_LEVEL: dict[str, tuple[int, ...]] = {
    "token": TOKEN_SHAPES,
    "atom": ATOM_SHAPES,
    "both": BOTH_SHAPES,
}


def _floor_clamp(value: int, buckets: tuple[int, ...]) -> int:
    """Largest bucket <= value, clamped to the ends. ``buckets`` must be sorted ascending."""
    if value <= buckets[0]:
        return buckets[0]
    if value >= buckets[-1]:
        return buckets[-1]
    lo = buckets[0]
    for b in buckets:
        if b > value:
            break
        lo = b
    return lo


def shape_bucket(length: int, level: str) -> int:
    """Bucket a token/atom count L for a kernel at ``level`` ("token" / "atom" / "both").

    Prefer the three named wrappers below at a call site: they cannot be passed the wrong level by
    a typo, and they say which level the kernel is at where a string argument does not.
    """
    try:
        buckets = SHAPES_BY_LEVEL[level]
    except KeyError:
        raise ValueError(
            f"level must be one of {sorted(SHAPES_BY_LEVEL)}, got {level!r}. It comes from the "
            f"`level` column of kernels/registry.csv and cannot be inferred from the kernel."
        ) from None
    return _floor_clamp(int(length), buckets)


def dim_bucket(d: int) -> int:
    """Bucket a channel width.

    Unlike the shape axis this does NOT clamp or floor: the declared widths are the widths kernels
    are built for, so a value outside the set means the caller is running a width nobody tuned, and
    silently folding it into a neighbour would hide that. Raise instead.
    """
    d = int(d)
    if d not in DIM_BUCKETS:
        raise ValueError(
            f"dimension {d} is not one of {DIM_BUCKETS}. Widths are exact: a config tuned at a "
            f"different width was tuned for a different register and smem footprint. Add the width "
            f"to DIM_BUCKETS and re-tune, or dispatch this call elsewhere."
        )
    return d


def pair_length(rows: int) -> int:
    """L from a pair kernel's flattened row count M = L*L, for call sites that only have M."""
    import math

    L = math.isqrt(int(rows))
    if L * L != int(rows):
        raise ValueError(
            f"row count {rows} is not a perfect square, so it did not come from an L*L pair; pass "
            f"L directly instead of asking this helper to recover it."
        )
    return L


def token_key(length: int) -> int:
    """Shape key for a token/pair-level kernel (`level=token` in registry.csv)."""
    return _floor_clamp(int(length), TOKEN_SHAPES)


def atom_key(length: int) -> int:
    """Shape key for an atom-level kernel (`level=atom`)."""
    return _floor_clamp(int(length), ATOM_SHAPES)


def both_key(length: int) -> int:
    """Shape key for a kernel used at both levels (`level=both`)."""
    return _floor_clamp(int(length), BOTH_SHAPES)


def length_of(shape) -> int:
    """L from the shape of an activation, BEFORE it is flattened for the launch.

    One rule covers both layouts, because the channel axis is always last:

        pair   (B, L, L, D) -> shape[-2] == L
        token  (B, L, D)    -> shape[-2] == L
        atom   (B, A, D)    -> shape[-2] == A

    So `L = shape[-2]`, with no square-root and no branch on which layout it is. That is the whole
    reason to read it here rather than at the kernel: once the tensor is `(M, D)` the information is
    gone -- `M` alone cannot say whether it is L or L*L, which is exactly why the old scheme had to
    bucket three different quantities against three different edge sets and still could not tell a
    linear M=384 from a pair M=384.

    An inner launcher that only receives the flattened `(M, D)` matrix therefore CANNOT call this;
    its caller must compute the key and pass it down.
    """
    dims = tuple(shape)
    if len(dims) < 2:
        raise ValueError(
            f"shape {dims} has no channel axis to drop; pass the activation's shape before it is "
            f"flattened, or compute the key at the caller that still has it."
        )
    return int(dims[-2])

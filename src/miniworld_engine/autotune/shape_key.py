"""Autotune shape keys: one definition of "shape", one bucket set per kernel level.

WHAT A SHAPE KEY IS
-------------------
``shape_key`` is the ONE parameter a kernel takes purely to be keyed on -- never read in a body
(a404fb9) -- and it carries the whole shape:

    shape_key = pack(<row-or-length bucket>, <every width axis>, <axis-name checksum>)

Both halves are shape. ``K``, ``ND``, ``H``, ``HEAD_DIM`` are dimensions of the tensors the kernel
reads, exactly as the row count is, so they belong inside the key rather than standing beside it in
``key=[...]``. The body still takes them as ``tl.constexpr`` parameters, because it reads them for
masks and loop bounds; what changed is that the autotune key names one thing. See :func:`pack`.

WHICH AXIS THE BUCKET IS, AND WHY IT DEPENDS ON THE LEVEL
---------------------------------------------------------
The first component is the bucketed LENGTH for a ``token`` or ``atom`` kernel and the bucketed ROW
COUNT for a ``level=both`` one. That is not an inconsistency to tidy away -- it is the fix for a
measured 1.73x regression, and the ``BOTH_ROWS`` block below is the argument. In short: a both-level
kernel is launched from both sides, the call site cannot say which, and rows are the one quantity
that means the same thing either way.

A pair kernel iterating M = L*L rows and a linear kernel iterating M = L rows are at the same shape
when they are at the same L, and they share a bucket space. That was not previously true: three edge
sets were in use against two different base quantities, so the same physical L landed in a different
bucket space depending on the family. The point of this module is that there is now one answer.

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

import zlib as _zlib

#: Channel width. Exact -- a kernel is compiled for one of these and no other.
DIM_BUCKETS: tuple[int, ...] = (64, 128, 256, 384, 512, 768)

#: Token/pair-level sequence lengths.
TOKEN_SHAPES: tuple[int, ...] = (128, 256, 384, 512)

#: Atom-level counts. Starts at 1024: an atom activation is the whole molecule's atoms, thousands
#: of them (`max_atoms: 5000` in the model's data config -- that file is not in this repository, so
#: treat the number as context, NOT as a cap), and 256 or 512 is a crop small enough that nothing in
#: the sweep is sized for it. Anything below 1024 floor-clamps into 1024.
#:
#: 8192 STAYS, and the reasoning that would remove it is a trap worth naming. `_floor_clamp` clamps
#: at the TOP as well, so the largest rung is the bucket every larger atom count lands in -- the
#: whole unbounded tail, not just [8192, 16384). Reading "max_atoms: 5000" as a cap makes 8192 look
#: unreachable (5000 floors into 4096) and suggests dropping it for ~4% of the build; the moment the
#: cap moves, every launch above 4096 would then land in a bucket tuned for a quarter of its work.
#: The rung is cheap insurance on an axis whose ceiling is set outside this repository.
ATOM_SHAPES: tuple[int, ...] = (1024, 2048, 4096, 8192)

#: The DiT families -- adaln, conditioned_transition, augmented_attention, the seventeen rows that
#: are `level=atom` with `width=single` -- key on the atom ladder but run on BOTH streams: the model
#: builds one DiffusionTransformer block class 24 times at d_single=768/d_cond=384 (the token side)
#: and 3 times at 128/128 (the atom side). Driving them as ONE list got both halves wrong. It built
#: shapes the model never presents -- an atom count of 8192 at a token width, eight of the eighteen
#: units -- and it gave the token side no bucket of its own: `atom_key` starts at 256, so a token
#: count of 128 and one of 384 both floored into 256, and 24 of the model's 27 blocks are
#: token-side.
#:
#: So: two lists, per side, with the widths that side actually has. The ranges are disjoint ON
#: PURPOSE -- atom counts start at 1024, token counts stop at 768 -- and that is what lets one
#: floor-clamp key both sides. `level=both` needs `both_key`'s row-count indirection precisely
#: because a pair L and an atom A of the same value are different launches; here no length can
#: have come from either side, so length is unambiguous and stays the key.
#: 128 is here because `_floor_clamp` clamps UP at the bottom -- anything at or below the smallest
#: rung returns that rung. Without it a 128-token DiT launch keys to the 256 bucket and runs a
#: config tuned for twice the tile count, which is the one direction this module's own docstring
#: says never to round. 128 is a declared token count for every other token kernel (TOKEN_SHAPES),
#: so it is one here too.
DIT_TOKEN_LENGTHS: tuple[int, ...] = (128, 256, 384, 512, 768)
DIT_ATOM_LENGTHS: tuple[int, ...] = (1024, 2048, 4096, 8192)

#: What `atom_key` floor-clamps into: the atom work list plus the token lengths above. Widening the
#: KEY set costs no units. What a build DRIVES is the work list, and no atom-only kernel is driven
#: at 384 or 768; the extra rungs exist so a token-side launch lands in a bucket of its own rather
#: than in one tuned for a different stream. Same distinction as BOTH_SHAPES against BOTH_ROWS.
ATOM_KEY_BUCKETS: tuple[int, ...] = tuple(sorted(set(ATOM_SHAPES) | set(DIT_TOKEN_LENGTHS)))
assert set(DIT_ATOM_LENGTHS) <= set(ATOM_SHAPES), "the atom work list is a slice of ATOM_SHAPES"
assert not (set(DIT_ATOM_LENGTHS) & set(DIT_TOKEN_LENGTHS)), (
    "the two sides must not share a length, or one floor-clamp cannot tell them apart")

#: Kernels used at both levels bucket against the union, so a call from either side lands where it
#: would have landed in its own set (a token 192 still floors to 128, an atom 300 still to 256).
#: This is the WORK-LIST axis -- which shapes a build drives -- not the cache key; see BOTH_ROWS.
BOTH_SHAPES: tuple[int, ...] = tuple(sorted(set(TOKEN_SHAPES) | set(ATOM_SHAPES)))

#: The cache key for a ``level=both`` kernel is its ROW COUNT, not its length. Keying on length
#: collides: a pair activation (B, L, L, D) at L=1024 and an atom activation (B, A, D) at A=1024
#: both have ``shape[-2] == 1024``, so both landed in bucket 1024 -- while the first launches
#: 1,048,576 rows and the second launches 1,024. Measured, in the shipped A6000 cache for
#: transition_layernorm_expand_swiglu_triton:
#:
#:     shape_key=384    0.6103 ms     pair  L=384,  M = 147,456
#:     shape_key=1024   0.0215 ms     atom  A=1024, M =   1,024
#:
#: The bucket grows and the time falls 28x, because the two values are not on the same axis. The
#: module bench then ran the pair side at L=1024 against a config tuned at 1,024 rows and lost
#: 1.73x against its own baseline (5.50 ms -> 9.50 ms, same card, same configuration).
#:
#: Rows are the right axis for these kernels and length is not: a pair launch of M rows and an
#: atom launch of M rows have the same launch geometry, so the same tile is right for both, and
#: nothing about "which side it came from" changes that. (Length is still right for ``token`` and
#: ``atom`` kernels, where it is unambiguous.)
#: The pair lengths a both-level kernel is tuned at. It used to carry 1024 as well, on the argument
#: that the module benches sweep there and production pairformer runs there, so tuning to 512 and
#: extrapolating was "close enough" of the kind this bucket set exists to stop.
#:
#: MEASURED, and the argument does not hold. Two both-level GEMMs swept at all five pair lengths
#: (bf16, three widths, full grid): taking the L=512 winner and running it at L=1024 costs
#: 1.003x-1.036x, and taking the L=1024 winner back to L=512 costs 1.000x-1.018x. Run-to-run noise
#: on the same config measured twice is 1.012x median and 1.059x at p99, so those numbers are not
#: distinguishable from measuring the same thing again. Of 52 length pairs compared, three exceed
#: the noise floor and all three are L=128 against a larger length -- 128 is the only pair bucket
#: that is really its own.
#:
#: Why it saturates: the two things that make a config depend on M both stop mattering above L=128
#: on this card. The activation stops fitting in L2 between 128 and 256 (4 MB against 6), and past
#: that it never fits again however much larger it gets; and wave quantization is under the noise
#: floor by L=256 (195 waves over 84 SMs, a tail bounded by 0.5%). Both thresholds MOVE WITH THE
#: CARD -- a B200's 126 MB L2 holds the activation until L~718, so on that card the boundary sits
#: between 512 and 1024 instead. This ladder is card-independent, so dropping 1024 rests on the
#: measurement above being about saturation rather than about A6000: anything past the largest
#: bucket floor-clamps into it, which is the mechanism that makes a missing rung a clamp and not
#: a miss.
BOTH_PAIR_LENGTHS: tuple[int, ...] = TOKEN_SHAPES

#: Rows, from every side a `level=both` kernel is driven from. The token side was missing: 128 and
#: 384 rows had no bucket and floored into someone else's, which is the same hole the DiT families
#: had on `atom_key` before they were split.
BOTH_ROWS: tuple[int, ...] = tuple(sorted(
    set(ATOM_SHAPES) | set(TOKEN_SHAPES) | {length * length for length in BOTH_PAIR_LENGTHS}))

#: The LENGTH axis per level. `both` is here for completeness, but nothing keys on it any more:
#: a both-level kernel is driven as two sided lists (`BOTH_PAIR_LENGTHS` + `ATOM_SHAPES`) and
#: keyed on `BOTH_ROWS`, because one length means two different launches for it.
SHAPES_BY_LEVEL: dict[str, tuple[int, ...]] = {
    "token": TOKEN_SHAPES,
    "atom": ATOM_SHAPES,
    "both": BOTH_SHAPES,
}


#: Radix the packer uses per axis. :func:`pack` refuses any axis at or above it, because a width
#: that large would carry into its neighbour's digit and two different shapes would share one key.
#:
#: The ceiling this has to clear is not a fixed number: `driver_width` makes the base width a knob,
#: and a family's widest derived axis scales with it. `ND2 = 8 * base` in conditioned_transition is
#: 1,024 at base 128 and 3,072 at 384, but exactly 4,096 at 512 and 6,144 at 768 -- both of which
#: the single-side ladder drives.
#:
#: Raising the radix does not fix that, it moves the wall: `shape_key` is a RUNTIME scalar argument
#: to the kernel, so the assembled value must stay an int64, and the budget is
#: bits(base) + log2(radix) * (axes + 1). At radix 8192 a three-axis fold on `both_key`'s top row
#: bucket is 72 bits. So the radix stays, and a launch whose axis does not fit RAISES -- see `pack`.
#: The builder turns that into a skipped unit with a printed reason, which is the honest outcome:
#: that width is outside what this packing can key, and a silent wrap would be a collision.
_RADIX = 4096


class ShapeKeyTooWide(ValueError):
    """An axis (or the assembled key) does not fit the packing.

    Its own type because it is a PERMANENT fact about a (kernel, width), exactly like OOM or
    OutOfResources: retrying the unit cannot make the axis smaller. The builder's skip path reads
    the type to decide whether a resumed run should claim it again.
    """


def pack(base: int, **axes: int) -> int:
    """One cache label carrying the WHOLE shape: the row/length bucket plus every width axis.

    ``K``, ``ND``, ``H``, ``HEAD_DIM`` and the rest are shape -- they are dimensions of the tensors
    the kernel reads, exactly as the row count is -- so they belong in the shape key rather than
    standing beside it as separate ``key=[...]`` entries. The body still takes them as
    ``tl.constexpr`` parameters (it reads them for masks and loop bounds); what changes is that the
    autotune key names ONE thing, and that thing is the shape.

    Order-independent by construction: axes are packed by SORTED NAME, so a launcher that writes
    ``atom_key(L, HEAD_DIM=d, H=h)`` and one that writes ``atom_key(L, H=h, HEAD_DIM=d)`` produce
    the same key. Ordering by argument position is how ``3d47a78`` and ``7c16d16`` both went wrong
    -- one site disagreeing with another about what a positional argument meant -- and naming the
    axes removes that failure mode instead of documenting it.

    The axis NAMES are folded in too. A launcher that forgets an axis, or names a different set
    than the one that wrote the cache, then produces a DIFFERENT key rather than a colliding one:
    the lookup misses, the reader warns and falls back to the full grid, and nothing silently runs
    a config tuned for another shape. That is the direction to be wrong in -- ``6948c77`` cost
    1.73x precisely because a lossy key collided instead of missing.
    """
    if not axes:
        return int(base)
    value = int(base)
    for name in sorted(axes):
        w = int(axes[name])
        if not 0 < w < _RADIX:
            raise ShapeKeyTooWide(
                f"shape axis {name}={w} is outside (0, {_RADIX}); packing it would carry into the "
                f"next axis's digit and two different shapes would share one autotune key. Raise "
                f"_RADIX and re-tune, or check that {name} is really a width."
            )
        value = value * _RADIX + w
    value = value * _RADIX + (_zlib.crc32(",".join(sorted(axes)).encode()) & (_RADIX - 1))
    # The per-axis check above is not the whole bound. `shape_key` reaches the kernel as a RUNTIME
    # scalar argument, so the assembled value has to stay an int64: the budget is
    # bits(base) + 12 * (axes + 1), and `both_key`'s top bucket (1,048,576 rows) leaves room for
    # two axes at 57 bits and overflows at three (69). Nothing folds three axes onto a row bucket
    # today -- the three-axis folds are all atom-keyed, at 61 -- but "today" is one driver width
    # away from being wrong, and an int64 that wraps is a COLLISION, which is the one failure this
    # function exists to make impossible.
    if value.bit_length() > 63:
        raise ShapeKeyTooWide(
            f"shape key {value} needs {value.bit_length()} bits: base {base} with {len(axes)} axis "
            f"/axes {sorted(axes)} does not fit an int64, and `shape_key` is a runtime kernel "
            f"argument. Fold fewer axes (a derived one is implied by its base) or narrow _RADIX."
        )
    return value


def unpack_base(value: int, n_axes: int) -> int:
    """The row/length bucket inside a key :func:`pack` built, given how many axes it folded.

    A coverage check has the two halves in different forms: the CACHE holds the packed key a launch
    recorded, while a declared work list holds the bare bucket a unit will drive. Comparing them
    directly reports every folded op as missing -- which is the whole cache, once every op folds.

    ``n_axes`` is not guessable from the value: the same integer is a different base for a different
    axis count. Read it from the kernel, which is what ``tools.key_gaps`` already resolves.
    """
    if n_axes <= 0:
        return int(value)
    return int(value) // (_RADIX ** (n_axes + 1))   # + 1 for the axis-name checksum digit


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
    if int(rows) != L * L:
        raise ValueError(
            f"row count {rows} is not a perfect square, so it did not come from an L*L pair; pass "
            f"L directly instead of asking this helper to recover it."
        )
    return L


def token_key(length: int, **axes: int) -> int:
    """Shape key for a token/pair-level kernel (`level=token` in registry.csv).

    ``**axes`` are the kernel's width dimensions (``K``, ``ND``, ``H``, ...). They are shape, so
    they belong in the key rather than beside it; see :func:`pack`."""
    return pack(_floor_clamp(int(length), TOKEN_SHAPES), **axes)


def atom_key(length: int, **axes: int) -> int:
    """Shape key for an atom-level kernel (`level=atom`). ``**axes``: see :func:`pack`.

    Buckets are :data:`ATOM_KEY_BUCKETS`, not :data:`ATOM_SHAPES`: the three DiT families on this
    key run token-side too, and 384 and 768 are token counts that need their own bucket.
    """
    return pack(_floor_clamp(int(length), ATOM_KEY_BUCKETS), **axes)


def both_key(rows: int, **axes: int) -> int:
    """Cache key for a kernel used at both levels (`level=both`), from its ROW COUNT.

    Rows, not length -- see :data:`BOTH_ROWS` for why, and for the 1.73x this cost. Call it as
    ``both_key(rows_of(x.shape))``; a call site that already has the flattened M passes that.
    """
    return pack(_floor_clamp(int(rows), BOTH_ROWS), **axes)


def rows_of(shape) -> int:
    """M -- the row count a launch iterates -- from an activation's PRE-flatten shape.

    Every leading axis multiplied out: pair (B, L, L, D) -> B*L*L, token/atom (B, L, D) -> B*L.
    This is what :func:`both_key` buckets, and it is the one quantity that does not depend on
    knowing which side of a ``level=both`` kernel the call came from.

    Refuses a 2-D shape for the same reason :func:`length_of` does -- not because M is unreadable
    there (it is exactly ``shape[0]``) but because a caller holding only ``(M, D)`` cannot know
    whether its M is the whole launch or one slice of it, and every call site in this repo that
    passed a pre-flattened matrix was doing so by mistake.
    """
    dims = tuple(shape)
    if len(dims) < 3:
        raise ValueError(
            f"shape {dims} is already flattened; pass the activation's shape BEFORE it is "
            f"flattened -- (B, L, D) or (B, L, L, D) -- or compute the key at the caller that "
            f"still has it and pass `shape_key=` down. See the note in rows_of's docstring.")
    out = 1
    for d in dims[:-1]:
        out *= int(d)
    return out


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
    if len(dims) < 3:
        # A PRE-flatten activation is (B, L, D) or (B, L, L, D) -- always at least 3-D. A 2-D
        # (M, D) is one that has ALREADY been flattened, and then shape[-2] is M, not L. Returning
        # it silently is how this went wrong everywhere: for a pair kernel M = L*L, which
        # both_key clamps to the top bucket at any L >= 91, so every sequence length shared one
        # cached config and nothing failed. Refusing here is the only place that catches it, since
        # the value that comes back is a perfectly plausible integer.
        #
        # Measured before this was tightened: over a real module run, 25 of 25 call sites passed
        # >= 3-D once the one genuine offender was fixed -- so nothing legitimate is being
        # rejected. Over the driver suite, 35 of 38 passed 2-D and every one was a bug.
        raise ValueError(
            f"shape {dims} is already flattened, so shape[-2] is M and not L. Pass the "
            f"activation's shape BEFORE it is flattened -- (B, L, D) or (B, L, L, D) -- or "
            f"compute the key at the caller that still has it and pass `shape_key=` down. See "
            f"the note in length_of's docstring."
        )
    return int(dims[-2])

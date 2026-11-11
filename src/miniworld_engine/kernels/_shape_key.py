"""Shape keys: the one int a configured kernel takes purely to be keyed on (never read in a body).

    shape_key = pack(<row-or-length bucket>, **<width axes>)

The bucket (ported unchanged from the old ``autotune/shape_key.py``):

* ``token_key(L)`` / ``atom_key(L)``: a ``token`` / ``atom`` level kernel buckets its LENGTH L (``length_of`` = ``shape[-2]``
  of the pre-flatten activation).
* ``both_key(M)``: a kernel launched from both levels buckets its ROW COUNT M (``rows_of`` = every leading axis multiplied
  out), never its length: a pair L=1024 and an atom A=1024 have the same ``shape[-2]`` but 1,048,576 vs 1,024 rows, and keying
  them together cost a measured 1.73x (the pair side ran a config tuned at 1,024 rows).
* Mapping is FLOOR with clamping at both ends: below the smallest rung -> the smallest, above the largest -> the largest,
  between -> the largest rung <= value. A config tuned at 128 asked to run 192 tiles more; one tuned at 256 would have been
  tuned for tiles it never sees.

Width axes (``K``, ``N``, ``HEAD_DIM``...) are exact. They are packed by SORTED NAME in base 4096 with a checksum of the names
as the last digit, so a launcher that names a different axis set gets a different key (a miss, never a collision). The tables
store the decoded form (``M=640,N=128``); ``_launch_config`` re-packs it with this same function.
"""

from __future__ import annotations

import zlib

import torch

TOKEN_SHAPES = (128, 256, 384, 512, 640, 768)
ATOM_SHAPES = (1024, 2048, 4096, 8192)
#: atom_key's rungs: the atom ladder plus the token lengths (the DiT families run at both; the two ranges are disjoint).
ATOM_KEY_BUCKETS = tuple(sorted(set(ATOM_SHAPES) | set(TOKEN_SHAPES)))
#: both_key's rungs: rows from every side -- token and atom counts plus the pair rows L*L.
BOTH_ROWS = tuple(sorted(set(ATOM_SHAPES) | set(TOKEN_SHAPES) | {n * n for n in TOKEN_SHAPES}))
#: the bucket names the tables use: M for a row bucket, L for a length bucket (descriptive; the lookup uses the value).
BASE_NAMES = ("M", "L")
_RADIX = 4096
#: set by kernels/_pretune.py once it has wrapped the key functions for recording (they are wrapped once per process)
_pretune_recording = False


class ShapeKeyTooWide(ValueError):
    """An axis (or the assembled key) does not fit the int64 packing: a permanent fact about a (kernel, width)."""


@torch.compiler.assume_constant_result
def _axis_name_tag(names: tuple[str, ...]) -> int:
    return zlib.crc32(",".join(names).encode()) & (_RADIX - 1)


def pack(base: int, **axes: int) -> int:
    """The row/length bucket plus every width axis, as one int64 (see the module docstring)."""
    if not axes:
        return int(base)
    value = int(base)
    for name in sorted(axes):
        w = int(axes[name])
        if not 0 < w < _RADIX:
            raise ShapeKeyTooWide(f"shape axis {name}={w} is outside (0, {_RADIX}); it would carry into the next axis's digit")
        value = value * _RADIX + w
    value = value * _RADIX + _axis_name_tag(tuple(sorted(axes)))
    if value.bit_length() > 63:
        raise ShapeKeyTooWide(f"shape key for base {base} and axes {sorted(axes)} needs {value.bit_length()} bits (> int64)")
    return value


def _floor_clamp(value: int, buckets: tuple[int, ...]) -> int:
    if value <= buckets[0]:
        return buckets[0]
    lo = buckets[0]
    for b in buckets:
        if b > value:
            break
        lo = b
    return lo


def token_key(length: int, **axes: int) -> int:
    return pack(_floor_clamp(int(length), TOKEN_SHAPES), **axes)


def atom_key(length: int, **axes: int) -> int:
    return pack(_floor_clamp(int(length), ATOM_KEY_BUCKETS), **axes)


def both_key(rows: int, **axes: int) -> int:
    return pack(_floor_clamp(int(rows), BOTH_ROWS), **axes)


def rows_of(shape) -> int:
    """M from the PRE-flatten shape: (B, L, L, D) -> B*L*L, (B, L, D) -> B*L. Refuses an already flattened 2-D shape."""
    dims = tuple(shape)
    if len(dims) < 3:
        raise ValueError(f"shape {dims} is already flattened; pass the activation's shape before it is flattened, "
                         f"or compute the key at the caller that still has it and pass `shape_key=` down")
    out = 1
    for d in dims[:-1]:
        out *= int(d)
    return out


def length_of(shape) -> int:
    """L = shape[-2] of the PRE-flatten activation (pair, token and atom layouts alike). Refuses a 2-D (M, D) shape."""
    dims = tuple(shape)
    if len(dims) < 3:
        raise ValueError(f"shape {dims} is already flattened, so shape[-2] is M and not L; pass the shape before it is "
                         f"flattened, or compute the key at the caller that still has it and pass `shape_key=` down")
    return int(dims[-2])

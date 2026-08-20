"""One function per kernel that launches it once, with inputs the repo states outright.

The point is to stop inferring. ``registry.csv`` says which kernels exist and which driver runs
each one; this module holds the drivers; ``autotune.run_all`` calls them and records what
happened. Nothing scans for capability asserts or guesses shapes -- a kernel runs or it raises,
and the exception text is the reason it did not.

A kernel with no driver yet has an empty ``driver`` cell and is reported ``untested``. That is a
visible hole to close, never a pass.

Drivers take no arguments, run on the current device, and raise on failure. Keep them small: the
job is to reach the kernel, not to check numerics -- accuracy lives in the bench.

Tile alignment
--------------
Every extent these helpers hand out by default is a multiple of 128 (``pair`` 128, ``single``
384, ``rows2d`` 512x384, ``drivers_ln._M`` 16384). The five config sets tile at 16/32/64/128, so
every extent divides every tile exactly and no kernel's boundary mask has ever been executed --
a missing or wrong ``mask=`` on a tail tile cannot be observed at these shapes.

``MINIWORLD_SHAPE_MODE=ragged`` subtracts a small amount from each extent that goes through
``ragged()``, putting a partial tile at the end of every axis. Set it before importing any
driver module; the constants are evaluated at import.

An extent a kernel genuinely cannot vary -- a fixed-shape CUDA GEMM, a power-of-two head dim,
a width baked into a weight layout -- goes through ``aligned_only(label, n, why)`` instead. That
keeps the extent aligned *and records why*, so the sweep reports it as an alignment requirement
rather than counting it as ragged coverage it never had.
"""

from __future__ import annotations

import os

import torch

BF16 = torch.bfloat16

#: "aligned" (default) or "ragged" -- see the module docstring.
SHAPE_MODE = os.environ.get("MINIWORLD_SHAPE_MODE", "aligned").strip().lower()
if SHAPE_MODE not in ("aligned", "ragged"):
    raise ValueError(f"MINIWORLD_SHAPE_MODE must be 'aligned' or 'ragged', got {SHAPE_MODE!r}")

#: label -> reason, for every extent a driver declared it cannot perturb.
ALIGNMENT_REQUIRED: dict[str, str] = {}


def ragged(n: int, *, by: int = 3, floor: int = 16) -> int:
    """``n`` when aligned, ``n - by`` when ragged, never below ``floor``.

    ``by`` defaults to 3: odd, smaller than the smallest tile (16), and not a divisor of any tile
    width, so the tail tile is partial for every one of the five config sets at once.
    """
    if SHAPE_MODE != "ragged":
        return n
    return max(floor, n - by)


def aligned_only(label: str, n: int, why: str) -> int:
    """Return ``n`` unchanged and record that this extent must stay tile-aligned.

    Use this instead of leaving a bare constant, so the ragged sweep can distinguish "this axis
    was proven to mask correctly" from "this axis was never perturbed".
    """
    ALIGNMENT_REQUIRED[label] = why
    return n


def dev() -> torch.device:
    return torch.device("cuda")


def pair(b: int = 1, n: int = 128, d: int = 128, dtype: torch.dtype = BF16) -> torch.Tensor:
    """A pair activation [B, N, N, D] -- the shape most of these kernels are written for."""
    return torch.randn(b, n, n, d, device=dev(), dtype=dtype)


def single(b: int = 1, n: int = 384, d: int = 384, dtype: torch.dtype = BF16) -> torch.Tensor:
    return torch.randn(b, n, d, device=dev(), dtype=dtype)


def rows2d(m: int = 512, n: int = 384, dtype: torch.dtype = BF16) -> torch.Tensor:
    return torch.randn(m, n, device=dev(), dtype=dtype)


def vec(n: int = 384, dtype: torch.dtype = BF16) -> torch.Tensor:
    return torch.randn(n, device=dev(), dtype=dtype)

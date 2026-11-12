"""64-bit index arithmetic for Triton kernels, paid only where a tensor can pass 2**31 elements.

Triton computes ``tl.program_id`` / ``tl.arange`` / loop counters and every integer argument below 2**31 in int32, and an int32 product
wraps BEFORE it reaches the pointer. A kernel whose per-program base is int64 (``program_id(..).to(tl.int64)``) is still exposed through a
separate in-tile term such as ``cols[None, :] * stride_col``: on a channel-major (m-major) view the channel stride is the row count
M = B * L * L, and ``K * M`` passes 2**31 at the pair sizes the engine serves (d 512 at L 2048, OpenDDE's refiner pair d 384 at
S 2944 / 3968). Casting that term on every call costs the hot loop (A100 TriMul output projection d 384: 4-5 %), so the launcher decides
with ``needs_wide`` and passes ``WIDE`` as a constexpr: below the bound the kernel compiles to the 32-bit code it always was.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

INT32_LIMIT = 2**31


@triton.jit
def wide(x, WIDE: tl.constexpr):
    """``x`` as int64 when the launcher set WIDE (an address product can pass 2**31), else unchanged."""
    return x.to(tl.int64) if WIDE else x  # WIDE is a constexpr: the untaken branch is never traced


def max_offset(t: torch.Tensor) -> int:
    """The largest element offset a kernel forms when it walks ``t`` at its own strides (sum of (size - 1) * |stride|)."""
    return sum((s - 1) * abs(st) for s, st in zip(t.shape, t.stride(), strict=True) if s > 0)


def needs_wide(*items: torch.Tensor | int | None) -> bool:
    """True when any tensor's largest offset (or any int extent product given directly) reaches 2**31."""
    for it in items:
        if it is None:
            continue
        n = max_offset(it) if isinstance(it, torch.Tensor) else int(it)
        if n >= INT32_LIMIT:
            return True
    return False


@triton.jit
def row_tile(N_CTX, BLOCK: tl.constexpr):
    """``(row, tile)`` of a 1-D grid laid out row-major over ``rows x cdiv(N_CTX, BLOCK)`` tiles (see ``row_tile_grid``); ``row`` is int64.

    CUDA caps grid axes y and z at 65535, so a per-row axis there (attention over B * H * L rows: H 16 at L 4096 already passes it) fails
    to launch; axis x takes 2**31 - 1. The tile index stays the fastest-varying one, as it was on axis x, so every row's tiles still run
    back to back."""
    n_tiles = tl.cdiv(N_CTX, BLOCK)
    pid = tl.program_id(0)
    return (pid // n_tiles).to(tl.int64), pid % n_tiles


def row_tile_grid(rows: int, n_ctx: int, block: int) -> list[int]:
    """The 1-D grid ``row_tile`` decodes: ``rows * cdiv(n_ctx, block)`` programs on axis x."""
    return [rows * ((n_ctx + block - 1) // block)]

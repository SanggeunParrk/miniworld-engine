"""The order a GEMM kernel visits its output tiles in, as one tuned axis.

A matmul kernel splits its output into a grid of tiles and gives one tile to each program. A tile
needs one row strip of the activation and one column strip of the weights, so two programs share
memory whenever they share a row or a column -- and whether that sharing turns into a cache hit is
decided by the ORDER the programs run in.

``GROUP_M`` is that order, as one number:

    GROUP_M = 1        walk the columns first -- one activation strip serves every column tile
    GROUP_M >= n_m     walk the rows first    -- one weight strip serves every row tile
    in between         blocks of GROUP_M rows, balancing the two

A 2-D grid ``(n_m, n_n)`` cannot express this: CUDA varies axis 0 fastest, so a kernel reading
``pid_m = program_id(0)`` is pinned at the second line above, whatever suits the shape. Measured on
an A6000 for conditioned_transition's expand GEMM at M=32768, d=768: the 2-D grid and GROUP_M=512
(= n_m there) are the same 4.675 ms to three decimals, and GROUP_M=8 is 3.569.

Which value wins is a property of the SHAPE AND THE CARD, which is why it belongs in the config
CSV rather than in a kernel. The benefit comes from the big operand not fitting in L2, so it tracks
its size against L2: measured across M, within noise up to activation/L2 = 2, appearing at 4 and
flat past it. An A6000 holds 6 MB and a B200 126 MB, so the same kernel at the same shape wants
different orders on different cards. Every ladder therefore carries both ends -- 1, and a rung
`tile_order` clamps to n_m -- so whatever a kernel did before the axis existed stays reachable and
a card that gains nothing loses nothing.

Tuned against tuned, with both arms free to choose tile, warps and stages, the axis was worth 1.10x
on that kernel. A fixed tile makes it look like 1.33x: the warp/stage pair that suits a row-first
walk is not the one that suits column-first, so holding them fixed hands the old order a bad
config. Judge a new conversion the same way -- against a tuned baseline, not a fixed one.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def tile_order(pid, n_m, n_n, GROUP_M: tl.constexpr):
    """(row tile, column tile) for program ``pid`` over an ``n_m`` x ``n_n`` grid.

    ``pid`` comes from a 1-D grid of ``n_m * n_n`` programs -- the launcher multiplies where it
    used to pass a tuple, and the kernel derives both indices here.
    """
    # Clamp to n_m so a ladder's top rung ALWAYS means column-first. Without it the guarantee is
    # arithmetic luck: a pair kernel has M = L*L, so at BLOCK_M1=16 and L=1536 n_m is 147,456 and
    # a rung of 65536 would group rather than reproduce the old schedule -- unreachable at exactly
    # the largest shapes. Clamped, any rung >= n_m collapses to first_m=0, size_m=n_m, which is
    # `pid_m = pid % n_m`: the 2-D grid, exactly.
    g = min(GROUP_M, n_m)
    per_group = g * n_n
    first_m = (pid // per_group) * g
    size_m = min(n_m - first_m, g)
    pid_m = first_m + ((pid % per_group) % size_m)
    pid_n = (pid % per_group) // size_m
    return pid_m, pid_n


def tile_grid(m, n, block_m, block_n):
    """The 1-D grid ``tile_order`` expects, from the two extents and their block sizes."""
    return (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

"""The A100 PWA kernels divide a row index by a runtime constant with a multiply-high (``hm_magic`` / ``tile_ij`` in kernels/pair_weighted_averaging/cuda/sm80/sm80_common.cuh):
``floor(n * m / 2^32)`` with ``m = floor((2^32 - 1) / d) + 1`` must equal ``n // d`` for every operand the kernels can pass (a CPU mirror of the device arithmetic)."""

import pytest


def _magic(d: int) -> int:
    return (0xFFFFFFFF // d + 1) & 0xFFFFFFFF  # uint32, as on the device


@pytest.mark.parametrize(
    "d",
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        12,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        129,
        192,
        256,
        384,
        512,
        1000,
        1024,
        2048,
        4096,
        8192,
    ],
)
def test_multiply_high_division_is_exact(d):
    if d == 1:
        pytest.skip(
            "d = 1 would need 2^32 as the multiplier: tile_ij handles ntile_s = 1 apart (i = tl), and kp = 1 only with S = 1 where s0 = 0"
        )
    m = _magic(d)
    # tile indices tl < 2^16 (L <= 1024 rows x ntile_s <= 64), MSA rows s0 < 2^14: the kernels' whole operand range
    for n in range(1 << 16):
        assert (n * m) >> 32 == n // d, (n, d)


def test_tile_index_split_matches_the_division():
    """tl -> (i, s0) as ``tile_ij`` computes it, against the plain division, for every ntile_s a registry depth can give (S 128 .. 8192, tile of 64 or 128 rows)."""
    for ntok in (64, 128):
        for depth in range(ntok, 8192 + 1, ntok):
            ntile_s = (depth + ntok - 1) // ntok
            m = _magic(ntile_s)
            for tl in range(0, 1024 * ntile_s, 7):
                i = (
                    tl if ntile_s == 1 else (tl * m) >> 32
                )  # tile_ij: one tile column is the identity
                assert (i, (tl - i * ntile_s) * ntok) == (
                    tl // ntile_s,
                    (tl % ntile_s) * ntok,
                )

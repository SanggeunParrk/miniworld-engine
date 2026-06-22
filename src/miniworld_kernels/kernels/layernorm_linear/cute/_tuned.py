"""Baked autotune result for LayerNormLinear (see ``tune.py`` / ``benchmark/tuned_lnl.*``).

A per-(M, d) lookup of the fastest *correct* GEMM config found by the sweep, used by the
``layernorm_linear`` dispatcher. Shapes not in the table fall back to the path's default
config (``None``), which is still correct — this only *speeds up* the benched shapes.

SAFE subset only (per the tuning decision): every M1 entry is **pingpong** with
**cluster_m=1**. The sweep found two config families that are fast but return wrong
results in a timing-dependent way — ``cluster_m=2`` and the non-pingpong *coop* path both
drop to cos ~0.96–0.999 at d=128 — so neither is baked here. Every M2 entry is bit-clean
(cos=0.999997); the racy ``tile_n>=160`` configs only ever won at d>=384, where the
dispatcher uses M1 anyway, so they never enter this table.
"""

from __future__ import annotations

from quack.gemm_config import GemmConfig


def _m1(tile_m: int, tile_n: int) -> GemmConfig:
    # all tuned M1 winners are pingpong + cluster (1,2), static persistent — see module docstring.
    return GemmConfig(
        tile_m=tile_m, tile_n=tile_n, pingpong=True, is_dynamic_persistent=False,
        cluster_m=1, cluster_n=2, swap_ab=False, max_swizzle_size=8, device_capacity=9,
    )


# (M, d) -> M1 GemmConfig. d=128 cells use the best *pingpong* config (the cl(2,1)/coop
# winners were faster but flaky); d>=256 winners were already pingpong cl(1,2).
_M1_TABLE = {
    (16384, 128): _m1(128, 128), (65536, 128): _m1(128, 128), (262144, 128): _m1(192, 128),
    (16384, 256): _m1(128, 128), (65536, 256): _m1(128, 128), (262144, 256): _m1(192, 128),
    (16384, 384): _m1(128, 192), (65536, 384): _m1(128, 192), (262144, 384): _m1(128, 192),
    # d=512: 192x128 was fastest but only 0.99991 on independent inputs (input-sensitive);
    # 128x128 is bit-clean (0.999997) everywhere, ~8% slower at the largest M. Safe pick.
    (16384, 512): _m1(128, 128), (65536, 512): _m1(128, 128), (262144, 512): _m1(128, 128),
    (16384, 768): _m1(128, 192), (65536, 768): _m1(128, 192), (262144, 768): _m1(128, 192),
}


def _m2(tile_m: int, tile_n: int) -> dict:
    return dict(tile_m=tile_m, tile_n=tile_n, cluster_m=1, cluster_n=1, pingpong=True)


# (M, d) -> M2 config dict. Only the shipping regime (d<=256, where M2 wins); all bit-clean.
_M2_TABLE = {
    (16384, 128): _m2(64, 128), (65536, 128): _m2(128, 128), (262144, 128): _m2(128, 128),
    (16384, 256): _m2(64, 256), (65536, 256): _m2(64, 256), (262144, 256): _m2(64, 256),
}


def m1_config_for(M: int, d: int):
    """Tuned M1 GemmConfig for (M, d), or None to use the path default."""
    return _M1_TABLE.get((M, d))


def m2_config_for(M: int, d: int):
    """Tuned M2 config dict for (M, d), or None to use the path default."""
    return _M2_TABLE.get((M, d))

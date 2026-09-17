"""Hopper candidate policies and native build integration.

Tunable defaults live here, never in kernel bodies. Instruction shapes and
full-width reduction constraints are explicit; unused knobs are excluded.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from functools import lru_cache, wraps

from quack.gemm_config import GemmConfig, _get_sm90_configs

# The GemmConfig fields that actually vary the sm90 kernel (device_capacity is fixed at 9;
# tile_k/num_warps are unused on WGMMA). These round-trip through the cache kwargs.
_TUNABLE_FIELDS = (
    "tile_m", "tile_n", "cluster_m", "cluster_n",
    "pingpong", "is_dynamic_persistent", "swap_ab", "max_swizzle_size",
)


def config_to_kwargs(c: GemmConfig) -> dict:
    """The tunable subset of a GemmConfig, as a cache-storable kwargs dict."""
    return {f: getattr(c, f) for f in _TUNABLE_FIELDS}


def kwargs_to_config(kw: dict) -> GemmConfig:
    """Rebuild a GemmConfig from cached kwargs (device_capacity pinned to 9 = sm90)."""
    unknown = set(kw) - set(_TUNABLE_FIELDS)
    if unknown:
        raise ValueError(f"unsupported Hopper config fields: {sorted(unknown)}")
    fields = {f: kw[f] for f in _TUNABLE_FIELDS if f in kw}
    return GemmConfig(device_capacity=9, **fields)


def validate_hopper_config(config):
    """Reject flags these custom epilogues cannot consume instead of ignoring them."""
    if (config.device_capacity != 9 or config.swap_ab or config.cluster_k != 1
            or config.tile_k is not None or config.num_warps is not None
            or config.use_tma_gather):
        raise ValueError("custom Hopper epilogues require unswapped SM90 WGMMA operands")


# --------------------------------------------------------------------------- #
# candidate spaces (the FULL library grid, filtered to sm90 / this epilogue)
# --------------------------------------------------------------------------- #
def _sm90(configs: Iterable[GemmConfig]) -> list[GemmConfig]:
    return [c for c in configs if c.device_capacity == 9 and not c.swap_ab]


# Keep the established defaults first. Expand only fields consumed by launchers.
CLUSTERS = ((1, 1), (1, 2), (2, 1), (2, 2))
SWIZZLES = (1, 2, 4, 8)
SCHEDULERS = (False, True)


def _expand(configs, *, dynamic=True, clusters=CLUSTERS):
    seeds = list(configs)
    result = list(seeds)
    seen = {tuple(config_to_kwargs(c).items()) for c in result}
    for c in seeds:
        for cm, cn in clusters:
            for dyn in SCHEDULERS if dynamic else (False,):
                for swizzle in SWIZZLES:
                    candidate = replace(c, cluster_m=cm, cluster_n=cn,
                                        is_dynamic_persistent=dyn, max_swizzle_size=swizzle)
                    key = tuple(config_to_kwargs(candidate).items())
                    if key not in seen:
                        seen.add(key)
                        result.append(candidate)
    return result


def _small_tiles():
    # Legal pingpong atom-layout 1x1 shapes; useful for small M/N and reduction tails.
    return [GemmConfig(tile_m=tm, tile_n=tn, pingpong=True,
                       is_dynamic_persistent=False, cluster_m=1, cluster_n=1,
                       device_capacity=9)
            for tm, tn in ((64, 64), (64, 128), (64, 192), (64, 256), (128, 64))]


def _cached_space(fn):
    # GemmConfig is frozen. Cache its immutable tuple, returning a fresh list so
    # callers can reorder/filter candidates without changing subsequent calls.
    @lru_cache(maxsize=128)
    def cached(*args, **kwargs):
        return tuple(fn(*args, **kwargs))
    @wraps(fn)
    def candidates(*args, **kwargs):
        return list(cached(*args, **kwargs))
    return candidates


@_cached_space
def gated_sm90_candidates() -> list[GemmConfig]:
    """Gated N tiles remain divisible by 32; no swapped operands."""
    return _expand(_sm90(_get_sm90_configs(epilogue="gated")) + _small_tiles())


@_cached_space
def plain_sm90_candidates() -> list[GemmConfig]:
    """Cooperative/pingpong tiles, CTA clusters, scheduler and swizzle sweep."""
    return _expand(_sm90(_get_sm90_configs(epilogue=None)) + _small_tiles())


@_cached_space
def lnbwd_pp_candidates() -> list[GemmConfig]:
    """Full-N reduction fixes pingpong, cluster_n=1 and tile_n=width.

    Sweep legal M tiles, multicast along M, scheduler and swizzle. Do not add
    cooperative or N clusters: the epilogue requires one full-width reduction.
    """
    seeds = [GemmConfig(tile_m=tm, tile_n=128, pingpong=True,
                       is_dynamic_persistent=False, cluster_m=cm, cluster_n=1,
                       swap_ab=False, max_swizzle_size=8, device_capacity=9)
             for tm in (64, 128, 192) for cm in (1, 2)]
    return _expand(seeds, clusters=((1, 1), (2, 1)))


@_cached_space
def tm2_candidates() -> list[GemmConfig]:
    """tm2 from-scratch dual-A gated GEMM candidate space. The only knob is ``tile_m`` — the
    number of stacked m64 WGMMA atoms (one warpgroup each), tiled over M via
    ``atom_layout=(tile_m//64,1,1)``. All configs compute identical math (tile_m=64 is the
    single-atom subset); it's a pure performance knob. tile_n=K=N is structural (single N-tile),
    no cluster (each CTA owns its M-tile). Row tails are padded by the launcher; candidates
    exceeding SMEM are dropped. Only ``tile_m`` is read by the kernel."""
    return [GemmConfig(tile_m=tm, tile_n=128, pingpong=False, is_dynamic_persistent=False,
                       cluster_m=1, cluster_n=1, swap_ab=False, max_swizzle_size=8, device_capacity=9)
            for tm in (64, 128, 192, 256)]


# --------------------------------------------------------------------------- #
# runtime config resolution (kernel side)
# --------------------------------------------------------------------------- #
def resolve_config(op, candidates, *, dtype, bucket, default=None, device_index=None, run=None):
    """Use a cached candidate or measure the declared grid during a native build."""
    from miniworld_engine.autotune.native import choose_config
    if default is not None and default in candidates:
        candidates = [default] + [c for c in candidates if c != default]
    kw = choose_config(op, [config_to_kwargs(c) for c in candidates], dtype=dtype,
                       bucket=bucket, device_index=device_index,
                       run=(lambda c: run(kwargs_to_config(c))) if run is not None else None)
    return kwargs_to_config(kw)


# WGMMA pingpong register limits (quack GemmSm90), not performance preferences.
LNBWD_TILE_N_MAX = {64: 256, 128: 208, 192: 128}


@_cached_space
def lnbwd_candidates(width, *, tile_m=None, cluster_m=None):
    return [replace(c, tile_n=width) for c in lnbwd_pp_candidates()
            if width <= LNBWD_TILE_N_MAX[c.tile_m]
            and (tile_m is None or c.tile_m == tile_m)
            and (cluster_m is None or c.cluster_m == cluster_m)]


@_cached_space
def fused_lnl_candidates():
    # M2 uses static scheduling and its own stats pipeline. Include the previously
    # qualified single-cluster tile alongside quack's cooperative/pingpong space.
    tiles = [GemmConfig(tile_m=128, tile_n=128, cluster_m=1, cluster_n=1,
                        pingpong=True, is_dynamic_persistent=False)]
    # The custom M2 stats pipeline fails numerical qualification for pingpong
    # M192/N128 on M=264 with a fused gate (all clusters/swizzles). This is an
    # epilogue restriction: the same tile is valid in the other GEMM families.
    return tiles + [c for c in plain_sm90_candidates()
                    if not c.is_dynamic_persistent and c.tile_m != 192 and c not in tiles]

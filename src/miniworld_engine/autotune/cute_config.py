"""Brute-force autotune for CuTe/CUTLASS GEMM kernels — the cute counterpart of the Triton
grid sweep.

The Triton kernels declare a full config grid and let ``triton.autotune`` bench every config
during a real run (captured by ``capture.py``). The cute kernels have NO Triton autotune loop:
historically each one **hardcoded a single hand-picked ``GemmConfig``** in its ``if config is
None:`` branch — which is exactly the "only tile_m / hardcoded config" problem. This module
removes that: a cute kernel instead

  1. declares its candidate space (``gated_sm90_candidates`` / ``plain_sm90_candidates`` —
     the SAME ``quack.gemm_config._get_sm90_configs`` grid the library ships, i.e. the full
     tile_m×tile_n × cluster × pingpong/coop sweep, NOT just tile_m), and
  2. calls :func:`resolve_config` to pick the cached fastest config for the running
     (gpu, dtype, shape-bucket), falling back to a documented default only on a cache miss.

The cache is built by :func:`sweep_and_cache`, which times EVERY candidate on-device (do_bench)
across representative shapes and writes the ranked cache that :func:`resolve_config` reads.

INVARIANT (as everywhere in autotune): every candidate computes the same math, so a cache
miss / stale / bad pick only costs speed, never correctness. Config is performance-only.
"""

from __future__ import annotations

from typing import Callable, Iterable

from quack.gemm_config import GemmConfig, _get_sm90_configs

from .cache import (
    config_space_hash,
    gpu_key,
    select_config,
    store_ranked_configs,
)

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
    fields = {f: kw[f] for f in _TUNABLE_FIELDS if f in kw}
    return GemmConfig(device_capacity=9, **fields)


def _as_cache_dicts(candidates: Iterable[GemmConfig]) -> list[dict]:
    """The cache layer (config_space_hash / select_config / store) speaks the
    ``{"kwargs": {...}}`` dict form, not raw GemmConfig — convert for it."""
    return [{"kwargs": config_to_kwargs(c)} for c in candidates]


# --------------------------------------------------------------------------- #
# candidate spaces (the FULL library grid, filtered to sm90 / this epilogue)
# --------------------------------------------------------------------------- #
def _sm90(configs: Iterable[GemmConfig]) -> list[GemmConfig]:
    return [c for c in configs if c.device_capacity == 9]


def gated_sm90_candidates() -> list[GemmConfig]:
    """Full sm90 sweep for a gated (swiglu/glu postact) GEMM: tile_n%32, no swap_ab, no m=192
    coop — exactly ``_get_sm90_configs(epilogue="gated")``."""
    return _sm90(_get_sm90_configs(epilogue="gated"))


def plain_sm90_candidates() -> list[GemmConfig]:
    """Full sm90 sweep for a plain (non-gated) GEMM epilogue."""
    return _sm90(_get_sm90_configs(epilogue=None))


def lnbwd_pp_candidates() -> list[GemmConfig]:
    """dgrad / dab LN-backward candidate space. tile_n=K and atom_layout 1×1 are fixed by the single
    full-N reduction (so pingpong, tile_m in {64,128,192}); tile_n here is a stable placeholder — the
    kernel uses tile_n=K. The remaining knobs are SWEPT, not guessed: tile_m and cluster_m (cluster_n
    stays 1 — the output N=K is a single N-tile, nothing to split across a cluster; cluster_m shares
    the B=Wᵀ load across M-CTAs). Configs that don't compile for a shape are dropped during the sweep."""
    return [GemmConfig(tile_m=tm, tile_n=128, pingpong=True, is_dynamic_persistent=False,
                       cluster_m=cm, cluster_n=1, swap_ab=False, max_swizzle_size=8, device_capacity=9)
            for tm in (64, 128, 192) for cm in (1, 2)]


def tm2_candidates() -> list[GemmConfig]:
    """tm2 from-scratch dual-A gated GEMM candidate space. The only knob is ``tile_m`` — the
    number of stacked m64 WGMMA atoms (one warpgroup each), tiled over M via
    ``atom_layout=(tile_m//64,1,1)``. All configs compute identical math (tile_m=64 is the
    single-atom subset); it's a pure performance knob. tile_n=K=N is structural (single N-tile),
    no cluster (each CTA owns its M-tile). Candidates that don't divide M or exceed SMEM for a
    shape are dropped during the sweep. Only ``tile_m`` is read by the kernel."""
    return [GemmConfig(tile_m=tm, tile_n=128, pingpong=False, is_dynamic_persistent=False,
                       cluster_m=1, cluster_n=1, swap_ab=False, max_swizzle_size=8, device_capacity=9)
            for tm in (64, 128, 192, 256)]


# --------------------------------------------------------------------------- #
# runtime config resolution (kernel side)
# --------------------------------------------------------------------------- #
def resolve_config(
    op: str,
    candidates: list[GemmConfig],
    *,
    dtype: str,
    bucket: str,
    default: GemmConfig,
    device_index: int | None = None,
) -> GemmConfig:
    """Pick the cached fastest GemmConfig for (gpu, dtype, bucket); ``default`` on miss.

    Passing ``candidates`` enables the config-space staleness check (a changed grid invalidates
    the old cache). During a capture run (``MINIWORLD_RUN_AUTOTUNE=1``) this returns ``default``
    so the sweep — not the cache — drives config choice."""
    best = select_config(op, dtype=dtype, bucket=bucket, candidates=_as_cache_dicts(candidates),
                         device_index=device_index)
    if best is None:
        return default
    try:
        return kwargs_to_config(best["kwargs"])
    except Exception:  # noqa: BLE001 -- malformed cache entry -> safe default
        return default


# --------------------------------------------------------------------------- #
# cache builder (brute-force timing sweep)
# --------------------------------------------------------------------------- #
def sweep_and_cache(
    op: str,
    dtype: str,
    cases: list[tuple[str, Callable[[GemmConfig], Callable[[], object]]]],
    candidates: list[GemmConfig],
    *,
    top_k: int = 5,
    warmup: int = 25,
    rep: int = 100,
    device_index: int | None = None,
    on_result: Callable[[str, str, GemmConfig, float], None] | None = None,
) -> None:
    """Brute-force every ``candidate`` on-device for each shape bucket and write the ranked cache.

    ``cases`` is a list of ``(bucket, make_run)`` where ``make_run(config)`` returns a
    zero-arg thunk that launches the kernel once with that config (compiled + inputs ready).
    A config that fails to compile/run for a shape is dropped (timed as +inf) — the sweep is
    a superset of what each kernel supports, and unsupported configs simply lose.
    """
    from triton.testing import do_bench  # noqa: PLC0415

    gk = gpu_key(device_index)
    csh = config_space_hash(_as_cache_dicts(candidates))
    for bucket, make_run in cases:
        ranked: list[tuple[GemmConfig, float]] = []
        for c in candidates:
            try:
                run = make_run(c)
                ms = float(do_bench(run, warmup=warmup, rep=rep, quantiles=(0.5, 0.2, 0.8))[0])
            except Exception:  # noqa: BLE001 -- unsupported/failed config -> skip
                continue
            ranked.append((c, ms))
            if on_result is not None:
                on_result(bucket, dtype, c, ms)
        if not ranked:
            continue
        ranked.sort(key=lambda t: t[1])
        # store_ranked_configs takes (config, ms); as_cfg_dict handles the dict form, so we
        # hand it the kwargs dict (cute config) rather than a GemmConfig.
        stored: list[tuple[object, float]] = [
            ({"kwargs": config_to_kwargs(c)}, ms) for c, ms in ranked
        ]
        store_ranked_configs(op, gk, dtype, bucket, stored, csh, top_k=top_k)

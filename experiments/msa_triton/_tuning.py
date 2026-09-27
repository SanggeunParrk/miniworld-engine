"""Autotune search spaces of the MSA Triton kernels, in the engine's GRID SPEC form (``axis,values``; the configs are the cartesian
product in axis order, see ``miniworld_engine.autotune.configs``).

Each kernel declares only its op name -- ``@triton.autotune(configs=configs("opm_epilogue_triton"), key=[...])`` -- which is the
engine's ``configs_for`` convention; ``write_specs(dir)`` writes the same spaces as ``<op>.csv`` files for a config directory.
Tile axes must match the kernel's constexpr names. ``PPS`` (persistent programs per SM) is a launch-grid axis: the kernel takes it as
an unused constexpr so the grid lambda can read it from META.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import triton

_META = ("num_warps", "num_stages")

# The SHIPPED search spaces (the `grid` config set): ladders as wide as the existing ops of the same kind (gemm: BLOCK 16..256,
# warps 1..8, stages 1..6; row / reduce kernels: rows 16..256, warps 1..8). They are NOT narrowed by any card's winners or by
# compile budget (docs/reports/autotune-space-20260916.md): the cache build measures them, and only configs that are impossible
# for correctness (a tile larger than an unmasked dimension; see each kernel's `prune`) or that the compiler / device rejects
# (the autotuner drops OutOfResources) are excluded.
_W = [1, 2, 4, 8]
_ST = [1, 2, 3, 4, 5, 6]
SPECS: dict[str, dict[str, list[int]]] = {
    # ---- OPM
    "opm_prologue_triton": {"BT": [16, 32, 64, 128, 256], "num_warps": _W, "num_stages": [1, 2, 3, 4]},
    "opm_epilogue_triton": {"TI": [1, 2, 4], "BJ": [16, 32, 64, 128], "BN": [32, 64, 128, 256], "BKD": [1, 2, 4], "num_warps": _W,
                            "num_stages": _ST},
    "opm_dgrad_triton": {"TI": [1, 2, 4], "BJ": [16, 32, 64, 128], "BN": [32, 64, 128, 256], "BKC": [16, 32, 64, 128],
                         "num_warps": _W, "num_stages": _ST},
    "opm_dwo_triton": {"BM": [16, 32, 64, 128], "BN": [32, 64, 128, 256], "BK": [16, 32, 64, 128], "SPLIT": [1, 2, 4, 8, 16, 32, 64, 128],
                       "num_warps": _W, "num_stages": _ST},
    "opm_prologue_bwd_triton": {"BT": [16, 32, 64, 128], "PPS": [1, 2, 4, 8, 16], "num_warps": _W, "num_stages": [1, 2, 3, 4]},
    # ---- PWA
    "pwa_pair_fwd_triton": {"BJ": [16, 32, 64, 128, 256], "num_warps": _W, "num_stages": [1, 2, 3, 4]},
    "pwa_value_fwd_triton": {"BS": [16, 32, 64, 128], "BN": [16, 32, 64, 128, 256, 512], "num_warps": _W, "num_stages": [1, 2, 3, 4]},
    "pwa_final_fwd_triton": {"BS": [16, 32, 64, 128], "num_warps": _W, "num_stages": [1, 2, 3, 4]},
    "pwa_glue_bwd_triton": {"BS": [16, 32, 64, 128], "PPS": [1, 2, 4, 8, 16], "num_warps": _W, "num_stages": [1, 2, 3, 4]},
    "pwa_dwv_triton": {"BS": [16, 32, 64, 128], "PPS": [1, 2, 4, 8, 16], "num_warps": _W, "num_stages": [1, 2, 3, 4]},
    "pwa_proj_bwd_triton": {"BS": [16, 32, 64, 128], "PPS": [1, 2, 4, 8, 16], "num_warps": _W, "num_stages": [1, 2, 3, 4]},
    "pwa_pair_bwd_triton": {"BJ": [16, 32, 64, 128, 256], "num_warps": _W, "num_stages": [1, 2, 3, 4]},
}

# DEVELOPMENT set (like the repo's blk64 / warp4 / mixed1 sets): a handful of configs so an iteration does not autotune the full
# ladder. A development input only -- never the shipped default. Selected with MSA_TRITON_CONFIG_SET=dev.
DEV: dict[str, dict[str, list[int]]] = {
    "opm_prologue_triton": {"BT": [64], "num_warps": [4], "num_stages": [1]},
    "opm_epilogue_triton": {"TI": [1, 2], "BJ": [64], "BN": [128], "BKD": [1, 2], "num_warps": [4, 8], "num_stages": [3]},
    "opm_dgrad_triton": {"TI": [1, 2], "BJ": [64], "BN": [64, 128], "BKC": [128], "num_warps": [4, 8], "num_stages": [2]},
    "opm_dwo_triton": {"BM": [128], "BN": [64, 128], "BK": [64], "SPLIT": [16, 32], "num_warps": [4, 8], "num_stages": [3]},
    "opm_prologue_bwd_triton": {"BT": [64], "PPS": [4], "num_warps": [4], "num_stages": [1]},
    "pwa_pair_fwd_triton": {"BJ": [64], "num_warps": [4], "num_stages": [1]},
    "pwa_value_fwd_triton": {"BS": [32], "BN": [128, 256], "num_warps": [4], "num_stages": [1]},
    "pwa_final_fwd_triton": {"BS": [32], "num_warps": [4], "num_stages": [1]},
    "pwa_glue_bwd_triton": {"BS": [32], "PPS": [2], "num_warps": [4], "num_stages": [1]},
    "pwa_dwv_triton": {"BS": [32], "PPS": [2], "num_warps": [4], "num_stages": [1]},
    "pwa_proj_bwd_triton": {"BS": [32], "PPS": [4], "num_warps": [4], "num_stages": [1]},
    "pwa_pair_bwd_triton": {"BJ": [64], "num_warps": [4], "num_stages": [1]},
}


def active_set() -> str:
    """``grid`` (the shipped spaces) unless MSA_TRITON_CONFIG_SET names the development set."""
    name = os.environ.get("MSA_TRITON_CONFIG_SET", "grid")
    if name not in ("grid", "dev"):
        raise ValueError(f"MSA_TRITON_CONFIG_SET must be 'grid' or 'dev', got {name!r}")
    return name


def configs(op: str) -> list:
    """The cartesian product of the active set's space for ``op`` as ``triton.Config`` objects (tile axes -> kwargs, the rest ->
    launch meta). Read when the kernel module imports, so the set is chosen by the environment before that."""
    spec = (DEV if active_set() == "dev" else SPECS)[op]
    axes = list(spec)
    out = []
    for vals in itertools.product(*(spec[a] for a in axes)):
        kw = dict(zip(axes, vals))
        meta = {m: kw.pop(m) for m in _META if m in kw}
        out.append(triton.Config(kw, **meta))
    return out


def write_specs(directory: str | Path, which: str = "grid") -> list[Path]:
    """Write every search space of ``which`` (``grid`` or ``dev``) as ``<op>.csv`` (GRID SPEC) under ``directory``."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for op, spec in (DEV if which == "dev" else SPECS).items():
        p = d / f"{op}.csv"
        p.write_text("axis,values\n" + "".join(f"{a},{' '.join(str(v) for v in vs)}\n" for a, vs in spec.items()))
        paths.append(p)
    return paths


def shape_key(*dims: int) -> int:
    """Autotune key bucket: the next power of two of each runtime extent, packed (so a sweep over L / S retunes per bucket)."""
    k = 0
    for x in dims:
        k = k * 64 + max(0, int(x) - 1).bit_length()
    return k


def prune(ok):
    """``early_config_prune`` keeping the configs for which ``ok(config_kwargs, call_args)`` holds -- correctness exclusions only
    (a tile axis larger than the unmasked dimension it tiles). Never used for performance."""
    def fn(configs, named_args, **kw):
        args = {**named_args, **kw}
        keep = [c for c in configs if ok(c.kwargs, args)]
        if not keep:
            raise ValueError("no config in the search space is valid for this call")
        return keep
    return fn


def pow2_at_least_16(**dims: int) -> str | None:
    """None if every named dimension is a power of two >= 16 (tl.arange / tl.dot constraints), else the reason."""
    bad = [f"{n}={v}" for n, v in dims.items() if v < 16 or v & (v - 1)]
    return None if not bad else "the Triton kernels need power-of-two dimensions >= 16, got " + ", ".join(bad)

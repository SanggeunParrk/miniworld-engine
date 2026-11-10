"""CUDA pass of the composed OuterProductMean (``gather.cu``), sm_90a. Built on first use (``load_extension``), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="opm_gather_sm90a",
        sources=[str(_dir / "gather.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def gather(prod, groups, k, weight=None, bias=None, eps=0.1):
    """prod [B G, N K, N K] bf16 -> [B N N, G K K] bf16 rows (g, k, l) [, LayerNorm * weight + bias over them]."""
    return ext().opm_gather(prod, int(groups), int(k), weight, bias, float(eps))



@functools.lru_cache(maxsize=None)
def chai_ext():
    """``chai.cu`` (the Chai-1 OuterProductMean's fused pair pass, TMA + wgmma), sm_90a. Built on first use."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="opm_chai_sm90a",
        sources=[str(_dir / "chai.cu")],
        extra_include_paths=[str(_dir)],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def chai_epilogue(o, wfold, bias, residual=None, save=False, eps=0.1):
    """O [8, N K, N K] bf16 (the grouped outer product, K = 8) -> [z [1, N, N, CZ] bf16, y [N N, 512] bf16, stats [N N, 2] fp32]: per pair the
    512 products LayerNorm'd (``eps``), projected with ``wfold`` [CZ, 512] (W_out . gamma) + ``bias`` [CZ] fp32 (b_out + W_out beta) and added to
    ``residual`` [1, N, N, CZ]; y / stats (the normalised products, the statistics) only with ``save`` (the training backward)."""
    return chai_ext().chai_epilogue(o, wfold, bias, residual, bool(save), float(eps))


def chai_proj(x, mask, gamma, beta, w, eps=1e-5, save=False):
    """x [T, 64] bf16 (T = S N), mask [T] bool | None, LayerNorm gamma / beta fp32, w [128, 64] bf16 (left | right projection): LayerNorm, both
    projections, mask -> [a, b [8, T, 8] bf16 (group-major: group g's 8 channels of every row), stats [T, 2] fp32 (mean, rstd) with ``save``]."""
    return chai_ext().chai_proj(x, mask, gamma, beta, w, float(eps), bool(save))


def chai_bwd(dy, y, stats, n):
    """dy [N N, 512] bf16 (dz W'), the forward's y and stats -> dO [8, N K, N K] bf16 (the product LayerNorm's backward in the grouped layout)."""
    return chai_ext().chai_bwd(dy, y, stats, int(n))


def chai_projbwd(x, mask, da, db, gamma, beta, stats, wlr):
    """Backward of ``chai_proj``: -> [dcat [T, 128], ln16 [T, 64], dx [T, 64], per-block [dgamma | dbeta] partials [T / 64, 128] fp32]."""
    return chai_ext().chai_projbwd(x, mask, da, db, gamma, beta, stats, wlr)


def prebuild(target, rows):
    """Every build of this module for the requirements ``rows`` (kernels/_prebuild.py); the training OPM extensions are in
    kernels/integrations/opm_train.py. The gather build serves the inference kernel (``gather``) of every
    row that has it; the Chai-1 build serves the ``chai_*`` kernels, which only a grouped
    (``groups``) OuterProductMean row launches."""
    from miniworld_engine.kernels._prebuild import _kv

    if target.arch == "sm80":   # A100: the sm_80 OPM builds are planned by kernels/integrations (opm_sm80); the grouped rows' passes here
        if any("groups" in _kv(r["options"]) and r["dtype"] in ("bfloat16", "master") for r in rows):
            from . import chai_sm80, ogemm_sm80

            yield chai_sm80.ext
            yield ogemm_sm80.ext   # the masked grouped product
        return
    names = set().union(*(r["kernels"] for r in rows)) if rows else set()
    if "outer_product_mean_gather_sm90" in names:
        yield ext
    if any("groups" in _kv(r["options"]) and any("_chai_" in k for k in r["kernels"]) for r in rows):
        yield chai_ext

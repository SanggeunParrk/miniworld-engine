"""Drivers for the ``layernorm`` family.

layernorm and layernorm_linear were one module (``drivers_ln.py``) and still
share the ``_L``/``_IS_PAIR``/``_M``/``_D``/``_PAIR_N``/``_act`` block, which lives in
``drivers/layernorm_linear.py``. ``_D_CUDA_BWD`` follows the requested width, including
ragged widths handled by the CUDA launcher's scalar fallback.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import (
    BF16,
    _ln_stats,
    dev,
    rows2d,
    vec,
)
from miniworld_engine.kernels.drivers.layernorm_linear import (
    _D,
    _IS_PAIR,
    _M,
    _PAIR_N,
    _act,
)

# The launcher validates vector alignment and falls back for ragged widths.
# Drive the requested width; pinning 128 silently left every wider build untuned.
_D_CUDA_BWD = _D


def _sm90plus() -> bool:
    """Is this an sm90+ card, where the row-scaled LN side is driven too?

    `dispatch.is_sm90plus`, imported lazily so a driver module stays importable on a machine with
    no CUDA (the CPU test suite imports every driver).
    """
    try:
        from miniworld_engine.modules.dispatch import is_sm90plus
        return is_sm90plus()
    except Exception:
        return False


# ── layernorm ────────────────────────────────────────────────────────────────────────────────


def layernorm_fwd_saveact_triton() -> None:
    from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm

    x = _act()
    # HAS_ROWSCALE is CARD-DEPENDENT, so this driver branches on the card rather than picking one
    # side for every build. The =1 program folds the AF pair-mask into the LN epilogue; its
    # production producers were the sm90+ trimul paths retired in v2.2.0, and the launcher still
    # accepts ``row_scale``, so the sm90+ side keeps it tuned.
    #
    #   sm80/sm86: =0 only. The triton trimul deliberately does NOT fold the mask into LN_in
    #     (unidirectional.py:225-227). A lookup from a path that cannot run is not a bucket worth
    #     tuning.
    #   sm90+: =1 is driven too, so an H100/B200 cache never leaves the masked LN on the
    #     heuristic subset.
    #
    # The registry row is `arch=sm80`, i.e. built on every card, so the branch has to live here.
    triton_layernorm(x, vec(_D), vec(_D), 1e-5)                              # HAS_ROWSCALE=0
    if _sm90plus():
        rs = torch.rand(x.reshape(-1, _D).shape[0], device=dev(), dtype=x.dtype)
        triton_layernorm(x, vec(_D), vec(_D), 1e-5, row_scale=rs)            # HAS_ROWSCALE=1


def layernorm_bwd_atomic_triton() -> None:
    from miniworld_engine.kernels.layernorm.compile_native import _bwd_atomic_impl

    # The pair activation (1, L, L, D), NOT its (M, D) flattening: `_bwd_atomic_impl` reshapes
    # internally and reads `both_key(rows_of(x.shape))` off the 4-D shape, so flattening here
    # would hand it M = L*L and clamp every L to the 8192 bucket.
    x = _act()
    mean, rstd = _ln_stats(x.reshape(-1, _D))  # [M] fp32, one row per (b, i, j)
    # HAS_ROWSCALE=0 is what `_bwd_atomic_impl` pins (compile_native.py:154). The =1 side comes
    # from a different launcher (main.py:419) and is USUALLY preceded at main.py:395 by a branch
    # routing bf16 with 128 <= N <= 512 to the hand-CUDA backward -- but "usually" is not "never":
    # that branch is wrapped in a bare `except Exception: pass`, so any nvcc/JIT failure falls
    # through to the triton launch, and the width guard does not cover the MSA width 64 the build
    # now drives. Cheap insurance on sm90+.
    _bwd_atomic_impl(torch.randn_like(x), x, vec(_D), mean, rstd)            # HAS_ROWSCALE=0
    if _sm90plus():
        from miniworld_engine import settings
        from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm

        previous = settings.current().layernorm_bwd_path
        settings.configure(layernorm_bwd_path="atomic")
        try:
            xg = _act().requires_grad_(True)
            rs = torch.rand(xg.reshape(-1, _D).shape[0], device=dev(), dtype=xg.dtype)
            triton_layernorm(xg, vec(_D), vec(_D), 1e-5, row_scale=rs).sum().backward()
        finally:
            settings.configure(layernorm_bwd_path=previous)


def layernorm_bwd_split_triton() -> None:
    # _bwd_persistent_impl allocates PART_DW/PART_DB as [SM*waves, N] fp32 and passes
    # partial_dw.stride(0) as stride_part, with grid (g, cdiv(N, BLOCK_K)).
    from miniworld_engine.kernels.layernorm.compile_native import _bwd_persistent_impl

    x = _act()  # pre-flatten, like layernorm_bwd_atomic_triton above
    mean, rstd = _ln_stats(x.reshape(-1, _D))
    _bwd_persistent_impl(torch.randn_like(x), x, vec(_D), mean, rstd)


def layernorm_fwd_mmajor_triton() -> None:
    from miniworld_engine.kernels.layernorm.triton.transpose import layer_norm_transpose

    # Channel-major (D, B, N), and M = B*N is what `_ln_transpose_dbn_bnd` keys on now (a
    # `level=both` kernel buckets on rows). So the two sides differ in the SHAPE OF B*N, not just
    # in a constant: the pair side is B=N=L (M = L*L) and the atom side is B=1, N=A (M = A).
    # Building the pair packing on both sides left this op's six atom buckets empty -- it was the
    # only hole in either card's cache after the row-key rebuild.
    x = (torch.randn(_D, _PAIR_N, _PAIR_N, device=dev(), dtype=BF16) if _IS_PAIR
         else torch.randn(_D, 1, _M, device=dev(), dtype=BF16))
    layer_norm_transpose(x, vec(_D), vec(_D), layout="dbn->bnd")


def layer_norm_fwd_kernel() -> None:
    from miniworld_engine.kernels.layernorm.cuda import layer_norm_cuda

    layer_norm_cuda.layer_norm_fwd(rows2d(_M, _D), vec(_D), vec(_D), 1e-5)


def layer_norm_bwd_main_kernel() -> None:
    from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda

    x = rows2d(_M, _D_CUDA_BWD)
    mean, rstd = _ln_stats(x)
    layer_norm_bwd_cuda(torch.randn_like(x), x, vec(_D_CUDA_BWD), mean, rstd)


def layer_norm_bwd_reduce_kernel() -> None:
    # Same launcher as the main kernel: layer_norm_cuda_bwd runs main then reduce.
    from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda

    x = rows2d(_M, _D_CUDA_BWD)
    mean, rstd = _ln_stats(x)
    layer_norm_bwd_cuda(torch.randn_like(x), x, vec(_D_CUDA_BWD), mean, rstd)

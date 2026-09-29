"""Drivers for the ``layernorm`` family.

layernorm, layernorm_linear and fused_ln_mask were one module (``drivers_ln.py``) and still
share the ``_L``/``_IS_PAIR``/``_M``/``_D``/``_PAIR_N``/``_act`` block, which lives in
``drivers/layernorm_linear.py``. ``_D_CUDA_BWD`` -- the width the CUDA backward is compiled
for -- is this family's own and stays here.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import (
    BF16,
    _ln_stats,
    aligned_only,
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

# The FEATURE width for the hand-CUDA LayerNorm BACKWARD only. `layer_norm_bwd_main_kernel`
# reads X/DY through a vector type -- `layer_norm_cuda_kernel.cu:276`
#     xp.vec = *reinterpret_cast<const VecT*>(x + col0);
# with VecT = uint2/uint4, EPT = TX_BYTES/sizeof(scalar_t) elements per transaction and
# col0 = (v*32 + lane)*EPT. Line 269 states the precondition outright: "N % EPT == 0 for every
# launched (N, TX_BYTES) combo, so a transaction whose base column is in range never overshoots
# N." The host side (`:465`) only chooses BETWEEN the two widths --
#     const int txb = (N % (32 * (16 / elt)) == 0) ? 16 : 8;
# -- so for bf16 it falls back to uint2 (EPT=4) and there is NO scalar tail. At N=125 the row base
# X + row*N is 250 bytes in, which is 2 (mod 8) on odd rows, and the last transaction (col0=124)
# also runs 3 columns past the row.
#
# This is the vector-width requirement of a hand-written CUDA kernel, not a Triton tile mask, so
# it is pinned rather than left to fault. It is pinned as NARROWLY as possible: only the two
# `layer_norm_bwd_cuda` drivers use it. `_D` stays ragged for the other 16 kernels, `_M` stays
# ragged here too (the kernel grid-strides rows under `row < M`, and N=128 keeps every row base
# 16B-aligned), and the SCALAR forward kernel keeps plain `_D` -- it is measured at _D=125 and
# passes, which is what shows the fault belongs to the vector path and not to "CUDA at 125".
_D_CUDA_BWD = aligned_only(
    "layernorm.cuda backward feature width (N)",
    128,
    "layer_norm_cuda_kernel.cu:269 declares 'N % EPT == 0' and :276 loads X/DY as uint2/uint4; "
    ":465 only picks uint4-vs-uint2 (no scalar tail), so bf16 N=125 (EPT=4) misaligns the row "
    "base and overruns the row -> 'AcceleratorError: CUDA error: misaligned address'. NOTE the "
    "precondition is nowhere enforced: there is no TORCH_CHECK on N % EPT and no fallback, so an "
    "out-of-contract N faults instead of being rejected -- a separate, real robustness defect.",
)


def _sm90plus() -> bool:
    """Is this card one where the cute paths -- the only producers of a row-scaled LN -- run?

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
    # side for every build. The =1 program folds the AF pair-mask into the LN epilogue, and every
    # producer of it is a cute path -- `cute/v6_training_merged.py:144`, `cute/bidir_training.py`,
    # `cute/inference.py` -- which `dispatch` selects only at sm90+.
    #
    #   sm80/sm86: =0 only. The triton trimul deliberately does NOT fold the mask into LN_in
    #     (unidirectional.py:225-227). `dev audit --replay` on an A100 DID show `HAS_ROWSCALE=1`
    #     lookups, and each is immediately followed by `NotImplementedError: Gemm Sm80 is not
    #     implemented yet` -- `builder.cases()` forces `implementation="cute"` and the case aborts.
    #     A lookup from a path that cannot run is not a bucket worth tuning.
    #   sm90+: =1 is production. Pinning this driver to =0 would ship an H100/B200 cache with the
    #     masked LN -- the normal case there -- permanently on the heuristic subset.
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
    # now drives. Cheap insurance on the card where the cute paths make it production anyway.
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

    x = rows2d(_M, _D_CUDA_BWD)  # feature width pinned: see _D_CUDA_BWD
    mean, rstd = _ln_stats(x)
    layer_norm_bwd_cuda(torch.randn_like(x), x, vec(_D_CUDA_BWD), mean, rstd)


def layer_norm_bwd_reduce_kernel() -> None:
    # Same launcher as the main kernel: layer_norm_cuda_bwd runs main then reduce.
    from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda

    x = rows2d(_M, _D_CUDA_BWD)  # feature width pinned: see _D_CUDA_BWD
    mean, rstd = _ln_stats(x)
    layer_norm_bwd_cuda(torch.randn_like(x), x, vec(_D_CUDA_BWD), mean, rstd)

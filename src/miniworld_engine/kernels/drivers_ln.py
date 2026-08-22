"""Drivers for the layernorm / layernorm_linear / fused_ln_mask kernels."""

from __future__ import annotations

import torch

from miniworld_engine.autotune.shape_key import both_key

from .drivers import BF16, aligned_only, both_level_is_pair, dev, driver_length, pair, ragged, rows2d, vec

# The pair row/feature extents the bench walks: `bench_kernel_layernorm` builds
# x = (1, L, L, d_pair) and `bench_kernel_layernorm_bwd` / `bench_kernel_gemm_epil` build
# (L*L, d_pair). d_pair defaults to 128 and `drivers.pair()` defaults to L=128, so M = 128*128.
#
# Every extent goes through `ragged()`: unchanged in the default aligned mode, minus 3 under
# MINIWORLD_SHAPE_MODE=ragged, which makes the tail tile partial for all five config sets.
#   _D      -- the LayerNorm *reduce* axis (the feature/hidden width). A wrong mask here
#              corrupts mean/rstd for the whole row, so this is the axis that matters most.
#              It also drives every gamma/beta vector width and the (N, K) projection weight,
#              which must move with it or the shapes stop matching.
#   _M      -- the row axis (one LayerNorm row per m), tiled by BLOCK_M / the persistent grid.
#   _PAIR_N -- the pair sequence length L of the (B, L, L, D) activations, so the flattened
#              row count B*L*L is ragged for the kernels that take a 4-D pair tensor.
#   _L      -- the pair sequence length L itself, the ONE quantity `autotune.shape_key` buckets.
#              `_M` is L*L and `_PAIR_N` is L, so both move together when MINIWORLD_DRIVER_LENGTH
#              moves and the shape the kernel records is the shape the driver asked for.
_L = driver_length(128)
# M = (ragged L)**2, NOT ragged(L**2). The five drivers that hand over a 4-D pair activation
# flatten to _PAIR_N**2 rows, so deriving _M any other way makes the constant disagree with the
# tensor in ragged mode (16381 vs 15625) -- and _M is what the flat drivers in this file build.
# Both values are ragged w.r.t. every tile width; the point is that there is only one M.
_IS_PAIR = both_level_is_pair(_L)   # token side -> (1,L,L,D); atom side -> (1,A,D)
# M is what the FLAT drivers in this file build, and it must equal what _act()
# flattens to: L*L on the pair side, A on the atom side.
_M = ragged(_L) ** 2 if both_level_is_pair(_L) else ragged(_L)
_D = ragged(128)
_PAIR_N = ragged(_L)

#: The bucket a launch at `_L` must record, for the launchers that only ever see the flattened
#: (M, D) matrix and therefore take the key as an explicit argument (`shape_key=`). `length_of`
#: cannot be used at those: M = L*L clamps to the top bucket (8192) at every L >= 91, and their
#: `shape_key=None` default clamps to the bottom one (128) -- either way the bucket stops moving
#: with L. Everything else here hands the wrapper the activation BEFORE it is flattened, which is
#: the same fix `modules/triangle_attention` took, and lets the wrapper read L = shape[-2] itself.
_SHAPE_KEY = both_key(_L)

#: n_head for the pair-bias projection, as `bench_kernel_tri_attn` derives it from d_pair.
#: Defined once here and imported by ``checks_ln`` so the projection's output width cannot
#: drift between driver and checker when _D moves.
_NH = _D // 32

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



def _act(d: int = None) -> torch.Tensor:
    """The PRE-FLATTEN activation for a level=both kernel at the driven bucket.

    Pair (1, L, L, D) on the token side, atom (1, A, D) on the atom side -- see
    ``drivers.both_level_is_pair``. Building a pair at every bucket asks for M = L*L rows where
    production hands over A: 67 million rows and 16 GiB at L=8192, which is what OOM'd 20 probes
    and drove one into an int32 offset overflow. `length_of` reads shape[-2] either way, so both
    layouts record the same shape_key.
    """
    d = _D if d is None else d
    if _IS_PAIR:
        return pair(n=_PAIR_N, d=d)
    return torch.randn(1, _PAIR_N, d, device=dev(), dtype=BF16)

def _ln_stats(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, rstd) fp32 [M] for a (M, N) x -- how bench_kernel_layernorm_bwd makes them."""
    xf = x.float()
    return xf.mean(-1), torch.rsqrt(xf.var(-1, unbiased=False) + eps)


def _mmajor(m: int = _M, n: int = _D) -> torch.Tensor:
    """(M, K) with strides (1, M): the m-major trimul view te_style/mmajor_bwd are written for."""
    return torch.randn(n, m, device=dev(), dtype=BF16).t()


# ── layernorm ────────────────────────────────────────────────────────────────────────────────


def layernorm_fwd_saveact_triton() -> None:
    from .layernorm.triton.main import triton_layernorm

    x = _act()
    triton_layernorm(x, vec(_D), vec(_D), 1e-5)


def layernorm_bwd_atomic_triton() -> None:
    from .layernorm.compile_native import _bwd_atomic_impl

    # The pair activation (1, L, L, D), NOT its (M, D) flattening: `_bwd_atomic_impl` reshapes
    # internally and reads `both_key(length_of(x.shape))` off the 4-D shape, so flattening here
    # would hand it M = L*L and clamp every L to the 8192 bucket.
    x = _act()
    mean, rstd = _ln_stats(x.reshape(-1, _D))  # [M] fp32, one row per (b, i, j)
    _bwd_atomic_impl(torch.randn_like(x), x, vec(_D), mean, rstd)


def layernorm_bwd_split_triton() -> None:
    # _bwd_persistent_impl allocates PART_DW/PART_DB as [SM*waves, N] fp32 and passes
    # partial_dw.stride(0) as stride_part, with grid (g, cdiv(N, BLOCK_K)).
    from .layernorm.compile_native import _bwd_persistent_impl

    x = _act()  # pre-flatten, like layernorm_bwd_atomic_triton above
    mean, rstd = _ln_stats(x.reshape(-1, _D))
    _bwd_persistent_impl(torch.randn_like(x), x, vec(_D), mean, rstd)


def layernorm_fwd_mmajor_triton() -> None:
    from .layernorm.triton.transpose import layer_norm_transpose

    # (D, B, N) with B = N = L, so M = B*N = L*L rows exactly as before. `_ln_transpose_dbn_bnd`
    # keys on `both_key(n)` -- the token axis of the (B, N, D) result -- so the old (D, 1, L*L)
    # packing handed it n = M and clamped to 8192 at every L.
    x = torch.randn(_D, _PAIR_N, _PAIR_N, device=dev(), dtype=BF16)
    layer_norm_transpose(x, vec(_D), vec(_D), layout="dbn->bnd")


def layer_norm_fwd_kernel() -> None:
    from .layernorm.cuda import layer_norm_cuda

    layer_norm_cuda.layer_norm_fwd(rows2d(_M, _D), vec(_D), vec(_D), 1e-5)


def layer_norm_bwd_main_kernel() -> None:
    from .layernorm.cuda import layer_norm_bwd_cuda

    x = rows2d(_M, _D_CUDA_BWD)  # feature width pinned: see _D_CUDA_BWD
    mean, rstd = _ln_stats(x)
    layer_norm_bwd_cuda(torch.randn_like(x), x, vec(_D_CUDA_BWD), mean, rstd)


def layer_norm_bwd_reduce_kernel() -> None:
    # Same launcher as the main kernel: layer_norm_cuda_bwd runs main then reduce.
    from .layernorm.cuda import layer_norm_bwd_cuda

    x = rows2d(_M, _D_CUDA_BWD)  # feature width pinned: see _D_CUDA_BWD
    mean, rstd = _ln_stats(x)
    layer_norm_bwd_cuda(torch.randn_like(x), x, vec(_D_CUDA_BWD), mean, rstd)


# ── fused_ln_mask ────────────────────────────────────────────────────────────────────────────


def layernorm_fwd_rowscale_triton() -> None:
    from .fused_ln_mask.cute.fused_ln_mask import fused_ln_mask

    x = _act()
    mask = (torch.rand(*x.shape[:-1], device=dev()) > 0.1).to(BF16)  # (B, L, L), required
    fused_ln_mask(x, vec(_D), vec(_D), mask, 1e-5)


# ── layernorm_linear ─────────────────────────────────────────────────────────────────────────


def layernorm_stats_triton() -> None:
    from .layernorm_linear.triton.stats import stats_triton

    # stats_triton asserts x.dim() == 2, so L cannot travel in the shape: pass the key.
    stats_triton(rows2d(_M, _D), 1e-5, shape_key=_SHAPE_KEY)


def layernorm_fwd_recompute_triton() -> None:
    from .layernorm_linear.triton.recompute import _recompute_xnormed

    x = rows2d(_M, _D)
    mean, rstd = _ln_stats(x)
    _recompute_xnormed(x, vec(_D), vec(_D), mean, rstd, shape_key=_SHAPE_KEY)


def layernorm_fwd_recompute_noaffine_triton() -> None:
    from .layernorm_linear.triton.recompute import _recompute_xhat

    x = rows2d(_M, _D)
    mean, rstd = _ln_stats(x)
    _recompute_xhat(x, mean, rstd, shape_key=_SHAPE_KEY)


def layernorm_fwd_saveact_strided_triton() -> None:
    from .layernorm_linear.triton.te_style import _ln_materialize

    _ln_materialize(_mmajor(), vec(_D), vec(_D), 1e-5, shape_key=_SHAPE_KEY)


def layernorm_bwd_atomic_strided_triton() -> None:
    from .layernorm_linear.triton.mmajor_bwd import _ln_bwd_atomic

    x = _mmajor()
    mean, rstd = _ln_stats(x)
    _ln_bwd_atomic(_mmajor(), x, vec(_D), mean, rstd, x.stride(), shape_key=_SHAPE_KEY)


def layernorm_bwd_split_mmajor_triton() -> None:
    # _ln_bwd_persistent_new allocates PDG/PDB as [SM*waves, K] fp32, passes pdg.stride(0)
    # as stride_part and x.stride(1) as the feature stride; grid (NP, cdiv(K, BLOCK_K)).
    from .layernorm_linear.triton.mmajor_bwd import _ln_bwd_persistent_new

    x = _mmajor()
    mean, rstd = _ln_stats(x)
    _ln_bwd_persistent_new(
        _mmajor(), x, vec(_D), mean, rstd, x.stride(), shape_key=_SHAPE_KEY
    )


def layernorm_linear_fwd_triton() -> None:
    from .layernorm_linear.triton.fused import layernorm_linear_triton_fwd

    w = (torch.randn(_D, _D, device=dev(), dtype=BF16) * (_D**-0.5)).contiguous()  # (N, K)
    # `layernorm_linear_triton_fwd` flattens to (M, K) itself and reshapes the result back, so
    # the reshape this used to do here only destroyed L -- the same defect that was fixed in
    # modules/triangle_attention/module.py.
    layernorm_linear_triton_fwd(_act(), vec(_D), vec(_D), w, None, 1e-5)


def layernorm_linear_fwd_fp32_triton() -> None:
    from .layernorm_linear.triton.pair_bias import _fwd_op

    pw = torch.randn(_NH, _D, device=dev(), dtype=BF16)
    _fwd_op(_act(), vec(_D), pw, 1e-5)  # pre-flatten: _fwd_op reads shape[-2]


def layernorm_linear_bwd_fp32_triton() -> None:
    from .layernorm_linear.triton.pair_bias import _bwd_op, _fwd_op

    # Both ops run here, so both have to record the same bucket: `_fwd_op` gets the pre-flatten
    # pair activation and reads L off it, `_bwd_op` only ever takes the (M, N) matrix and so takes
    # the key explicitly (its own `shape_key=None` fallback is `both_key(M)` -> 8192).
    x = _act()
    x2 = x.reshape(-1, _D).contiguous()
    lnw, pw = vec(_D), torch.randn(_NH, _D, device=dev(), dtype=BF16)
    out, mean, rstd = _fwd_op(x, lnw, pw, 1e-5)
    _bwd_op(torch.randn_like(out), x2, lnw, pw, mean, rstd, shape_key=_SHAPE_KEY)


def gemm_lnl_fused_sm90_kernel() -> None:
    from .layernorm_linear.cute.gemm_layernorm_linear_fused import layernorm_linear_cute_fused

    w = (torch.randn(_D, _D, device=dev(), dtype=BF16) * (_D**-0.5)).contiguous()  # (N, K)
    layernorm_linear_cute_fused(rows2d(_M, _D), vec(_D), vec(_D), w, None, 1e-5)

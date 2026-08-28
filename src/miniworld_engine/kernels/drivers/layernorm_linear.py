"""Drivers for the ``layernorm_linear`` family -- and the shape block ``layernorm`` and
``fused_ln_mask`` import.

The three families were one module (``drivers_ln.py``) and still share
``_L``/``_IS_PAIR``/``_M``/``_D``/``_PAIR_N``/``_act``; the block lives here because
layernorm_linear has the most kernels reading it.
"""
from __future__ import annotations

import torch

from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.drivers import (
    BF16,
    _ln_stats,
    both_level_is_pair,
    dev,
    driver_length,
    driver_width,
    pair,
    ragged,
    rows2d,
    vec,
)

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
_D = ragged(driver_width(128))
_PAIR_N = ragged(_L)

#: The bucket a launch at `_L` must record, for the launchers that only ever see the flattened
#: (M, D) matrix and therefore take the key as an explicit argument (`shape_key=`). A `level=both`
#: kernel keys on ROWS (autotune/shape_key.py::BOTH_ROWS), so this is the row count the activation
#: below flattens to -- pair L*L, atom A -- and not L. Everything else here hands the wrapper the
#: activation BEFORE it is flattened, which is the same fix `modules/triangle_attention` took, and
#: lets the wrapper read the row count itself.
_SHAPE_KEY = both_key(_M)

#: n_head for the pair-bias projection, as `bench_kernel_tri_attn` derives it from d_pair.
#: Defined once here and imported by ``checks_ln`` so the projection's output width cannot
#: drift between driver and checker when _D moves.
_NH = _D // 32



def _act(d: int | None = None) -> torch.Tensor:
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


def _mmajor(m: int = _M, n: int = _D) -> torch.Tensor:
    """(M, K) with strides (1, M): the m-major trimul view te_style/mmajor_bwd are written for."""
    return torch.randn(n, m, device=dev(), dtype=BF16).t()


# ── layernorm_linear ─────────────────────────────────────────────────────────────────────────


def layernorm_stats_triton() -> None:
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

    # stats_triton asserts x.dim() == 2, so L cannot travel in the shape: pass the key.
    stats_triton(rows2d(_M, _D), 1e-5, shape_key=_SHAPE_KEY)


def layernorm_fwd_recompute_triton() -> None:
    from miniworld_engine.kernels.layernorm_linear.triton.recompute import (
        _recompute_xnormed,
    )

    x = rows2d(_M, _D)
    mean, rstd = _ln_stats(x)
    _recompute_xnormed(x, vec(_D), vec(_D), mean, rstd, shape_key=_SHAPE_KEY)


def layernorm_fwd_recompute_noaffine_triton() -> None:
    from miniworld_engine.kernels.layernorm_linear.triton.recompute import (
        _recompute_xhat,
    )

    x = rows2d(_M, _D)
    mean, rstd = _ln_stats(x)
    _recompute_xhat(x, mean, rstd, shape_key=_SHAPE_KEY)


def layernorm_fwd_saveact_strided_triton() -> None:
    from miniworld_engine.kernels.layernorm_linear.triton.te_style import (
        _ln_materialize,
    )

    _ln_materialize(_mmajor(), vec(_D), vec(_D), 1e-5, shape_key=_SHAPE_KEY)


def layernorm_bwd_atomic_strided_triton() -> None:
    from miniworld_engine.kernels.layernorm_linear.triton.mmajor_bwd import (
        _ln_bwd_atomic,
    )

    x = _mmajor()
    mean, rstd = _ln_stats(x)
    _ln_bwd_atomic(_mmajor(), x, vec(_D), mean, rstd, x.stride(), shape_key=_SHAPE_KEY)


def layernorm_bwd_split_mmajor_triton() -> None:
    # _ln_bwd_persistent_new allocates PDG/PDB as [SM*waves, K] fp32, passes pdg.stride(0)
    # as stride_part and x.stride(1) as the feature stride; grid (NP, cdiv(K, BLOCK_K)).
    from miniworld_engine.kernels.layernorm_linear.triton.mmajor_bwd import (
        _ln_bwd_persistent_new,
    )

    x = _mmajor()
    mean, rstd = _ln_stats(x)
    _ln_bwd_persistent_new(
        _mmajor(), x, vec(_D), mean, rstd, x.stride(), shape_key=_SHAPE_KEY
    )


def layernorm_linear_fwd_triton() -> None:
    from miniworld_engine.kernels.layernorm_linear.triton.fused import (
        layernorm_linear_triton_fwd,
    )

    w = (torch.randn(_D, _D, device=dev(), dtype=BF16) * (_D**-0.5)).contiguous()  # (N, K)
    # `layernorm_linear_triton_fwd` flattens to (M, K) itself and reshapes the result back, so
    # the reshape this used to do here only destroyed L -- the same defect that was fixed in
    # modules/triangle_attention/module.py.
    layernorm_linear_triton_fwd(_act(), vec(_D), vec(_D), w, None, 1e-5)


def layernorm_linear_fwd_fp32_triton() -> None:
    from miniworld_engine.kernels.layernorm_linear.triton.pair_bias import _fwd_op

    pw = torch.randn(_NH, _D, device=dev(), dtype=BF16)
    _fwd_op(_act(), vec(_D), pw, 1e-5)  # pre-flatten: _fwd_op reads shape[-2]


def layernorm_linear_bwd_fp32_triton() -> None:
    from miniworld_engine.kernels.layernorm_linear.triton.pair_bias import (
        _bwd_op,
        _fwd_op,
    )

    # Both ops run here, so both have to record the same bucket: `_fwd_op` gets the pre-flatten
    # pair activation and reads L off it, `_bwd_op` only ever takes the (M, N) matrix and so takes
    # the key explicitly (its own `shape_key=None` fallback is `both_key(M)` -> 8192).
    x = _act()
    x2 = x.reshape(-1, _D).contiguous()
    lnw, pw = vec(_D), torch.randn(_NH, _D, device=dev(), dtype=BF16)
    out, mean, rstd = _fwd_op(x, lnw, pw, 1e-5)
    _bwd_op(torch.randn_like(out), x2, lnw, pw, mean, rstd, shape_key=_SHAPE_KEY)


def gemm_lnl_fused_sm90_kernel() -> None:
    from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
        layernorm_linear_cute_fused,
    )

    w = (torch.randn(_D, _D, device=dev(), dtype=BF16) * (_D**-0.5)).contiguous()  # (N, K)
    layernorm_linear_cute_fused(rows2d(_M, _D), vec(_D), vec(_D), w, None, 1e-5)

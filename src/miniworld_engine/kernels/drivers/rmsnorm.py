"""Drivers for the ``rmsnorm`` family.

THREE entry points, and they do not share a width. ``triton_rmsnorm`` normalizes per attention
HEAD -- ``modules/swa_atom_attention`` over ``head_dim = d_model // n_heads`` (module.py:457) and
``kernels/triangle_attention/whole_op.py`` over its own ``d_head`` -- so its normalized axis is
small and the ladders in ``configs/grid/rmsnorm_*.csv`` start their BLOCK_K at 32 rather than at
layernorm's 64. ``triton_rmsnorm_modulate`` and ``triton_rmsnorm_adamod`` are DiT-block ops and
act on the whole ``d_model`` vector. A driver that used one width for all three would tune a
shape two of them never present.

Every constexpr combination production actually reaches is driven here, because ``HAS_WEIGHT``
and ``HAS_MODULATION`` are in the kernels' autotune ``key``: each combination is a separate
compiled kernel AND a separate cache bucket, so one driven combination says nothing about the
others. Both values of ``HAS_WEIGHT`` are real -- SWA's q/k normalization is non-affine
(module.py:433) and triangle_attention passes a weight -- and so the weighted and unweighted
forms are both launched below rather than one standing in for the other.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import (
    BF16,
    both_level_is_pair,
    dev,
    driver_length,
    driver_width,
    ragged,
    vec,
)

_L = driver_length(128)
#: ``level=both``: the token side hands over a 4-D pair activation, so its rows are ``L*L``; the
#: atom side hands over a 3-D single one and its rows are ``L``. The same split
#: ``drivers/layernorm_linear.py`` makes, for the same reason -- one length, two row counts.
_M = ragged(_L) ** 2 if both_level_is_pair(_L) else ragged(_L)

#: d_model. The base every other width here derives from, per :func:`driver_width` -- overriding
#: it moves the whole family the way changing the model's width would.
_D_BASE = driver_width(128)
#: head_dim, the axis ``triton_rmsnorm`` reduces. The SWA atom block runs d_model 128 over 4
#: heads; the ratio, not the 32, is what follows the base.
_D = ragged(_D_BASE // 4)
#: d_model itself, for the two DiT-block entry points.
_DM = ragged(_D_BASE)
#: d_cond FOLLOWS d_model the way the model pairs them -- 384 conditioning a token width, 128 on
#: the atom side -- for the reason ``drivers/conditioned_transition.py`` spells out at length.
#: ``by=5`` so that in ragged mode d_cond and d_model are DIFFERENT non-aligned values: a mask bug
#: on the conditioning axis, or a launcher reading one width where it means the other, is
#: invisible while the two are equal.
_DC = ragged(384 if _D_BASE > 128 else 128, by=5)
_EPS = 1e-5


def _rows(d: int) -> torch.Tensor:
    """The PRE-flatten activation at width ``d``, as the real callers hand it over.

    3-D, not (M, D): ``rows_of`` refuses an already-flattened shape on purpose -- a caller holding
    only (M, D) cannot say whether its M is the whole launch or one slice of it. SWA passes
    [N, S, H, D] and triangle_attention its own 4-D; either way the launcher flattens.
    """
    return torch.randn(1, _M, d, device=dev(), dtype=BF16, requires_grad=True)



# ── rmsnorm ──────────────────────────────────────────────────────────────────────────────────


def _mod_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(x, scale, shift)`` at d_model. scale/shift are per-ELEMENT -- chunks of a projection of
    the conditioning vector -- so they carry x's shape, not a per-channel one."""
    return _rows(_DM), _rows(_DM), _rows(_DM)


def rmsnorm_fwd_triton() -> None:
    """rmsnorm_fwd_kernel, at all FOUR (HAS_WEIGHT, HAS_MODULATION) combinations.

    One driver and not four rows, because a config ladder belongs to a KERNEL: the modulate form
    is `rmsnorm_fwd_kernel` with a constexpr flipped, and it reads `configs_for(
    "rmsnorm_fwd_triton")` like the plain form does. A second registry row would have to name a
    ladder nothing reads. The four DO get four cache buckets -- both flags are in the autotune
    key -- which is why all four are launched here.

    Two widths for the same reason: the q/k normalization reduces over head_dim and the DiT
    modulate over d_model, and the shape key separates them.
    """
    from miniworld_engine.kernels.rmsnorm.interface import (
        triton_rmsnorm,
        triton_rmsnorm_modulate,
    )

    triton_rmsnorm(_rows(_D), vec(_D), _EPS)   # triangle_attention: affine
    triton_rmsnorm(_rows(_D), None, _EPS)      # SWA q/k: non-affine
    x, sc, sh = _mod_args()
    triton_rmsnorm_modulate(x, sc, sh, vec(_DM), _EPS)
    x, sc, sh = _mod_args()
    triton_rmsnorm_modulate(x, sc, sh, None, _EPS)


def rmsnorm_bwd_triton() -> None:
    """rmsnorm_bwd_kernel, via the autograd entry points, at all four combinations."""
    from miniworld_engine.kernels.rmsnorm.interface import (
        triton_rmsnorm,
        triton_rmsnorm_modulate,
    )

    triton_rmsnorm(_rows(_D), vec(_D), _EPS).sum().backward()
    triton_rmsnorm(_rows(_D), None, _EPS).sum().backward()
    x, sc, sh = _mod_args()
    triton_rmsnorm_modulate(x, sc, sh, vec(_DM), _EPS).sum().backward()
    x, sc, sh = _mod_args()
    triton_rmsnorm_modulate(x, sc, sh, None, _EPS).sum().backward()

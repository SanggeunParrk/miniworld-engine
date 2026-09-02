"""Drivers for the ``rmsnorm`` family -- and the shape block ``drivers/rmsnorm_adamod.py`` shares.

``triton_rmsnorm`` normalizes per attention HEAD: ``modules/swa_atom_attention`` over
``head_dim = d_model // n_heads`` (module.py:457) and ``kernels/triangle_attention/whole_op.py``
over its own ``d_head``. Its normalized axis is small, which is why the ladders in
``configs/grid/rmsnorm_*.csv`` start BLOCK_K at 32 rather than layernorm's 64. The adaLN modulate
that acts on the whole ``d_model`` vector is the separate ``rmsnorm_adamod`` family, which reuses
``_DM``/``_DC`` from here (see docs/kernels/rmsnorm-adamod.md).

Both values of ``HAS_WEIGHT`` are driven: it is in the autotune key, so each is a separate
compiled kernel and cache bucket, and both are real -- SWA's q/k normalization is non-affine
(module.py:433) and triangle_attention passes a weight.
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
#: Heads the SWA atom block runs (`modules/swa_atom_attention/module.py:457`,
#: ``head_dim = d_model // n_heads``); d_model 128 over 4 of them is head_dim 32. The RATIO is
#: what follows the swept base, which is why it is here and not a width of its own.
_N_HEADS = 4
#: head_dim, the axis ``triton_rmsnorm`` reduces -- not d_model, which is what the DiT-block
#: entry points normalize over.
_D = ragged(_D_BASE // _N_HEADS)
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


# ── rmsnorm ──────────────────────────────────────────────────────────────────────────────────


def rmsnorm_fwd_triton() -> None:
    """rmsnorm_fwd_kernel, at both values of HAS_WEIGHT.

    Two cache buckets -- the flag is in the autotune key -- so both are launched. The modulate
    form (adaLN's `rmsnorm(x)*(1+scale)+shift`) is NOT here: it lives in the `rmsnorm_adamod`
    family, which folds the conditioning projection in too. See docs/kernels/rmsnorm-adamod.md.
    """
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm

    triton_rmsnorm(_rows(_D), vec(_D), _EPS)   # triangle_attention: affine
    triton_rmsnorm(_rows(_D), None, _EPS)      # SWA q/k: non-affine


def rmsnorm_bwd_triton() -> None:
    """rmsnorm_bwd_kernel, via the autograd entry point, at both values of HAS_WEIGHT."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm

    triton_rmsnorm(_rows(_D), vec(_D), _EPS).sum().backward()
    triton_rmsnorm(_rows(_D), None, _EPS).sum().backward()

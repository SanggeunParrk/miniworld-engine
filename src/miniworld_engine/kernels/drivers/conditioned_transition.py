"""Drivers for the ``conditioned_transition`` family -- and the shape block ``adaln`` imports.

The two families were one module (``drivers_adaln.py``). They still share one set of extents
(``_D_BASE``/``_M``/``_D``/``_DC``/``_SHAPE_KEY``, and ``drivers._rand``/``drivers.FP32``); the
block lives here because conditioned_transition has twice as many kernels reading it.
``drivers/adaln.py`` imports it from here. Everything below applies to both families.

One argument-free function per registry kernel; each calls that kernel's own launcher (the host
function in the kernel's file that does ``kernel[grid](...)``) so the argument list is the one the
repo already writes, not one invented here. Numerics are the bench's job -- these only have to
reach the kernel.

Shapes. ``d = 128`` is the repo's default pair width (``benchmarks/runners/bench.py``'s
``BenchConfig.d_pair = 128``, which is what ``bench_kernel_adaln`` /
``bench_kernel_cond_transition_tail`` build their modules with) and it is the "atom stream" width
the fused paths in both families are written around (``interface.py::ATOM_D_MAX = 128``).
``M = 512`` is ``drivers.rows2d``'s default row count -- the bench's own M is ``L*L >= 147456``,
far more than a reach-the-kernel launch needs. Those are the ALIGNED-mode values; every extent
goes through ``drivers.ragged()`` so ``MINIWORLD_SHAPE_MODE=ragged`` makes each axis' last tile
partial (see the constant block below for which axis each one moves).

shape_key. Every kernel here has ``shape_key`` in its ``key=[...]`` and every registry row this
file drives is ``level=atom``, so the bucket is ``atom_key(L)`` with L the ATOM count. Two layouts
reach the kernels and they need opposite treatment. The OUTER entry points
(``adaln_inference_fused``, ``triton_adaptive_layer_norm``, ``cond_transition_inference``) flatten
the activation themselves and read the pre-flatten shape (or take ``length=``) for the key, so they
are handed the ``(1, M, D)`` activation production passes -- a 2-D ``(M, D)`` is exactly what
``length_of`` refuses, since shape[-2] of an already-flattened matrix is M and not L. Every other
launcher reached from here is an INNER one: it unpacks ``M, N = t.shape`` and therefore can only be
given the flat matrix, so per ``length_of``'s docstring ("its caller must compute the key and pass
it down") this file computes ``_SHAPE_KEY`` once and passes it as ``shape_key=``.

dtypes. adaLN runs bf16 (``drivers.BF16``, and ``bench_kernel_adaln`` uses bf16 unless the sweep
asks for fp32). The conditioned_transition family runs fp32: every file in it states "fp32 io with
TF32 tensor cores" and ``bench_kernel_cond_transition_tail`` is fp32-only. The one exception is
``gated_projection_bwd_gate_flat_lowp_triton``, whose registry name says ``lowp`` -- bf16 there.
"""
from __future__ import annotations

import torch

from miniworld_engine.autotune.shape_key import atom_key
from miniworld_engine.kernels.drivers import FP32, _rand, driver_length, ragged

# Tile alignment (see drivers.py). Every extent below is an aligned base run through
# ``ragged()``: unchanged by default, minus a small odd amount under MINIWORLD_SHAPE_MODE=ragged,
# which puts a partial tile at the end of that axis for all five config sets at once.
#
# The four extents are four SEPARATE axes, perturbed independently rather than by one shared
# constant, because the kernels here tile them with different BLOCK_* knobs:
#   _M   row axis        (BLOCK_M1 / BLOCK_E)         512 -> 509
#   _D   d_hidden = NX, and K/D in the tail          (BLOCK_K_NX, BLOCK_N, ...)  128 -> 125
#   _DC  d_cond   = NC/DC                            (BLOCK_K_NC, BLOCK_K_DC)    128 -> 123
#   _ND  expand width                                (BLOCK_K_ND / BLOCK_N)      512 -> 509
# _DC uses by=5 rather than by=3 so the two hidden widths are DIFFERENT non-aligned values: a
# mask bug on the d_cond axis is invisible while d_cond == d_hidden, and so is a launcher that
# reads one width where it means the other. 5 has the same property that motivates 3 -- odd,
# below the smallest tile (16), a divisor of none of 16/32/64/128 -- so 123 is a partial tile in
# every config set too.
# _ND is perturbed on its OWN axis instead of inheriting 4*_D (which would also be non-aligned,
# 500): the tail kernels take ND from wa.shape[0] and D from ws.shape[0] with no relation between
# them, and an independent 509 keeps ND's tile remainder (61) from coinciding with _D's, so a mask
# bug that only shows when the two axes are congruent mod BLOCK cannot hide. _N_EXPAND is
# therefore only the aligned-mode base of _ND, not a live multiplier.
_D_BASE = 128  # BenchConfig.d_pair default / interface.ATOM_D_MAX
_N_EXPAND = 4  # ConditionedTransition(D, D, n=4), as bench_kernel_cond_transition_tail builds it

_M = ragged(driver_length(512))       # drivers.rows2d default row count
_D = ragged(_D_BASE)                  # d_hidden (NX) / the tail's K and D
_DC = ragged(_D_BASE, by=5)           # d_cond (NC / DC) -- separately tiled axis
_ND = ragged(_N_EXPAND * _D_BASE)     # expand width (ND)

# The autotune SHAPE bucket for every inner launcher below (see the docstring). L is ``_M``: the
# driver builds one batch, so its rows ARE the atoms, and ``atom_key(_M)`` is the same value the
# outer entry points compute from the ``(1, _M, D)`` activation via ``length_of`` -- ragged mode
# included, where both sides see _M = 509. Derived from _M rather than written out, so
# MINIWORLD_DRIVER_LENGTH moves the recorded bucket with the shape.
_SHAPE_KEY = atom_key(_M)


# ── conditioned_transition ────────────────────────────────────────────────────────────────────
# Post-adaLN tail of ConditionedTransition(D, D, n=4):
#   x (M,K=D) cond (M,DC) | Wa,Wb (ND,K) | Ws (D,ND) | Wsc (D,DC) | bsc (D,)
# K/D come from _D and DC from _DC: the module ties them (d_in == d_cond == d_out) but the
# launchers read each from its own tensor and tile them with different BLOCK_* knobs, so the
# drivers keep them as two axes.  ND is _ND, its own axis.  See the constant block above.


def _ct_args(dtype=FP32):
    """(x, cond, wa, wb, ws, wsc, bsc) -- the signature every tail entry point takes."""
    return (_rand(_M, _D, dtype=dtype), _rand(_M, _DC, dtype=dtype),
            _rand(_ND, _D, dtype=dtype), _rand(_ND, _D, dtype=dtype),
            _rand(_D, _ND, dtype=dtype), _rand(_D, _DC, dtype=dtype), _rand(_D, dtype=dtype))


def cond_transition_fwd_b2b():
    """inference._cond_transition_inference_kernel -- the fully fused b2b inference path."""
    from miniworld_engine.kernels.conditioned_transition.triton.inference import (
        cond_transition_inference,
    )

    # OUTER entry point: takes the flat matrix but names L itself, via ``length=``.
    cond_transition_inference(*_ct_args(), length=_M)


def cond_transition_expand_swiglu():
    """composed._expand_swiglu_kernel: h = silu(x@Waᵀ)*(x@Wbᵀ)."""
    from miniworld_engine.kernels.conditioned_transition.triton.composed import (
        _expand_swiglu,
    )

    x, _, wa, wb, *_ = _ct_args()
    _expand_swiglu(x, wa, wb, shape_key=_SHAPE_KEY)


def cond_transition_squeeze_gate():
    """composed._squeeze_gate_kernel: y = sigmoid(cond@Wscᵀ+bsc)*(h@Wsᵀ); h is (M, ND)."""
    from miniworld_engine.kernels.conditioned_transition.triton.composed import (
        _squeeze_gate,
    )

    _, cond, _, _, ws, wsc, bsc = _ct_args()
    _squeeze_gate(_rand(_M, _ND, dtype=FP32), cond, ws, wsc, bsc, shape_key=_SHAPE_KEY)


def cond_transition_swiglu():
    """training._swiglu_fwd_kernel: h = silu(a)*b, a/b the (M, ND) expand halves."""
    from miniworld_engine.kernels.conditioned_transition.triton.training import _swiglu

    _swiglu(_rand(_M, _ND, dtype=FP32), _rand(_M, _ND, dtype=FP32), shape_key=_SHAPE_KEY)


def cond_transition_bwd_swiglu_flat():
    """training._swiglu_bwd_kernel via _swiglu_bwd_packed(a, b, dh) -> dab (M, 2ND)."""
    from miniworld_engine.kernels.conditioned_transition.triton.training import (
        _swiglu_bwd_packed,
    )

    _swiglu_bwd_packed(_rand(_M, _ND, dtype=FP32), _rand(_M, _ND, dtype=FP32),
                       _rand(_M, _ND, dtype=FP32), shape_key=_SHAPE_KEY)


def cond_transition_fwd_b2b_saveact():
    """training._b2b_fwd_train_kernel (atom fused b2b training forward).

    fp32: the ConditionedTransitionTailFunction forward reroutes bf16 away from this kernel with
    the comment "bf16 fused b2b train kernel is broken (dtype/spill)", so bf16 would only measure
    that known break.
    """
    from miniworld_engine.kernels.conditioned_transition.triton.training import (
        _b2b_fwd_train,
    )

    _b2b_fwd_train(*_ct_args(), shape_key=_SHAPE_KEY)


def cond_transition_expand_swiglu_saveact():
    """train_fused._fwd_expand_swiglu_kernel -> (h, ab=[a|b])."""
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import (
        _fwd_expand_swiglu,
    )

    x, _, wa, wb, *_ = _ct_args()
    _fwd_expand_swiglu(x, wa, wb, shape_key=_SHAPE_KEY)


def cond_transition_squeeze_gate_saveact():
    """train_fused._fwd_squeeze_gate_kernel -> (y, out, scale)."""
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import (
        _fwd_squeeze_gate,
    )

    _, cond, _, _, ws, wsc, bsc = _ct_args()
    _fwd_squeeze_gate(_rand(_M, _ND, dtype=FP32), cond, ws, wsc, bsc, shape_key=_SHAPE_KEY)



def cond_transition_bwd_gemm():
    """train_fused._dgemm_kernel as the backward calls it: dcond = dscale(M,D) @ Wsc(D,DC)."""
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import (
        _dgemm,
    )

    wsc = _rand(_D, _DC, dtype=FP32)
    _dgemm(_rand(_M, _D, dtype=FP32), wsc, _M, _DC, _D, wsc.stride(0), wsc.stride(1),
           shape_key=_SHAPE_KEY)


def cond_transition_bwd_swiglu_dx():
    """train_fused._dx_fused_kernel: dx = da@Wa + db@Wb, da/db recomputed from (dh, ab)."""
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import (
        _dx_fused,
    )

    _, _, wa, wb, *_ = _ct_args()
    _dx_fused(_rand(_M, _ND, dtype=FP32), _rand(_M, 2 * _ND, dtype=FP32), wa, wb,
              shape_key=_SHAPE_KEY)


def cond_transition_bwd_gate_squeeze_dx():
    """train_fused._dh_gatebwd_kernel: dh = (sigmoid(scale)*dy) @ Ws, out/scale/dy (M, D)."""
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import (
        _dh_gatebwd,
    )

    _, _, _, _, ws, *_ = _ct_args()
    _dh_gatebwd(_rand(_M, _D, dtype=FP32), _rand(_M, _D, dtype=FP32), _rand(_M, _D, dtype=FP32),
                ws, _ND, shape_key=_SHAPE_KEY)


def cond_transition_bwd_swiglu_dx_packed():
    """train_fused._dx_swiglubwd_kernel: dx = dab @ Wcat, Wcat = cat([Wa, Wb]) (2ND, K)."""
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import (
        _dx_swiglubwd,
    )

    _, _, wa, wb, *_ = _ct_args()
    _dx_swiglubwd(_rand(_M, _ND, dtype=FP32), _rand(_M, 2 * _ND, dtype=FP32),
                  torch.cat([wa, wb], dim=0), shape_key=_SHAPE_KEY)


def cond_transition_bwd_swiglu_packed():
    """train_fused._swiglu_bwd_pack_kernel: dab = [da|db] from (dh (M,ND), ab (M,2ND))."""
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import (
        _swiglu_bwd_pack,
    )

    _swiglu_bwd_pack(_rand(_M, _ND, dtype=FP32), _rand(_M, 2 * _ND, dtype=FP32),
                     shape_key=_SHAPE_KEY)


def cond_transition_bwd_dw():
    """train_fused._wgrad_kernel as the backward's dWs would use it: dWs(D,ND) = dout(M,D)ᵀ @ h(M,ND)."""
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import (
        _wgrad,
    )

    _wgrad(_rand(_M, _D, dtype=FP32), _rand(_M, _ND, dtype=FP32), _D, _ND,
           shape_key=_SHAPE_KEY)

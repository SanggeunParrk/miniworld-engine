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

dtypes. Every activation here is built at ``drivers.BF16``, which is the name
``MINIWORLD_DRIVER_DTYPE`` switches -- so a unit declared bf16 and a unit declared fp32 build
different tensors, which is the whole point of declaring two. It used to be the fixed
``drivers.FP32`` at every site, matching "fp32 io with TF32 tensor cores" in the family's files;
that made fp32 the only precision this family COULD be built at, and registry.csv declared fp32
alone to match. Both halves were true and together they meant the model's own precision was never
tuned: krystal runs bf16. The one kernel that must stay fp32 says so at its own call site
(``cond_transition_fwd_b2b_saveact``), by naming torch.float32 rather than by pinning the family.
"""
from __future__ import annotations

from miniworld_engine.autotune.shape_key import atom_key
from miniworld_engine.kernels.drivers import (
    BF16,
    _rand,
    driver_length,
    driver_width,
    ragged,
)

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
_D_BASE = driver_width(128)  # BenchConfig.d_pair default / interface.ATOM_D_MAX
# ConditionedTransition(d_hidden=128, d_cond=384, n=2) -- module.py's own defaults, NOT the bench's.
# This used to be 4, "as bench_kernel_cond_transition_tail builds it", and the two disagree: the
# module's n is 2, so production asks for ND = 2*d_hidden while every cached entry said ND = 4*d.
# ND is folded into the shape key, so those are different buckets and the family's 84 entries never
# matched a single production launch -- it missed and fell back to the heuristic subset, every time.
# A driver's extents are a claim about what the model runs; when they are copied from a bench
# instead, the cache is tuned for the bench.
_N_EXPAND = 2
#: d_cond FOLLOWS d_hidden the way the model pairs them, instead of being a constant. krystal
#: builds exactly two combinations and every model config (debug/small/medium/large) declares the
#: same widths for them: `token_dit` is d_single=768 conditioned on d_cond=384 (AlphaFold-3's
#: c_token and c_s) and `atom_dit` is 128 conditioned on 128 (c_atom for both). So d_cond is 384
#: whenever d_hidden is a token width and 128 on the atom side, and a driver that pinned it to one
#: number tuned a d_cond the other side never presents. The two stay SEPARATE axes -- separately
#: tiled as NC / DC -- because on the token side they are unequal.
_DC_BASE = 384 if _D_BASE > 128 else 128

_M = ragged(driver_length(512))       # drivers.rows2d default row count
_D = ragged(_D_BASE)                  # d_hidden (NX) / the tail's K and D
_DC = ragged(_DC_BASE, by=5)          # d_cond (NC / DC) -- separately tiled axis
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


def _ct_args(dtype=BF16):
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
    _squeeze_gate(_rand(_M, _ND, dtype=BF16), cond, ws, wsc, bsc, shape_key=_SHAPE_KEY)


def cond_transition_swiglu():
    """training._swiglu_fwd_kernel: h = silu(a)*b, a/b the (M, ND) expand halves."""
    from miniworld_engine.kernels.conditioned_transition.triton.training import _swiglu

    _swiglu(_rand(_M, _ND, dtype=BF16), _rand(_M, _ND, dtype=BF16), shape_key=_SHAPE_KEY)


def cond_transition_bwd_swiglu_flat():
    """training._swiglu_bwd_kernel via _swiglu_bwd_packed(a, b, dh) -> dab (M, 2ND)."""
    from miniworld_engine.kernels.conditioned_transition.triton.training import (
        _swiglu_bwd_packed,
    )

    _swiglu_bwd_packed(_rand(_M, _ND, dtype=BF16), _rand(_M, _ND, dtype=BF16),
                       _rand(_M, _ND, dtype=BF16), shape_key=_SHAPE_KEY)


def cond_transition_bwd_gemm_swiglu():
    """training._dh_swiglu_bwd_kernel: dh = dout @ ws fused into the SwiGLU backward.

    ``ws`` is (D, ND) -- the squeeze weight as the kernel's B operand, not transposed. The
    driver hands the same shapes the backward does, so the K extent tuned here is D and not ND.
    """
    from miniworld_engine.kernels.conditioned_transition.triton.training import (
        _dh_swiglu_bwd,
    )

    _dh_swiglu_bwd(_rand(_M, _D, dtype=BF16), _rand(_D, _ND, dtype=BF16),
                   _rand(_M, _ND, dtype=BF16), _rand(_M, _ND, dtype=BF16),
                   shape_key=_SHAPE_KEY)


def cond_transition_fwd_b2b_saveact():
    """training._b2b_fwd_train_kernel (atom fused b2b training forward).

    Built at the switchable `BF16` name like the rest of the family, and it was not always.

    It used to name `torch.float32` outright, because ConditionedTransitionTailFunction reroutes
    bf16 away from this kernel and a bf16 unit "would only measure that known break". Half of that
    break is gone: the reroute read "broken (dtype/spill)", the dtype half was `tl.dot` handed an
    fp32 accumulator beside a bf16 weight so the kernel did not COMPILE at bf16, and that is fixed
    -- the checker measures 3.13e-03 at bf16, the same as the inference twin the model already runs
    there. What is left is the SPILL half, and `training.py` says outright that lifting the reroute
    "needs a measurement, not an argument".

    A pinned driver is what made that measurement impossible to take. This function calls
    `_b2b_fwd_train` directly, below the autograd Function, so the reroute does not apply here and
    a bf16 unit measures the kernel rather than the detour. Pinning it to fp32 meant the one thing
    that could settle the question could never be produced -- the reroute justified the pin and the
    pin protected the reroute. The reroute itself stays in production until the numbers are in.
    """
    from miniworld_engine.kernels.conditioned_transition.triton.training import (
        _b2b_fwd_train,
    )

    _b2b_fwd_train(*_ct_args(dtype=BF16), shape_key=_SHAPE_KEY)


def cond_transition_expand_swiglu_saveact():
    """fwd_saveact._fwd_expand_swiglu_kernel -> (h, ab=[a|b])."""
    from miniworld_engine.kernels.conditioned_transition.triton.fwd_saveact import (
        _fwd_expand_swiglu,
    )

    x, _, wa, wb, *_ = _ct_args()
    _fwd_expand_swiglu(x, wa, wb, shape_key=_SHAPE_KEY)


def cond_transition_squeeze_gate_saveact():
    """fwd_saveact._fwd_squeeze_gate_kernel -> (y, out, scale)."""
    from miniworld_engine.kernels.conditioned_transition.triton.fwd_saveact import (
        _fwd_squeeze_gate,
    )

    _, cond, _, _, ws, wsc, bsc = _ct_args()
    _fwd_squeeze_gate(_rand(_M, _ND, dtype=BF16), cond, ws, wsc, bsc, shape_key=_SHAPE_KEY)




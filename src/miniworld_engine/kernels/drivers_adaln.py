"""Drivers for the ``adaln`` and ``conditioned_transition`` families.

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

from .drivers import BF16, dev, driver_length, ragged

FP32 = torch.float32

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
_EPS = 1e-5   # nn.LayerNorm default; modules/adaptive_layernorm/module.py takes the default

# The autotune SHAPE bucket for every inner launcher below (see the docstring). L is ``_M``: the
# driver builds one batch, so its rows ARE the atoms, and ``atom_key(_M)`` is the same value the
# outer entry points compute from the ``(1, _M, D)`` activation via ``length_of`` -- ragged mode
# included, where both sides see _M = 509. Derived from _M rather than written out, so
# MINIWORLD_DRIVER_LENGTH moves the recorded bucket with the shape.
_SHAPE_KEY = atom_key(_M)


def _rand(*shape, dtype=BF16):
    return torch.randn(*shape, device=dev(), dtype=dtype)


# ── adaLN ─────────────────────────────────────────────────────────────────────────────────────
# Parameter set of modules/adaptive_layernorm/module.py::AdaptiveLayerNorm(d_hidden, d_cond):
#   ln_cond.weight (NC,) | to_scale.weight (NX, NC) + bias (NX,) | to_bias.weight (NX, NC)


def _adaln_args(m: int = _M, nx: int = _D, nc: int = _DC, dtype=BF16, *, batched: bool = False):
    """(x, cond, lnw, Ws, scale_b, Wb) -- the 6 tensor args every adaLN entry point takes.

    ``batched`` picks x/cond's layout, which is not cosmetic: it is what the entry point can accept.
    The OUTER entry points reshape x/cond themselves and take the autotune key from the pre-flatten
    shape, so they need the ``(1, M, D)`` activation production hands them. The INNER launchers
    unpack ``M, N = t.shape`` and can only take the flat ``(M, D)``; those get ``shape_key=``
    instead. The four weights are weights, not activations -- their rank never changes.
    """
    lead = (1,) if batched else ()
    return (_rand(*lead, m, nx, dtype=dtype), _rand(*lead, m, nc, dtype=dtype),
            _rand(nc, dtype=dtype),
            _rand(nx, nc, dtype=dtype), _rand(nx, dtype=dtype), _rand(nx, nc, dtype=dtype))


def layernorm_fwd_strided():
    """fused3._ln_kernel via inference.py's weighted-LN launcher (HAS_W=True)."""
    from .adaln.triton.inference import _cond_affine

    _cond_affine(_rand(_M, _DC), _rand(_DC), _EPS, shape_key=_SHAPE_KEY)


def adaln_fwd():
    """inference._adaln_fused_kernel -- the single-kernel inference path (small d)."""
    from .adaln.triton.inference import adaln_inference_fused

    # OUTER entry point: it does the reshape and reads x's pre-flatten shape for the key.
    x, cond, lnw, ws, sb, wb = _adaln_args(batched=True)
    adaln_inference_fused(x, cond, lnw, ws, sb, wb, _EPS, _EPS)


def adaln_epilogue():
    """inference._adaln_epilogue_kernel. SB is (M, 2N) = [scale | bias] (kernel comment)."""
    from .adaln.triton.inference import _adaln_epilogue

    _adaln_epilogue(_rand(_M, _D), _rand(_M, 2 * _D), _EPS, shape_key=_SHAPE_KEY)


def adaln_gemm_gate():
    """fused3._gemm_gate_kernel, both settings of the SAVE_GATE constexpr (it is in the autotune
    key, so the two launchers are two distinct compiles): _gemm_gate is False, _gemm_gate_train
    True."""
    from .adaln.triton.fused3 import _gemm_gate, _gemm_gate_train

    x_norm, cond_norm, _, ws, sb, wb = _adaln_args()
    _gemm_gate(x_norm, cond_norm, ws, wb, sb, shape_key=_SHAPE_KEY)
    _gemm_gate_train(x_norm, cond_norm, ws, wb, sb, shape_key=_SHAPE_KEY)


def adaln_bwd_pre():
    """fused3._bwd_elem_kernel: (dy, x_norm, gate), all (M, N) in x's dtype (see _Fused3TrainFn:
    gate comes from _gemm_gate_train as torch.empty_like(x_norm))."""
    from .adaln.triton.fused3 import _bwd_elem

    _bwd_elem(_rand(_M, _D), _rand(_M, _D), _rand(_M, _D), shape_key=_SHAPE_KEY)


def adaln_epilogue_saveact():
    """training._epilogue_train_kernel: x (M,N), sb (M,2N) raw [scale|bias], scale_bias (N,)
    folded in (HAS_SB=True)."""
    from .adaln.triton.training import _epilogue_train

    _epilogue_train(_rand(_M, _D), _rand(_M, 2 * _D), _EPS, _rand(_D), shape_key=_SHAPE_KEY)


def adaln_bwd_pre_dx():
    """training._bwd_x_kernel: (dy, x, mean_x, rstd_x, gate). mean/rstd are fp32 (M,) --
    te_style._ln_materialize and _epilogue_train both allocate them that way."""
    from .adaln.triton.training import _bwd_x

    stat = torch.empty(_M, device=dev(), dtype=FP32).fill_(1.0)
    _bwd_x(_rand(_M, _D), _rand(_M, _D), stat, stat.clone(), _rand(_M, _D),
           shape_key=_SHAPE_KEY)


def adaln_bwd_dx_dlnw():
    """training._dgrad_condln_kernel. Launcher docstring: D=(2NX, M), w_cat=(2NX, NC); cond is the
    contiguous (M, NC) input, mean_c/rstd_c fp32 (M,), lnw (NC,)."""
    from .adaln.triton.training import _dgrad_condln

    stat = torch.empty(_M, device=dev(), dtype=FP32).fill_(1.0)
    _dgrad_condln(_rand(2 * _D, _M), _rand(2 * _D, _DC), _rand(_M, _DC),
                  stat, stat.clone(), _rand(_DC), shape_key=_SHAPE_KEY)


# main.py's four kernels are launched only by TritonAdaptiveLayerNormFunction: the forward kernel
# by forward(), and all three backward kernels together by backward(). Driving the autograd
# Function is what reaches them with the tensors the forward actually saved (x_hat/cond_norm/gate
# are fp32 or gate-dtype buffers it allocates itself).


def _adaln_main(*, backward: bool):
    from .adaln.triton.main import triton_adaptive_layer_norm

    # OUTER entry point: TritonAdaptiveLayerNormFunction reshapes x/cond and keys both the
    # forward and (via ctx.orig_x_shape) all three backward kernels off the pre-flatten shape.
    x, cond, lnw, ws, sb, wb = _adaln_args(batched=True)
    if backward:
        x.requires_grad_(True)
    y = triton_adaptive_layer_norm(x, cond, lnw, ws, sb, wb, _EPS, _EPS)
    if backward:
        y.backward(torch.randn_like(y))


def adaln_fwd_saveact():
    """main.adaln_fwd_kernel (forward only)."""
    _adaln_main(backward=False)


def adaln_bwd_dx_dbias():
    """main.adaln_bwd_input_kernel (via the autograd backward)."""
    _adaln_main(backward=True)


def adaln_bwd_dw():
    """main.adaln_bwd_weight_kernel (via the autograd backward)."""
    _adaln_main(backward=True)


def adaln_bwd_dlnw():
    """main.adaln_bwd_lnw_kernel (via the autograd backward)."""
    _adaln_main(backward=True)


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
    from .conditioned_transition.triton.inference import cond_transition_inference

    # OUTER entry point: takes the flat matrix but names L itself, via ``length=``.
    cond_transition_inference(*_ct_args(), length=_M)


def cond_transition_expand_swiglu():
    """composed._expand_swiglu_kernel: h = silu(x@Waᵀ)*(x@Wbᵀ)."""
    from .conditioned_transition.triton.composed import _expand_swiglu

    x, _, wa, wb, *_ = _ct_args()
    _expand_swiglu(x, wa, wb, shape_key=_SHAPE_KEY)


def cond_transition_squeeze_gate():
    """composed._squeeze_gate_kernel: y = sigmoid(cond@Wscᵀ+bsc)*(h@Wsᵀ); h is (M, ND)."""
    from .conditioned_transition.triton.composed import _squeeze_gate

    _, cond, _, _, ws, wsc, bsc = _ct_args()
    _squeeze_gate(_rand(_M, _ND, dtype=FP32), cond, ws, wsc, bsc, shape_key=_SHAPE_KEY)


def cond_transition_swiglu():
    """training._swiglu_fwd_kernel: h = silu(a)*b, a/b the (M, ND) expand halves."""
    from .conditioned_transition.triton.training import _swiglu

    _swiglu(_rand(_M, _ND, dtype=FP32), _rand(_M, _ND, dtype=FP32), shape_key=_SHAPE_KEY)


def cond_transition_bwd_swiglu_flat():
    """training._swiglu_bwd_kernel via _swiglu_bwd_packed(a, b, dh) -> dab (M, 2ND)."""
    from .conditioned_transition.triton.training import _swiglu_bwd_packed

    _swiglu_bwd_packed(_rand(_M, _ND, dtype=FP32), _rand(_M, _ND, dtype=FP32),
                       _rand(_M, _ND, dtype=FP32), shape_key=_SHAPE_KEY)


def cond_transition_fwd_b2b_saveact():
    """training._b2b_fwd_train_kernel (atom fused b2b training forward).

    fp32: the ConditionedTransitionTailFunction forward reroutes bf16 away from this kernel with
    the comment "bf16 fused b2b train kernel is broken (dtype/spill)", so bf16 would only measure
    that known break.
    """
    from .conditioned_transition.triton.training import _b2b_fwd_train

    _b2b_fwd_train(*_ct_args(), shape_key=_SHAPE_KEY)


def cond_transition_expand_swiglu_saveact():
    """train_fused._fwd_expand_swiglu_kernel -> (h, ab=[a|b])."""
    from .conditioned_transition.triton.train_fused import _fwd_expand_swiglu

    x, _, wa, wb, *_ = _ct_args()
    _fwd_expand_swiglu(x, wa, wb, shape_key=_SHAPE_KEY)


def cond_transition_squeeze_gate_saveact():
    """train_fused._fwd_squeeze_gate_kernel -> (y, out, scale)."""
    from .conditioned_transition.triton.train_fused import _fwd_squeeze_gate

    _, cond, _, _, ws, wsc, bsc = _ct_args()
    _fwd_squeeze_gate(_rand(_M, _ND, dtype=FP32), cond, ws, wsc, bsc, shape_key=_SHAPE_KEY)



def cond_transition_bwd_gemm():
    """train_fused._dgemm_kernel as the backward calls it: dcond = dscale(M,D) @ Wsc(D,DC)."""
    from .conditioned_transition.triton.train_fused import _dgemm

    wsc = _rand(_D, _DC, dtype=FP32)
    _dgemm(_rand(_M, _D, dtype=FP32), wsc, _M, _DC, _D, wsc.stride(0), wsc.stride(1),
           shape_key=_SHAPE_KEY)


def cond_transition_bwd_swiglu_dx():
    """train_fused._dx_fused_kernel: dx = da@Wa + db@Wb, da/db recomputed from (dh, ab)."""
    from .conditioned_transition.triton.train_fused import _dx_fused

    _, _, wa, wb, *_ = _ct_args()
    _dx_fused(_rand(_M, _ND, dtype=FP32), _rand(_M, 2 * _ND, dtype=FP32), wa, wb,
              shape_key=_SHAPE_KEY)


def cond_transition_bwd_gate_squeeze_dx():
    """train_fused._dh_gatebwd_kernel: dh = (sigmoid(scale)*dy) @ Ws, out/scale/dy (M, D)."""
    from .conditioned_transition.triton.train_fused import _dh_gatebwd

    _, _, _, _, ws, *_ = _ct_args()
    _dh_gatebwd(_rand(_M, _D, dtype=FP32), _rand(_M, _D, dtype=FP32), _rand(_M, _D, dtype=FP32),
                ws, _ND, shape_key=_SHAPE_KEY)


def cond_transition_bwd_swiglu_dx_packed():
    """train_fused._dx_swiglubwd_kernel: dx = dab @ Wcat, Wcat = cat([Wa, Wb]) (2ND, K)."""
    from .conditioned_transition.triton.train_fused import _dx_swiglubwd

    _, _, wa, wb, *_ = _ct_args()
    _dx_swiglubwd(_rand(_M, _ND, dtype=FP32), _rand(_M, 2 * _ND, dtype=FP32),
                  torch.cat([wa, wb], dim=0), shape_key=_SHAPE_KEY)


def cond_transition_bwd_swiglu_packed():
    """train_fused._swiglu_bwd_pack_kernel: dab = [da|db] from (dh (M,ND), ab (M,2ND))."""
    from .conditioned_transition.triton.train_fused import _swiglu_bwd_pack

    _swiglu_bwd_pack(_rand(_M, _ND, dtype=FP32), _rand(_M, 2 * _ND, dtype=FP32),
                     shape_key=_SHAPE_KEY)


def cond_transition_bwd_dw():
    """train_fused._wgrad_kernel as the backward's dWs would use it: dWs(D,ND) = dout(M,D)ᵀ @ h(M,ND)."""
    from .conditioned_transition.triton.train_fused import _wgrad

    _wgrad(_rand(_M, _D, dtype=FP32), _rand(_M, _ND, dtype=FP32), _D, _ND,
           shape_key=_SHAPE_KEY)

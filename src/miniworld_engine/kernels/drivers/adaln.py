"""Drivers for the ``adaln`` family.

adaLN and its ``conditioned_transition`` tail were one module (``drivers_adaln.py``) and still
share one set of extents. That block -- ``_M``/``_D``/``_DC``/``_SHAPE_KEY`` -- and the shape,
dtype and shape_key rationale behind it live in ``drivers/conditioned_transition.py``, which is
the family with the most kernels reading it; these drivers import it from there.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, FP32, _rand, dev
from miniworld_engine.kernels.drivers.conditioned_transition import (
    _D,
    _DC,
    _M,
    _SHAPE_KEY,
)

_EPS = 1e-5   # nn.LayerNorm default; modules/adaptive_layernorm/module.py takes the default


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
    from miniworld_engine.kernels.adaln.triton.inference import _cond_affine

    _cond_affine(_rand(_M, _DC), _rand(_DC), _EPS, shape_key=_SHAPE_KEY)


def adaln_fwd():
    """inference._adaln_fused_kernel -- the single-kernel inference path (small d)."""
    from miniworld_engine.kernels.adaln.triton.inference import adaln_inference_fused

    # OUTER entry point: it does the reshape and reads x's pre-flatten shape for the key.
    x, cond, lnw, ws, sb, wb = _adaln_args(batched=True)
    adaln_inference_fused(x, cond, lnw, ws, sb, wb, _EPS, _EPS)


def adaln_epilogue():
    """inference._adaln_epilogue_kernel. SB is (M, 2N) = [scale | bias] (kernel comment)."""
    from miniworld_engine.kernels.adaln.triton.inference import _adaln_epilogue

    _adaln_epilogue(_rand(_M, _D), _rand(_M, 2 * _D), _EPS, shape_key=_SHAPE_KEY)


def adaln_gemm_gate():
    """inference._adaln_gemm_gate_kernel. cond ALREADY normalised, weights ALREADY (NC, NX).

    Both are the caller's job, so the driver hands them the way the dispatch does. Passing
    (NX, NC) here would tune a strided K axis the real call never presents, and passing raw cond
    would tune a normalisation the kernel does not do.
    """
    from miniworld_engine.kernels.adaln.triton.inference import _adaln_gemm_gate

    stat = torch.empty(_M, device=dev(), dtype=FP32).fill_(1.0)
    _adaln_gemm_gate(_rand(_M, _DC), _rand(_DC, _D), _rand(_D), _rand(_DC, _D),
                     _rand(_M, _D), stat, stat.clone(), shape_key=_SHAPE_KEY)


def adaln_epilogue_saveact():
    """training._epilogue_train_kernel: x (M,N), sb (M,2N) raw [scale|bias], scale_bias (N,)
    folded in (HAS_SB=True)."""
    from miniworld_engine.kernels.adaln.triton.training import _epilogue_train

    _epilogue_train(_rand(_M, _D), _rand(_M, 2 * _D), _EPS, _rand(_D), shape_key=_SHAPE_KEY)


def adaln_bwd_pre_dx():
    """training._bwd_x_kernel: (dy, x, mean_x, rstd_x, gate). mean/rstd are fp32 (M,) --
    te_style._ln_materialize and _epilogue_train both allocate them that way."""
    from miniworld_engine.kernels.adaln.triton.training import _bwd_x

    stat = torch.empty(_M, device=dev(), dtype=FP32).fill_(1.0)
    _bwd_x(_rand(_M, _D), _rand(_M, _D), stat, stat.clone(), _rand(_M, _D),
           shape_key=_SHAPE_KEY)


def adaln_bwd_dx_dlnw():
    """training._dgrad_condln_kernel. Launcher docstring: D=(2NX, M), w_cat=(2NX, NC); cond is the
    contiguous (M, NC) input, mean_c/rstd_c fp32 (M,), lnw (NC,)."""
    from miniworld_engine.kernels.adaln.triton.training import _dgrad_condln

    stat = torch.empty(_M, device=dev(), dtype=FP32).fill_(1.0)
    _dgrad_condln(_rand(2 * _D, _M), _rand(2 * _D, _DC), _rand(_M, _DC),
                  stat, stat.clone(), _rand(_DC), shape_key=_SHAPE_KEY)


# main.py's four kernels are launched only by TritonAdaptiveLayerNormFunction: the forward kernel
# by forward(), and all three backward kernels together by backward(). Driving the autograd
# Function is what reaches them with the tensors the forward actually saved (x_hat/cond_norm/gate
# are fp32 or gate-dtype buffers it allocates itself).


def _adaln_main(*, backward: bool):
    from miniworld_engine.kernels.adaln.triton.main import triton_adaptive_layer_norm

    # OUTER entry point: TritonAdaptiveLayerNormFunction reshapes x/cond and keys both the
    # forward and (via ctx.orig_x_shape) all three backward kernels off the pre-flatten shape.
    x, cond, lnw, ws, sb, wb = _adaln_args(batched=True)
    if backward:
        x.requires_grad_(True)
    y = triton_adaptive_layer_norm(x, cond, lnw, ws, sb, wb, _EPS, _EPS)
    if backward:
        y.backward(torch.randn_like(y))



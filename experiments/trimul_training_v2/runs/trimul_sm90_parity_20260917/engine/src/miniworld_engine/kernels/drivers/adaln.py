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
    _L,
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
    shape, so they need the ``(B, L, D)`` activation production hands them. The INNER launchers
    unpack ``M, N = t.shape`` and can only take the flat ``(M, D)``; those get ``shape_key=``
    instead. The four weights are weights, not activations -- their rank never changes.

    THE BATCHED SHAPE IS (M // L, L, D), NOT (1, M, D). An outer entry point keys on
    ``length_of(x.shape)``, which is ``shape[-2]``, so a `(1, M, D)` activation tells it the length
    is M. Since `76daae51` made `_M = max(_L, 8192)` -- the ROWS to tune at, deliberately larger
    than the length -- that meant every unit of `adaln_fwd_triton`, at every L the sweep drives,
    recorded the single bucket `atom_key(8192)` and overwrote the previous one, while the buckets
    production actually looks up (128 through 4096) stayed empty forever. `dev audit` on the A6000
    reported it as 16 missing (dtype, bucket) pairs -- the largest hole on the card.

    Splitting the same M rows into `(M // L, L, D)` gives the entry point the length it keys on AND
    the row count the tuning needs, which is also exactly the shape production hands it: a batch of
    sequences, not one sequence of 8,192.
    """
    if not batched:
        return (_rand(m, nx, dtype=dtype), _rand(m, nc, dtype=dtype), _rand(nc, dtype=dtype),
                _rand(nx, nc, dtype=dtype), _rand(nx, dtype=dtype), _rand(nx, nc, dtype=dtype))
    length = min(_L, m)
    batch = max(1, m // length)
    return (_rand(batch, length, nx, dtype=dtype), _rand(batch, length, nc, dtype=dtype),
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


def adaln_fwd_gate():
    """training._adaln_fwd_gate_kernel: the fused training forward (y + gate), cond ALREADY
    normalised, weights ALREADY (NC, NX), x stats as (rstd, c1=mean*rstd). Same shapes the
    training forward hands it, so the tuned K extent is NC and not NX.
    """
    from miniworld_engine.kernels.adaln.triton.training import _adaln_fwd_gate

    stat = torch.empty(_M, device=dev(), dtype=FP32).fill_(1.0)
    _adaln_fwd_gate(_rand(_M, _DC), _rand(_DC, _D), _rand(_D), _rand(_DC, _D),
                    _rand(_M, _D), stat, stat.clone(), shape_key=_SHAPE_KEY)


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
    """training._epilogue_train_kernel: x (M,N), sb (M,2N) raw [scale|bias]. HAS_SB=1 ONLY.

    `HAS_SB = scale_bias is not None` (training.py:206) is keyed, but the sole production launcher
    (training.py:699) always passes `to_scale.bias`, which always exists:
    modules/adaptive_layernorm/module.py:42 builds the Linear and modules/primitives.py defaults
    `bias=True`. So =0 is unreachable, and replay asked for HAS_SB=1 nine times and =0 never."""
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

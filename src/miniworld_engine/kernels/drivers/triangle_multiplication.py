"""Drivers for the ``triangle_multiplication`` family.

Its three kernels live in ``modules/triangle_multiplication/baseline_dtv1.py``, not under
``kernels/`` -- registry.csv's ``family`` column is what puts them here. They shared a module
(``drivers_trans.py``) with the transition drivers and no code; the ``TRIMUL_*`` extents below
are their own, and the launcher/shape_key rationale they follow is written out in
``drivers/transition.py``.
"""
from __future__ import annotations

import torch

from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.drivers import (
    both_level_is_pair,
    dev,
    driver_length,
    ragged,
    rows2d,
)

# ------------------------------------------------------- triangle_multiplication (dt-v1)

TRIMUL_L = driver_length(128)  # L: i == j of the (1, L, L, d) pair activation the module flattens
#: Same pair/atom split as ROWS. Three token-level kernels share this constant and are never
#: driven above 512, so they keep the pair layout they had.
TRIMUL_IS_PAIR = both_level_is_pair(TRIMUL_L)
TRIMUL_ROWS = (ragged(TRIMUL_L) ** 2 if TRIMUL_IS_PAIR
               else ragged(TRIMUL_L))  # M = b*i*j for a (1, L, L, d) pair activation, or A
TRIMUL_D = ragged(128)  # d: the contraction width; the gate/proj weights have 2*d or d rows
#: ``baseline_dtv1._shape_key`` is ``token_key(seq_len)``, and ``seq_len=None`` -- what a driver that
#: passes nothing gets -- means ``token_key(0)``, the clamped BOTTOM bucket 128 at every L. These
#: launchers already take ``seq_len=``; the driver is the caller that has to supply it.
TRIMUL_SHAPE_KEY = token_key(TRIMUL_L)  # what ``seq_len=TRIMUL_L`` below makes them record


def trimul_gemm_gate_saveact_triton() -> None:
    """_input_gated_gemm_kernel via _input_gemm_fwd. w_gate/w_proj have 2*D rows and the
    public entry passes TRANSPOSE_OUT=True; mask=None (APPLY_MASK=False)."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import _input_gemm_fwd

    xn = rows2d(TRIMUL_ROWS, TRIMUL_D)
    _input_gemm_fwd(xn, rows2d(2 * TRIMUL_D, TRIMUL_D), rows2d(2 * TRIMUL_D, TRIMUL_D), None, True,
                    seq_len=TRIMUL_L)


def trimul_outproj_gemm_gate_saveact_triton() -> None:
    """_output_gated_gemm_kernel via _output_gemm_fwd: x_normed/x_out (M, D), weights (D, D)."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import _output_gemm_fwd

    xn = rows2d(TRIMUL_ROWS, TRIMUL_D)
    _output_gemm_fwd(xn, rows2d(TRIMUL_ROWS, TRIMUL_D),
                     rows2d(TRIMUL_D, TRIMUL_D), rows2d(TRIMUL_D, TRIMUL_D),
                     seq_len=TRIMUL_L)


def gated_projection_bwd_gate_recompute_flat_triton() -> None:
    """_gated_gemm_bwd_elemwise_kernel via _elemwise_bwd_combined: grad/ab are the (2D, M)
    transposed input-path activations, sig_m is fp32 (the launcher's dtype contract)."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import (
        _elemwise_bwd_combined,
    )

    n = 2 * TRIMUL_D
    sig_m = torch.rand(n, TRIMUL_ROWS, device=dev(), dtype=torch.float32)
    _elemwise_bwd_combined(rows2d(n, TRIMUL_ROWS), rows2d(n, TRIMUL_ROWS), sig_m,
                           seq_len=TRIMUL_L)

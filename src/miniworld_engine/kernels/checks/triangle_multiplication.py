"""Torch references for the ``triangle_multiplication`` family.

Its kernels live in ``modules/triangle_multiplication/baseline_dtv1.py``; registry.csv's
``family`` column is what puts their checkers here. They shared a module (``checks_trans.py``)
with the transition references, whose two rules -- same operands as the driver, and reference =
what the source computes -- apply here too and are written out in ``checks/transition.py``. The
gated-projection reference itself is ``checks._proj``: ``out = sigmoid(x@wg^T) * (x@wp^T)``.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _proj
from miniworld_engine.kernels.drivers import dev, rows2d
from miniworld_engine.kernels.drivers.triangle_multiplication import (
    TRIMUL_D,
    TRIMUL_ROWS,
)

# ------------------------------------------------------- triangle_multiplication (dt-v1)


def trimul_gemm_gate_saveact_triton():
    """Input gated GEMM: ``ab = sigmoid(xn@wg^T) * (xn@wp^T)`` plus the saved gate
    ``sig_m = sigmoid(xn@wg^T)`` (fp32, reused by the backward). The driver passes
    mask=None -> APPLY_MASK=False, so no row scale enters either output, and
    TRANSPOSE_OUT=True -> both outputs are written (N, M), hence the transposed reference."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import _input_gemm_fwd

    xn = rows2d(TRIMUL_ROWS, TRIMUL_D)
    wg, wp = rows2d(2 * TRIMUL_D, TRIMUL_D), rows2d(2 * TRIMUL_D, TRIMUL_D)
    ab, sig_m = _input_gemm_fwd(xn, wg, wp, None, True)

    gate, proj = _proj(xn, wg, wp)
    sig = torch.sigmoid(gate)
    return {"ab": (ab, (sig * proj).T), "sig_m": (sig_m, sig.T)}


def trimul_outproj_gemm_gate_saveact_triton():
    """Output gated GEMM: the gate reads x_normed and the projection reads x_out -- two
    DIFFERENT activations through two weights (baseline_dtv1.py:405-406), unlike the input
    path where both dots share one operand. Outputs stay (M, N); sig is fp32."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import _output_gemm_fwd

    xn = rows2d(TRIMUL_ROWS, TRIMUL_D)
    x_out = rows2d(TRIMUL_ROWS, TRIMUL_D)
    wg, wp = rows2d(TRIMUL_D, TRIMUL_D), rows2d(TRIMUL_D, TRIMUL_D)
    ab, sig = _output_gemm_fwd(xn, x_out, wg, wp)

    gate = xn.float() @ wg.float().T
    proj = x_out.float() @ wp.float().T
    s = torch.sigmoid(gate)
    return {"ab": (ab, s * proj), "sig": (sig, s)}


def gated_projection_bwd_gate_recompute_flat_triton():
    """Elementwise backward of the gated projection, from the SAVED sigmoid (so nothing is
    recomputed here): ``d_gate = grad*ab*(1-sig_m)``, ``d_proj = grad*sig_m``, both in fp32
    then cast to grad's dtype. The launcher packs them into one (2N, M) buffer, d_gate in
    the first N rows and d_proj in the last N (baseline_dtv1.py:490-512), so the two halves
    are checked separately -- a swapped pack would otherwise pass on symmetry alone.
    sig_m is drawn from U[0,1) because it is a stored sigmoid, and fp32 as the launcher
    requires."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import (
        _elemwise_bwd_combined,
    )

    n = 2 * TRIMUL_D
    grad, ab = rows2d(n, TRIMUL_ROWS), rows2d(n, TRIMUL_ROWS)
    sig_m = torch.rand(n, TRIMUL_ROWS, device=dev(), dtype=torch.float32)
    d_combined = _elemwise_bwd_combined(grad, ab, sig_m)

    gf = grad.float()
    return {
        "d_gate": (d_combined[:n], gf * ab.float() * (1.0 - sig_m)),
        "d_proj": (d_combined[n:], gf * sig_m),
    }

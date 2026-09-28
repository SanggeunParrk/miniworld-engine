"""Drivers for the ``tm1`` family.

trimul_inproj, tm1, tm2 and gated_projection were one module (``drivers_trimul.py``) and still
share ``D``/``L``/``IS_PAIR``/``M`` and the ``_x``/``_rows``/``_w``/``_bdll`` builders, which
live in ``drivers/trimul_inproj.py`` together with the shape and lazy-import rationale for all
four.
"""
from __future__ import annotations

from miniworld_engine.kernels.drivers.trimul_inproj import _w, _x

# ── tm1 ──────────────────────────────────────────────────────────────────────────────────────

def trimul_gemm_gate_triton() -> None:
    """fused_sigmoid_gate_fwd_kernel, via triton_tm1."""
    from miniworld_engine.kernels.tm1.triton.main import triton_tm1

    # _x(), not _rows(): TritonTM1Function reads token_key(length_of(x.shape)) BEFORE its own
    # rearrange to (M, d), so a pre-flattened (M, d) gives it M = L*L -> clamped to 512.
    triton_tm1(_x(), _w(), _w(), _w(), _w())


def trimul_bwd_gate_recompute_triton() -> None:
    """fused_sigmoid_gate_bwd_kernel, via TritonTM1Function.backward."""
    from miniworld_engine.kernels.tm1.triton.main import triton_tm1

    # _x(): same reason as the forward -- ctx.original_shape is what the backward keys on.
    x = _x().requires_grad_()
    left, right = triton_tm1(x, _w(), _w(), _w(), _w())
    (left.sum() + right.sum()).backward()

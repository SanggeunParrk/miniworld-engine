"""Drivers for the ``tm2`` family.

trimul_inproj, tm1, tm2 and gated_projection were one module (``drivers_trimul.py``) and still
share ``D``/``L``/``IS_PAIR``/``M`` and the ``_x``/``_rows``/``_w``/``_bdll`` builders, which
live in ``drivers/trimul_inproj.py`` together with the shape and lazy-import rationale for all
four.
"""
from __future__ import annotations

from miniworld_engine.kernels.drivers.trimul_inproj import _w, _x

# ── tm2 ──────────────────────────────────────────────────────────────────────────────────────

def trimul_outproj_gemm_gate_triton() -> None:
    """fused_sigmoid_gate2_fwd_kernel, via triton_tm2."""
    from miniworld_engine.kernels.tm2.triton.main import triton_tm2

    # _x(), not _rows(): TritonTM2Function reads token_key(length_of(x.shape)) before its own
    # rearrange, so a pre-flattened (M, d) gives it M = L*L -> clamped to 512.
    triton_tm2(_x(), _x(), _w(), _w())


def trimul_outproj_bwd_gate_recompute_triton() -> None:
    """fused_sigmoid_gate2_bwd_kernel, via TritonTM2Function.backward."""
    from miniworld_engine.kernels.tm2.triton.main import triton_tm2

    # _x(): same reason as the forward -- ctx.original_shape is what the backward keys on.
    x, y = _x().requires_grad_(), _x().requires_grad_()
    triton_tm2(x, y, _w(), _w()).sum().backward()


def tm2_dual_kernel() -> None:
    """Drive the same configurable/padded launcher used by production."""
    from miniworld_engine.kernels.tm2.cute.tm2_cute_kernel import tm2_dual_from_scratch
    tm2_dual_from_scratch(_x(), _x(), _w().t().contiguous(), _w().t().contiguous())

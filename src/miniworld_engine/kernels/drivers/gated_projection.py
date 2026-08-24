"""Drivers for the ``gated_projection`` family.

trimul_inproj, tm1, tm2 and gated_projection were one module (``drivers_trimul.py``) and still
share ``D``/``L``/``IS_PAIR``/``M`` and the ``_x``/``_rows``/``_w``/``_bdll`` builders, which
live in ``drivers/trimul_inproj.py`` together with the shape and lazy-import rationale for all
four.
"""
from __future__ import annotations

import torch
import triton

from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.drivers.trimul_inproj import D, M, _rows, _w, _x

# ── gated_projection/triton/main.py ──────────────────────────────────────────────────────────

def gated_projection_gate_triton() -> None:
    """sigmoid_gate_fwd_kernel, via TritonGatedProjectionFunction.

    ``_x()``, not ``_rows()``: the wrapper takes ``* hd`` and flattens to (M, hd) itself, and it
    reads ``both_key(rows_of(original_shape))`` from the shape it was HANDED. Handing it the
    already-flattened (M, D) makes length_of return M = L*L, which clamps to the top bucket 8192
    at every L >= 91 -- the launched shape is identical either way, only the key differs.
    """
    from miniworld_engine.kernels.gated_projection.triton.main import triton_gated_projection

    triton_gated_projection(_x(), _x(), _w())


def gated_projection_bwd_gate_triton() -> None:
    """sigmoid_gate_bwd_kernel, launched as TritonGatedProjectionFunction.backward launches it.

    The autograd path is not used: that backward returns ``.float()`` grads for bf16 inputs.
    """
    from miniworld_engine.kernels.gated_projection.triton.main import (
        sigmoid_gate_bwd_kernel,
    )

    gate, x, grad_out = _rows(), _rows(), _rows()
    dgate, dx = torch.empty_like(gate), torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731
    sigmoid_gate_bwd_kernel[grid](gate, x, grad_out, dgate, dx, gate.stride(0), x.stride(0),
                                  M, D, shape_key=both_key(M))


def gated_projection_gate_flat_triton() -> None:
    """_sigmul_fwd, via bias_only_attention's sigmoid_gate_fused (one of its two callers)."""
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import sigmoid_gate_fused

    sigmoid_gate_fused(_x(), _x())


def gated_projection_bwd_gate_flat_triton() -> None:
    """_sigmul_bwd, via _SigmoidGate.backward."""
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import sigmoid_gate_fused

    gate, out = _x().requires_grad_(), _x().requires_grad_()
    sigmoid_gate_fused(gate, out).sum().backward()

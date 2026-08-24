"""Accuracy checkers for the ``gated_projection`` family.

trimul_inproj, tm1, tm2 and gated_projection were one module (``checks_trimul.py``). The two
layout rules these follow -- return a dict with one entry per block, and check the kernel rather
than the cuBLAS GEMMs the launcher wraps it in -- plus the lazy-import rule, are written out in
``checks/trimul_inproj.py``. The shapes come from ``drivers/trimul_inproj.py`` and the shared
helpers from ``checks/__init__.py``.
"""
from __future__ import annotations

import torch
import triton

from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.checks import _f
from miniworld_engine.kernels.drivers.trimul_inproj import D, M, _rows, _x

# ── gated_projection/triton/main.py ──────────────────────────────────────────────────────────

def gated_projection_gate_triton():
    """sigmoid_gate_fwd_kernel: out = sigmoid(gate) * rep, row-tiled (M, R) with a column loop.

    Launched at TritonGatedProjectionFunction.forward's launch site. The public
    ``triton_gated_projection`` would append its ``@ out_weight`` cuBLAS GEMM to the result; the
    kernel is the sigmoid-multiply, so it is checked without the GEMM's error on top.
    """
    from miniworld_engine.kernels.gated_projection.triton.main import (
        sigmoid_gate_fwd_kernel,
    )

    gate, x = _rows(), _rows()
    out = torch.empty_like(x)
    grid = lambda meta: [triton.cdiv(M, meta["BLOCK_M1"])]
    sigmoid_gate_fwd_kernel[grid](gate, x, gate.stride(0), x.stride(0), out, M, D,
                                  shape_key=both_key(M))
    return out, torch.sigmoid(_f(gate)) * _f(x)


def gated_projection_bwd_gate_triton():
    """sigmoid_gate_bwd_kernel: d_rep = dy*s, d_gate = dy*rep*s*(1-s), s = sigmoid(gate).

    Same launch as ``drivers_trimul.gated_projection_bwd_gate_triton`` (the autograd backward
    returns ``.float()`` grads, so the kernel is launched directly there too).
    """
    from miniworld_engine.kernels.gated_projection.triton.main import (
        sigmoid_gate_bwd_kernel,
    )

    gate, x, grad_out = _rows(), _rows(), _rows()
    dgate, dx = torch.empty_like(gate), torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)
    sigmoid_gate_bwd_kernel[grid](gate, x, grad_out, dgate, dx, gate.stride(0), x.stride(0),
                                  M, D, shape_key=both_key(M))
    s = torch.sigmoid(_f(gate))
    dy = _f(grad_out)
    return {"dgate": (dgate, dy * _f(x) * s * (1.0 - s)),
            "drep": (dx, dy * s)}


def gated_projection_gate_flat_triton():
    """_sigmul_fwd: the flat (1-D, all-contiguous) form of the same y = sigmoid(g) * o."""
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import (
        sigmoid_gate_fused,
    )

    gate, out = _x(), _x()
    return sigmoid_gate_fused(gate, out), torch.sigmoid(_f(gate)) * _f(out)


def gated_projection_bwd_gate_flat_triton():
    """_sigmul_bwd, via _SigmoidGate.backward.

    ``da`` is random rather than the driver's ``.sum()`` (all-ones): a uniform upstream grad
    cannot distinguish ``da*o*s*(1-s)`` from an expression that drops the ``da`` factor.
    """
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import (
        sigmoid_gate_fused,
    )

    gate, out = _x().requires_grad_(), _x().requires_grad_()
    da = _x()
    sigmoid_gate_fused(gate, out).backward(da)
    s = torch.sigmoid(_f(gate))
    return {"dg": (gate.grad, _f(da) * _f(out) * s * (1.0 - s)),
            "do": (out.grad, _f(da) * s)}

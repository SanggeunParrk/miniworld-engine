"""Accuracy checkers for the ``tm2`` family.

trimul_inproj, tm1, tm2 and gated_projection were one module (``checks_trimul.py``). The two
layout rules these follow -- return a dict with one entry per block, and check the kernel rather
than the cuBLAS GEMMs the launcher wraps it in -- plus the lazy-import rule, are written out in
``checks/trimul_inproj.py``. The shapes come from ``drivers/trimul_inproj.py`` and the shared
helpers from ``checks/__init__.py``.
"""
from __future__ import annotations

import torch

from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels._tiles import tile_grid
from miniworld_engine.kernels.checks import _exact_fp32_matmul, _f
from miniworld_engine.kernels.drivers.trimul_inproj import D, L, M, _rows, _w, _x

# ── tm2 ──────────────────────────────────────────────────────────────────────────────────────

def trimul_outproj_gemm_gate_triton():
    """fused_sigmoid_gate2_fwd_kernel: out = sigmoid(x_gate @ W_gate) * (x_out @ W_out).

    Two independent GEMMs sharing one K loop and one output tile. ``TritonTM2Function.forward``
    is this kernel alone, so ``triton_tm2`` is its launch site.

    The two operands are DIFFERENT tensors (``x`` gates, ``y`` is projected), so the driver's two
    separate ``_rows()`` draws are what makes a crossed ``x_gate``/``x_out`` or ``W_gate``/``W_out``
    pointer visible at all -- with one shared input the swap is invisible. As in tm1, ``N`` is both
    the contraction and the output extent, so a ragged D makes both tails partial at once.

    Reference: ``tm2.reference.tm2_pytorch``, in fp32.
    """
    from miniworld_engine.kernels.tm2.reference import tm2_pytorch
    from miniworld_engine.kernels.tm2.triton.main import triton_tm2

    _exact_fp32_matmul()
    # ``_x()`` for both, for the same reason as tm1 above: TritonTM2Function keys on
    # length_of(x.shape) before its rearrange. They are still two INDEPENDENT draws, so a
    # crossed x_gate/x_out or W_gate/W_out pointer stays visible.
    x, y = _x(), _x()
    Wg, Wo = _w(), _w()
    out = triton_tm2(x, y, Wg, Wo)
    return out, tm2_pytorch(_f(x), _f(y), _f(Wg), _f(Wo))


def trimul_outproj_bwd_gate_recompute_triton():
    """fused_sigmoid_gate2_bwd_kernel: recomputes A = x@Wg and B = y@Wo, emits dA/dB.

    ``dA = dB * B * (1-g)`` with ``dB = dy * g`` is again the textbook ``dy * B * g * (1-g)``;
    the reference uses the textbook form. Launched at TritonTM2Function.backward's launch site,
    for the same reason as the tm1 backward above.
    """
    from miniworld_engine.kernels.tm2.triton.main import (
        fused_sigmoid_gate2_bwd_kernel,
    )

    x, y = _rows(), _rows()
    Wg, Wo = _w(), _w()
    grad_out = _rows()
    dA, dB = torch.empty_like(x), torch.empty_like(x)
    grid = lambda meta: tile_grid(M, D, meta["BLOCK_M1"], meta["BLOCK_N"])
    fused_sigmoid_gate2_bwd_kernel[grid](x, y, Wg, Wo, grad_out, dA, dB, M, D,
                                        shape_key=token_key(L, N=D))
    g = torch.sigmoid(_f(x) @ _f(Wg))
    B = _f(y) @ _f(Wo)
    dy = _f(grad_out)
    return {"dA": (dA, dy * B * g * (1.0 - g)),
            "dB": (dB, dy * g)}

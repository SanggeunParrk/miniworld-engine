"""Accuracy checkers for the ``tm1`` family.

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
from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.trimul_inproj import D, L, M, _bdll, _rows, _w, _x

# ── tm1 ──────────────────────────────────────────────────────────────────────────────────────

def trimul_gemm_gate_triton():
    """fused_sigmoid_gate_fwd_kernel: (sigmoid(x@WLg)*(x@WL), sigmoid(x@WRg)*(x@WR)).

    ``TritonTM1Function.forward`` is this kernel and nothing else -- no cuBLAS around it -- so the
    public ``triton_tm1`` is the kernel's launch site and the comparison is of the kernel alone.

    ``N`` is the contraction extent AND the output width here (square weights), so at a ragged D
    the ``for k_start in range(0, N, BLOCK_K)`` loop and the ``offs_n`` output tile go partial at
    once: a K column past ``N`` reaching either accumulator, or an N column past ``N`` reaching
    the store, both move the answer away from a reference that contracts exactly D columns.

    Reference: ``tm1.reference.tm1_pytorch``, the module's own definition, in fp32.
    """
    from miniworld_engine.kernels.tm1.reference import tm1_pytorch
    from miniworld_engine.kernels.tm1.triton.main import triton_tm1

    _exact_fp32_matmul()
    # ``_x()``, not ``_rows()``: TritonTM1Function reads token_key(length_of(x.shape)) BEFORE its
    # own rearrange to (M, d), and ``length_of`` refuses an already-flattened (M, d) -- shape[-2]
    # there is M, not L. It reshapes left/right back to x's shape, and the reference contracts
    # over the last axis either way, so the comparison is the same numbers at (1, L, L, D).
    x = _x()
    WL, WLg, WR, WRg = _w(), _w(), _w(), _w()
    left, right = triton_tm1(x, WL, WLg, WR, WRg)
    ref_l, ref_r = tm1_pytorch(_f(x), _f(WL), _f(WLg), _f(WR), _f(WRg))
    return {"left": (left, ref_l), "right": (right, ref_r)}


def trimul_bwd_gate_recompute_triton():
    """fused_sigmoid_gate_bwd_kernel: recomputes the four in-projection GEMMs, emits 4 grads.

    ``dLA = dLB * LB * (1 - Lg)`` with ``dLB = dleft * Lg`` is the same expression as the textbook
    ``dleft * LB * s * (1-s)`` -- substituting dLB gives it term for term -- so the reference is
    written in the textbook form and the identity is what gets tested.

    Launched at TritonTM1Function.backward's launch site. dLA/dLB/dRA/dRB are the kernel's own
    outputs; the autograd path reduces them through six cuBLAS GEMMs into (dx, 4x dW), which would
    both blur which block is wrong and add matmul error to the comparison.
    """
    from miniworld_engine.kernels.tm1.triton.main import (
        fused_sigmoid_gate_bwd_kernel,
    )

    x = _rows()
    WL, WLg, WR, WRg = _w(), _w(), _w(), _w()
    dleft, dright = _rows(), _rows()
    dLA, dLB, dRA, dRB = (torch.empty_like(x) for _ in range(4))
    grid = lambda meta: tile_grid(M, D, meta["BLOCK_M1"], meta["BLOCK_N"])
    fused_sigmoid_gate_bwd_kernel[grid](x, WLg, WL, WRg, WR, dleft, dright,
                                        dLA, dLB, dRA, dRB, M, D, shape_key=token_key(L, N=D))
    xf = _f(x)
    LB, RB = xf @ _f(WL), xf @ _f(WR)
    Lg, Rg = torch.sigmoid(xf @ _f(WLg)), torch.sigmoid(xf @ _f(WRg))
    dl, dr = _f(dleft), _f(dright)
    return {"dLA": (dLA, dl * LB * Lg * (1.0 - Lg)),
            "dLB": (dLB, dl * Lg),
            "dRA": (dRA, dr * RB * Rg * (1.0 - Rg)),
            "dRB": (dRB, dr * Rg)}


def gated_projection_gate_inplace_flat_triton():
    """tm1/cute/launch.py _gate_mul_kernel: IN-PLACE proj *= sigmoid(gate) over a flat buffer.

    proj is both operand and destination, so the reference is taken from a copy made before the
    launch. (The kernel's @autotune carries ``restore_value=['proj_ptr']``, so the tuning sweep
    itself does not compound the multiply -- exactly one application survives.)
    """
    from miniworld_engine.kernels.tm1.cute.launch import _fused_gate_mul

    proj, gate = _bdll().contiguous(), _bdll().contiguous()
    ref = torch.sigmoid(_f(gate)) * _f(proj)
    _fused_gate_mul(proj, gate)
    return proj, ref


def gated_projection_gate_packed_flat_triton():
    """tm1/cute/launch.py _glu_wide_kernel: the two operands are HALVES OF ONE (1, 2D, L, L).

    Flattened, ``wide`` is the gate channels [0:D] then the proj channels [D:2D], each L*L long;
    the kernel pairs flat element e with e + D*L*L. Slicing the channel axis reproduces exactly
    that pairing on the reference side.
    """
    from miniworld_engine.kernels.tm1.cute.launch import _glu_wide

    wide = _bdll(2 * D)
    out = torch.empty(1, D, L, L, device=dev(), dtype=BF16)
    _glu_wide(out, wide, D, L)
    return out, torch.sigmoid(_f(wide[:, :D])) * _f(wide[:, D:])

"""Numerical checks for the ``swa_dit`` family, against ``reference.swa_dit_block_reference`` in fp32 on the same values.

The fused block is checked whole -- each kernel feeds the next, so a stage's error shows at the block's output or in the
gradients it produces -- at a checker-sized shape the dense fp32 reference can afford: A = 8 augments x B = 2 batch
elements (the hoisted modulation is indexed per batch element), S = 333 atoms (partial tiles on every axis), ragged
seqused. A forward kernel is scored on the output; each backward kernel on the gradients it produces (dmod is the
gradient of the hoisted modulation, which every backward kernel accumulates a slice of).

The block dispatches on the activation dtype, so a row is checked at the precision it declares and refuses the other
(``MINIWORLD_DRIVER_DTYPE``). The bf16 Triton rows pin the hand-CUDA stages off; each ``*_sm90_cuda`` row turns on its
own stage alone, so its error is not shared with another CUDA stage, and raises (arch-gated) off sm_90. The fp32 rows
are Triton only. The reference runs with TF32 off: the fp32 kernels' TF32 projections and bf16 attention operands are
part of the error their band prices.
"""
from __future__ import annotations

from miniworld_engine.kernels.checks import _f, _fixed, _grads, _no_tf32
from miniworld_engine.kernels.drivers.swa_dit import (
    _TRITON,
    _inputs,
    _pinned,
    _require,
    _require_cuda,
)

_NAMES = ("dq", "dmod", "dw_qkv", "dw_gate", "dw_out", "dw_up", "dw_down")
_SHAPE = {"a": 8, "b": 2, "s": 333, "seed": 1}
_CUDA_ONLY = {"engine_backend": "auto", "swa_dit_qkvg_fwd_cuda": False, "swa_dit_ffn_fwd_cuda": False,
              "swa_dit_ffn_bwd_cuda": False}


def _forward_pair(precision, **pins):
    import torch

    from miniworld_engine.kernels.swa_dit.interface import swa_dit_block
    from miniworld_engine.kernels.swa_dit.reference import swa_dit_block_reference

    _require(precision)
    _fixed()
    q, mod, cos, sin, seqused, *weights, b, hw = _inputs(**_SHAPE)
    with _no_tf32():
        expected = swa_dit_block_reference(_f(q), _f(mod), cos, sin, seqused, *map(_f, weights), b, hw)
    with _pinned(**pins), torch.no_grad():
        actual = swa_dit_block(q, mod, cos, sin, seqused, *weights, b, hw)
    return {"out": (actual, expected)}


def _backward_pairs(names, precision, **pins):
    from miniworld_engine.kernels.swa_dit.interface import swa_dit_block
    from miniworld_engine.kernels.swa_dit.reference import swa_dit_block_reference

    _require(precision)
    _fixed()
    q, mod, cos, sin, seqused, *weights, b, hw = _inputs(**_SHAPE)
    with _no_tf32(), _pinned(**pins):
        got = _grads(lambda q_, m_, *w: swa_dit_block(q_, m_, cos, sin, seqused, *w, b, hw), [q, mod, *weights],
                     lambda q_, m_, *w: swa_dit_block_reference(q_, m_, cos, sin, seqused, *w, b, hw), _NAMES)
    return {n: got[n] for n in names}


def swa_dit_inproj_fwd_triton():
    return _forward_pair("bf16", **_TRITON)


def swa_dit_softmax_fwd_triton():
    return _forward_pair("bf16", **_TRITON)


def swa_dit_output_swiglu_fwd_triton():
    return _forward_pair("bf16", **_TRITON)


def swa_dit_swiglu_bwd_triton():
    return _backward_pairs(("dmod", "dw_up", "dw_down"), "bf16", **_TRITON, swa_dit_ffn_dw="mat")


def swa_dit_swiglu_dw_triton():
    return _backward_pairs(("dw_up", "dw_down"), "bf16", **_TRITON, swa_dit_ffn_dw="fused")


def swa_dit_output_bwd_triton():
    return _backward_pairs(("dmod", "dw_out"), "bf16", **_TRITON)


def swa_dit_softmax_bwd_dq_triton():
    return _backward_pairs(("dq", "dw_qkv"), "bf16", **_TRITON)


def swa_dit_softmax_bwd_dkdv_triton():
    return _backward_pairs(("dq", "dw_qkv"), "bf16", **_TRITON)


def swa_dit_inproj_bwd_triton():
    return _backward_pairs(("dq", "dmod", "dw_qkv", "dw_gate"), "bf16", **_TRITON)


def swa_dit_inproj_fwd_sm90_cuda():
    _require_cuda("qkvg")
    return _forward_pair("bf16", **{**_CUDA_ONLY, "swa_dit_qkvg_fwd_cuda": True})


def swa_dit_output_swiglu_fwd_sm90_cuda():
    _require_cuda("fwd")
    return _forward_pair("bf16", **{**_CUDA_ONLY, "swa_dit_ffn_fwd_cuda": True})


def swa_dit_swiglu_bwd_sm90_cuda():
    _require_cuda("bwd")
    return _backward_pairs(("dq", "dmod", "dw_up", "dw_down"), "bf16",
                           **{**_CUDA_ONLY, "swa_dit_ffn_bwd_cuda": True, "swa_dit_ffn_dw": "mat"})


def swa_dit_inproj_fwd_fp32_triton():
    return _forward_pair("fp32")


def swa_dit_output_swiglu_fwd_fp32_triton():
    return _forward_pair("fp32")


def swa_dit_swiglu_bwd_fp32_triton():
    return _backward_pairs(("dmod", "dw_up", "dw_down"), "fp32")


def swa_dit_output_bwd_fp32_triton():
    return _backward_pairs(("dmod", "dw_out"), "fp32")


def swa_dit_inproj_bwd_fp32_triton():
    return _backward_pairs(("dq", "dmod", "dw_qkv", "dw_gate"), "fp32")

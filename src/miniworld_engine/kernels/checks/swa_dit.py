"""Numerical checks for the ``swa_dit`` family, against ``reference.swa_dit_block_reference`` in fp32 on the same values.

The fused block is checked whole -- each kernel feeds the next, so a stage's error shows at the block's output or in the
gradients it produces -- at a checker-sized shape the dense fp32 reference can afford: A = 8 augments x B = 2 batch
elements (the hoisted modulation is indexed per batch element), S = 333 atoms (partial tiles on every axis), ragged
seqused. A forward kernel is scored on the output; each backward kernel on the gradients it produces (dmod is the
gradient of the hoisted modulation, which every backward kernel accumulates a slice of).

The block dispatches on the activation dtype, so a row is checked at the precision it declares and refuses the other
(``MINIWORLD_DRIVER_DTYPE``). The Triton rows (bf16 and fp32) pin ``force_portable_kernels``; each bf16
``*_sm90_cuda`` row runs the block with every hand-CUDA stage it serves (there is no per-stage switch) and raises
(arch-gated) off sm_90; the ``*_fp32_sm90_cuda`` rows run the fp32 CUDA extension (all stages at once -- it is one
forward and one backward launch sequence) under TF32-allowing matmul precision, which is the only precision it serves.
The reference runs with TF32 off: the fp32 kernels' TF32 projections and bf16 attention operands are part of the error
their band prices.
"""
from __future__ import annotations

from miniworld_engine.kernels.checks import _f, _fixed, _grads, _no_tf32
from miniworld_engine.kernels.drivers.swa_dit import (
    _CUDA_FP32,
    _TRITON,
    _TRITON_FP32,
    _inputs,
    _pinned,
    _require,
    _require_cuda,
    tf32_allowed,
)

_NAMES = ("dq", "dmod", "dw_qkv", "dw_gate", "dw_out", "dw_up", "dw_down")
_SHAPE = {"a": 8, "b": 2, "s": 333, "seed": 1}
_CUDA = {"force_portable_kernels": False}


def _forward_pair(precision, **pins):
    import torch

    from miniworld_engine.kernels.swa_dit.interface import swa_dit_block
    from miniworld_engine.kernels.swa_dit.reference import swa_dit_block_reference

    _require(precision)
    _fixed()
    q, mod, cos, sin, seqused, *weights, b, hw = _inputs(**_SHAPE)
    with _no_tf32():
        expected = swa_dit_block_reference(_f(q), _f(mod), cos, sin, seqused, *map(_f, weights), B=b, half_window=hw)
    with _pinned(**pins), torch.no_grad():
        actual = swa_dit_block(q, mod, cos, sin, seqused, *weights, B=b, half_window=hw)
    return {"out": (actual, expected)}


def _backward_pairs(names, precision, **pins):
    from miniworld_engine.kernels.swa_dit.interface import swa_dit_block
    from miniworld_engine.kernels.swa_dit.reference import swa_dit_block_reference

    _require(precision)
    _fixed()
    q, mod, cos, sin, seqused, *weights, b, hw = _inputs(**_SHAPE)
    with _no_tf32(), _pinned(**pins):
        got = _grads(lambda q_, m_, *w: swa_dit_block(q_, m_, cos, sin, seqused, *w, B=b, half_window=hw), [q, mod, *weights],
                     lambda q_, m_, *w: swa_dit_block_reference(q_, m_, cos, sin, seqused, *w, B=b, half_window=hw), _NAMES)
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
    return _forward_pair("bf16", **{**_CUDA, "swa_dit_rows_impl": "wgmma"})


def swa_dit_output_swiglu_fwd_sm90_cuda():
    _require_cuda("fwd")
    return _forward_pair("bf16", **{**_CUDA, "swa_dit_rows_impl": "wgmma"})


def swa_dit_swiglu_bwd_sm90_cuda():
    _require_cuda("bwd")
    return _backward_pairs(("dq", "dmod", "dw_up", "dw_down"), "bf16",
                           **{**_CUDA, "swa_dit_ffn_dw": "mat", "swa_dit_rows_impl": "wgmma"})


def swa_dit_softmax_fwd_sm90_cuda():
    _require_cuda("attn")
    return _forward_pair("bf16", **_CUDA)


def swa_dit_softmax_bwd_sm90_cuda():
    """The attention backward's dq / dk / dv reach the reference through dq and dW_qkv (and dmod via the qkvg backward)."""
    _require_cuda("attn")
    return _backward_pairs(("dq", "dmod", "dw_qkv"), "bf16", **_CUDA)


def swa_dit_output_bwd_sm90_cuda():
    _require_cuda("slab")
    return _backward_pairs(("dq", "dmod", "dw_out", "dw_gate"), "bf16", **_CUDA)


def swa_dit_inproj_bwd_sm90_cuda():
    _require_cuda("slab")
    return _backward_pairs(("dq", "dmod", "dw_qkv", "dw_gate"), "bf16", **_CUDA)


def swa_dit_inproj_fwd_rows_sm90_cuda():
    _require_cuda("slab")
    return _forward_pair("bf16", **{**_CUDA, "swa_dit_rows_impl": "slab"})


def swa_dit_output_fwd_rows_sm90_cuda():
    _require_cuda("slab")
    return _forward_pair("bf16", **{**_CUDA, "swa_dit_rows_impl": "slab"})


def swa_dit_swiglu_fwd_rows_sm90_cuda():
    _require_cuda("slab")
    return _forward_pair("bf16", **{**_CUDA, "swa_dit_rows_impl": "slab"})


def swa_dit_swiglu_bwd_rows_sm90_cuda():
    _require_cuda("slab")
    return _backward_pairs(("dq", "dmod", "dw_up", "dw_down"), "bf16",
                           **{**_CUDA, "swa_dit_ffn_dw": "mat", "swa_dit_rows_impl": "slab"})


def swa_dit_inproj_fwd_fp32_triton():
    return _forward_pair("fp32", **_TRITON_FP32)


def swa_dit_output_swiglu_fwd_fp32_triton():
    return _forward_pair("fp32", **_TRITON_FP32)


def swa_dit_swiglu_bwd_fp32_triton():
    return _backward_pairs(("dmod", "dw_up", "dw_down"), "fp32", **_TRITON_FP32)


def swa_dit_output_bwd_fp32_triton():
    return _backward_pairs(("dmod", "dw_out"), "fp32", **_TRITON_FP32)


def swa_dit_inproj_bwd_fp32_triton():
    return _backward_pairs(("dq", "dmod", "dw_qkv", "dw_gate"), "fp32", **_TRITON_FP32)


# ---- the hoisted modulation and the weight-gradient GEMMs

def _modulation_pairs(names):
    import torch
    import torch.nn.functional as F

    from miniworld_engine.kernels.swa_dit.interface import swa_dit_hoist_modulation

    _require("bf16")
    _require_cuda("mod")
    _fixed()
    g = torch.Generator(device="cpu").manual_seed(3)
    b, s = 2, 333                                  # the checker's shape: a partial row tile
    c = torch.randn(b, s, 128, generator=g).to("cuda", torch.bfloat16).requires_grad_()
    w = (torch.randn(768, 128, generator=g) * 0.05).to("cuda", torch.bfloat16).requires_grad_()
    gm = torch.randn(b * s, 768, generator=g).to("cuda")
    with _pinned(force_portable_kernels=False):
        mod = swa_dit_hoist_modulation(c, w)
        mod.backward(gm)
    c64, w64 = c.detach().double().requires_grad_(), w.detach().double().requires_grad_()
    ref = F.silu(c64).reshape(-1, 128) @ w64.t()
    ref.backward(gm.double())
    return {"mod": (mod, ref.detach()), "dc": (c.grad, c64.grad), "dw": (w.grad, w64.grad)}


def swa_dit_gemm_fwd_sm90_cuda():
    return {"mod": _modulation_pairs(("mod",))["mod"]}


def swa_dit_gemm_bwd_sm90_cuda():
    pairs = _modulation_pairs(("dc", "dw"))
    return {"dc": pairs["dc"], "dw": pairs["dw"]}


def _wgrad_pairs():
    _require_cuda("wgrad")
    _require_cuda("slab")
    _require_cuda("attn")
    pins = {"force_portable_kernels": False, "swa_dit_ffn_dw": "mat", "swa_dit_rows_impl": "slab"}
    return _backward_pairs(("dw_qkv", "dw_gate", "dw_out", "dw_up", "dw_down"), "bf16", **pins)


def swa_dit_dw_sm90_cuda():
    _require_cuda("wgwg")
    return _wgrad_pairs()


# ---- fp32 CUDA: the kernel runs under TF32-allowing matmul precision (the only one it serves), forward and backward
# alike; the reference stays true fp32. Every stage is one extension, so each row is scored on what its stage produces.

def _cuda_fp32_forward_pair():
    import torch

    from miniworld_engine.kernels.swa_dit.interface import swa_dit_block
    from miniworld_engine.kernels.swa_dit.reference import swa_dit_block_reference

    _require("fp32")
    _require_cuda("fp32")
    _fixed()
    q, mod, cos, sin, seqused, *weights, b, hw = _inputs(**_SHAPE)
    with _no_tf32():
        expected = swa_dit_block_reference(_f(q), _f(mod), cos, sin, seqused, *map(_f, weights), B=b, half_window=hw)
    with tf32_allowed(), _pinned(**_CUDA_FP32), torch.no_grad():
        actual = swa_dit_block(q, mod, cos, sin, seqused, *weights, B=b, half_window=hw)
    return {"out": (actual, expected)}


def _cuda_fp32_backward_pairs(names):
    import torch

    from miniworld_engine.kernels.swa_dit.interface import swa_dit_block
    from miniworld_engine.kernels.swa_dit.reference import swa_dit_block_reference

    _require("fp32")
    _require_cuda("fp32")
    _fixed()
    q, mod, cos, sin, seqused, *weights, b, hw = _inputs(**_SHAPE)
    inputs = [q, mod, *weights]
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    with tf32_allowed(), _pinned(**_CUDA_FP32):     # forward AND backward: the backward's dispatch reads it too
        out = swa_dit_block(leaves[0], leaves[1], cos, sin, seqused, *leaves[2:], B=b, half_window=hw)
        dy = torch.randn_like(out)
        out.backward(dy)
    refs = [_f(t).requires_grad_(True) for t in inputs]
    with _no_tf32():
        swa_dit_block_reference(refs[0], refs[1], cos, sin, seqused, *refs[2:], B=b, half_window=hw).backward(dy)
    pairs = {n: (lf.grad, rf.grad) for n, lf, rf in zip(_NAMES, leaves, refs, strict=True)}
    return {n: pairs[n] for n in names}


def swa_dit_transpose_fp32_sm90_cuda():
    return _cuda_fp32_backward_pairs(("dq", "dmod", "dw_qkv", "dw_gate", "dw_out", "dw_up", "dw_down"))


def swa_dit_inproj_fwd_fp32_sm90_cuda():
    return _cuda_fp32_forward_pair()


def swa_dit_softmax_fwd_fp32_sm90_cuda():
    return _cuda_fp32_forward_pair()


def swa_dit_output_swiglu_fwd_fp32_sm90_cuda():
    return _cuda_fp32_forward_pair()


def swa_dit_swiglu_bwd_fp32_sm90_cuda():
    return _cuda_fp32_backward_pairs(("dmod", "dw_up", "dw_down"))


def swa_dit_output_bwd_fp32_sm90_cuda():
    return _cuda_fp32_backward_pairs(("dmod", "dw_out"))


def swa_dit_softmax_bwd_fp32_sm90_cuda():
    return _cuda_fp32_backward_pairs(("dq", "dw_qkv"))


def swa_dit_inproj_bwd_fp32_sm90_cuda():
    return _cuda_fp32_backward_pairs(("dq", "dmod", "dw_qkv", "dw_gate"))


def swa_dit_dw_fp32_sm90_cuda():
    _require_cuda("wgwg32")
    return _cuda_fp32_backward_pairs(("dw_qkv", "dw_gate", "dw_out", "dw_up", "dw_down"))


def swa_dit_transpose_sm90_cuda():
    """The packs against the torch layouts, bit for bit."""
    import torch

    from miniworld_engine.kernels.swa_dit import dispatch as D
    from miniworld_engine.kernels.swa_dit.cuda.loader import extension

    _require("bf16")
    _require_cuda("pack")
    g = torch.Generator(device="cpu").manual_seed(5)
    wu = torch.randn(512, 128, generator=g).to("cuda", torch.bfloat16)
    wd = torch.randn(128, 256, generator=g).to("cuda", torch.bfloat16)
    got = extension("pack").pack_ffn(wu, wd)
    ref = (D._pack_ffn64(wu), *D._pack_ffn(wu, wd))
    return {n: (a.float(), b.float()) for n, a, b in zip(("wab64", "wab32", "wdt", "wabt"), got, ref, strict=True)}

"""Training core of the bias-only attention in FEATURE-MAJOR layout: no permute copy, no concatenation, no transposed GEMM operand copy.

The core of ``BiasOnlyAttention`` (value / gate projection, ``softmax(bias) v``, sigmoid gate, output projection) is, in token-major
layout, a chain of cuBLAS GEMMs joined by layout changes: the value has to become head-major ``[H, L, A*Dh]`` for the batched GEMM
(a copy), the product comes back as ``[H, L, A*Dh]`` and the gate pass permutes it again, and the backward undoes all of it (a
permuted ``dout``, a ``cat`` of the value and gate gradients). Those copies were 330 us of a 3.3 ms bias-only DiT training step
(``M = 48 * 512`` rows, bf16), as much as the GEMMs they connect.

The same chain in FEATURE-major layout needs none of them. For ONE pair (``B = 1``; ``M = A * L`` token rows ``(a, l)``) keep every
``[tokens, features]`` tensor as its transpose ``t[(h, d), (a, l)]``, which is what a GEMM ``W @ x^T`` produces (cuBLAS reads ``x`` as
a transposed operand: a flag, not a copy):

    row ``(d, a)`` of head ``h`` of ``Vt`` (``[H, Dh * A, L]``) is the contiguous run ``value[a, :, h, d]`` over the keys ``j``, so

        out^T_h = Vt_h @ P_h^T         one plain batched GEMM over the heads, ``P_h^T`` a transposed VIEW of the weights
        dVt_h   = dOut^T_h @ P_h       likewise
        dP_h    = dOut^T_h^T @ Vt_h    contraction over the ``(d, a)`` rows

and the features-major outputs of two of them are the halves of ``dvg^T`` already in place, so ``dvg = cat(dvalue, dgate)`` is two
writes into one buffer. The sigmoid gate pass is the flat ``_sigmul_fwd`` / ``_sigmul_bwd`` of the gated_projection family on
contiguous operands (the gate half of ``vg^T`` is a contiguous slice of it).

Everything below is bf16 / fp32-TF32 GEMMs through cuBLAS plus two flat Triton passes, in ONE autograd Function per call; the
pair-bias softmax (the weights ``P``) stays outside, in autograd, and receives ``dP``.
"""

from __future__ import annotations

import torch
import triton

from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels._cast import cast_pack
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.gated_projection.triton.main import _sigmul_bwd, _sigmul_fwd


def _core_fwd_fake(vgt, p, shape_key=0):
    c = vgt.shape[0] // 2
    return vgt.new_empty((c, vgt.shape[1])), vgt.new_empty((c, vgt.shape[1]))


@opaque(fake=_core_fwd_fake, name="bias_only_attention_core_t_fwd")
def _core_fwd(vgt: torch.Tensor, p: torch.Tensor, shape_key: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``vgt`` [2C, M] (value rows then gate rows, feature-major), ``p`` [H, L, L] (softmax weights) -> (out^T, gsig^T), both [C, M]:
    ``out^T = Vt @ P^T`` per head and ``gsig^T = sigmoid(gate^T) * out^T``."""
    c, m = vgt.shape[0] // 2, vgt.shape[1]
    h, length = p.shape[0], p.shape[1]
    outt = torch.bmm(vgt[:c].view(h, c // h * (m // length), length), p.transpose(1, 2)).view(c, m)
    gsigt = torch.empty_like(outt)
    n = outt.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_E"]),)  # noqa: E731
    _sigmul_fwd[grid](vgt[c:], outt, gsigt, n, shape_key=shape_key)
    return outt, gsigt


def _core_bwd_fake(dgsigt, vgt, outt, p, shape_key=0):
    return vgt.new_empty(vgt.shape), p.new_empty(p.shape)


@opaque(fake=_core_bwd_fake, name="bias_only_attention_core_t_bwd")
def _core_bwd(dgsigt: torch.Tensor, vgt: torch.Tensor, outt: torch.Tensor, p: torch.Tensor,
              shape_key: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (``dvg^T`` [2C, M]: value gradient rows then gate gradient rows, ``dP`` [H, L, L])."""
    c, m = outt.shape
    h, length = p.shape[0], p.shape[1]
    rows = c // h * (m // length)
    dvgt = torch.empty_like(vgt)
    doutt = torch.empty_like(outt)
    n = outt.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_E"]),)  # noqa: E731
    _sigmul_bwd[grid](dgsigt, vgt[c:], outt, dvgt[c:], doutt, n, shape_key=shape_key)
    d3 = doutt.view(h, rows, length)
    torch.bmm(d3, p, out=dvgt[:c].view(h, rows, length))                  # dVt_h = dOut^T_h @ P_h
    dp = torch.bmm(d3.transpose(1, 2), vgt[:c].view(h, rows, length))     # dP_h = dOut_h @ V_h^T, contraction over (d, a)
    return dvgt, dp


class BiasOnlyCoreT(torch.autograd.Function):
    """``o = to_out(sigmoid(g) * (softmax(bias) v))`` with ``[v; g] = x @ W_vg^T``, for x [M, d] with M = A * L (one pair), ``p`` [H, L, L].

    Weights fp32 master on an H100 under bf16 activations: cast inside (one multi-tensor copy) and the weight gradients are fp32 GEMM
    outputs, never rounded to bf16 (the ``integrations.h100_master`` contract). Otherwise the weights are used as given.
    """

    @staticmethod
    def forward(ctx, x, wv, wg, wo, p):
        ctx.master = x.dtype == torch.bfloat16 and wv.dtype == torch.float32
        if ctx.master:
            if wv.is_contiguous() and wg.is_contiguous() and wo.is_contiguous():
                w_cat, w_out = cast_pack([wv, wg, wo], 2)   # one launch, the halves of the concatenation written in place
            else:
                w_cat, w_out = torch.cat([wv, wg], 0).to(torch.bfloat16), wo.to(torch.bfloat16)
        else:
            w_cat, w_out = torch.cat([wv, wg], 0), wo
        key = both_key(x.shape[0])
        ctx.key = key
        vgt = torch.mm(w_cat, x.t())                 # [2C, M]
        outt, gsigt = _core_fwd(vgt, p, shape_key=key)
        o = torch.mm(gsigt.t(), w_out.t())           # [M, d]
        ctx.save_for_backward(x, w_cat, w_out, p, vgt, outt, gsigt)
        ctx.c = wv.shape[0]
        return o

    @staticmethod
    def backward(ctx, do):
        x, w_cat, w_out, p, vgt, outt, gsigt = ctx.saved_tensors
        do = do.contiguous()
        f32 = torch.float32 if ctx.master else None
        dgsigt = torch.mm(w_out.t(), do.t())         # [C, M]
        dwo = torch.mm(do.t(), gsigt.t(), out_dtype=f32) if f32 else torch.mm(do.t(), gsigt.t())
        dvgt, dp = _core_bwd(dgsigt, vgt, outt, p, shape_key=ctx.key)
        dx = torch.mm(dvgt.t(), w_cat)               # [M, d]
        dw = torch.mm(dvgt, x, out_dtype=f32) if f32 else torch.mm(dvgt, x)   # [2C, d]
        return dx, dw[:ctx.c], dw[ctx.c:], dwo, dp


def bias_only_core_t(x, wv, wg, wo, p):
    return BiasOnlyCoreT.apply(x, wv, wg, wo, p)

"""``tdt_attention``: the training attention core as one autograd op (bf16 operands inside an fp32 model).

Takes the engine's tensors -- q, k, v [A, 1, L, H, 48] fp32 (after qk_norm), bias [1, L, L, H] fp32 or head-major
[H, 1, L, L] (``pair_bias_all(..., head_major=True)``, no permute either way), key mask [A, 1, L] bool or None -- and
returns O [A, 1, L, H, 48] fp32, the engine's ``_kernel_attention_pair_bias`` contract. The casts are fused Triton passes
(prep.py).
"""
import math
import os
import torch
from .ref import LOG2E, D
from .fwd import attn_fwd
from .bwd import attn_dq, attn_dkv, attn_dkv2, attn_dqb, attn_dkv_nobias
from .prep import prep_qkv, prep_do, bias_prep


class _TdtAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, mask):  # noqa: D102
        A, B, L, H, Dh = q.shape
        assert B == 1 and Dh == D
        ctx.head_major = bias.shape == (H, 1, L, L)
        assert ctx.head_major or bias.shape == (1, L, L, H)
        qs, kb, vb = prep_qkv(q, k, v, LOG2E / math.sqrt(D))
        bb = bias_prep(bias.reshape(H, L, L) if ctx.head_major else bias[0].permute(2, 0, 1), LOG2E)
        km = None
        if mask is not None:
            km = torch.where(mask.reshape(A, L), 0.0, -1e30).float().contiguous()
        o, lse = attn_fwd(qs, kb, vb, bb, km, A, L)
        ctx.save_for_backward(qs, kb, vb, bb, km if km is not None else torch.empty(0, device=q.device), o, lse)
        ctx.has_mask = km is not None
        ctx.shape = (A, L, H, Dh)
        return o.view(A, 1, L, H, Dh)

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        qs, kb, vb, bb, km, o, lse = ctx.saved_tensors
        km = km if ctx.has_mask else None
        A, L, H, Dh = ctx.shape
        dob, dd = prep_do(do, o, A, L, H)
        mode = os.environ.get("TDT_BWD", "dqb" if A % 3 == 0 else "atomic")
        if mode == "dqb":                   # dbias in the dQ pass, 3 samples summed on chip before the L2 reds
            dq, db = attn_dqb(qs, kb, vb, dob, bb, km, lse, dd, A, L)
            dk, dv = attn_dkv_nobias(qs, kb, vb, dob, bb, km, lse, dd, A, L)
        else:                               # dbias by L2 atomics per sample in the dK / dV pass
            dq = attn_dq(qs, kb, vb, dob, bb, km, lse, dd, A, L)
            dk, dv, db = attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, A, L)
        shp = (A, 1, L, H, Dh)
        db = db.unsqueeze(1) if ctx.head_major else db.permute(1, 2, 0).unsqueeze(0)   # views: the hoist's cat copies
        return dq.view(shp), dk.view(shp), dv.view(shp), db, None


def tdt_attention(q, k, v, bias, mask=None):
    """softmax(q k^T / sqrt 48 + bias) v with the sm_90 training kernels; the engine's tensor contract."""
    if mask is not None and mask.ndim == 2:
        mask = mask[None].expand(q.shape[0], *mask.shape)
    return _TdtAttention.apply(q, k, v, bias, mask)

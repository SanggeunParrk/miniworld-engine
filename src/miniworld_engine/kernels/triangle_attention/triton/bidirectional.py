"""Fused bidirectional triangle self-attention (starting + ending) in ONE autograd Function.

Reuses the v7 single-direction Triton kernels (``_attn_fwd`` / ``_attn_bwd_*``) unchanged — they
are fully stride-parametrized (read strides for q/k/v, write strides for out and dq/dk/dv, bias
strides), so BOTH directions run through the same kernels:

    starting  = the tensors as-is (row = i, contract over the key axis).
    ending    = the (i<->j)-transposed VIEWS (row = j); because the OUTPUT is also a transposed
                view into the concat buffer, the kernel writes the ending result straight into
                the concat's ``[i,j]`` ending slice — no explicit ``.transpose()``/``cat``.

The Function takes the DOUBLED projections ``[B,L,L,2*d_hidden]`` and returns the concat output
and doubled grads directly, so there is no ``split``/``cat``/``transpose`` in the autograd graph
(those materializations were ~21% of the naive bidir self-attention step)."""
from __future__ import annotations

from miniworld_engine.kernels._compile import opaque

import torch
import triton
from einops import rearrange, reduce

from miniworld_engine.autotune.shape_key import token_key

from .main import (
    _attn_bwd_dkdv,
    _attn_bwd_dq,
    _attn_bwd_preprocess,
    _attn_fwd,
)


def _fwd_dir(q, k, v, bias, out, m, sm_scale):
    """One direction forward. q,k,v [B,H,L,L,D] (any-stride views), bias [B,H,L,L], out
    [B,H,L,L,D] view into the concat buffer, m [B,H,L,L] fresh. Returns the contiguous bias
    (saved for backward)."""
    B, H, L, _, D = q.shape
    bias = bias.contiguous()
    grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * H, L]
    _attn_fwd[grid](
        q, k, v, bias, sm_scale, m, out,
        *q.stride(), *out.stride(), *bias.stride(), *m.stride(),
        B, H, L, D, HEAD_DIM_PAD=triton.next_power_of_2(D), shape_key=token_key(L),
    )
    return bias


def _bwd_dir(q, k, v, bias, m, out, dout, dq, dk, dv, sm_scale):
    # NOTE the three launches below pass exactly the positional tails main.py passes to the same
    # three kernels. They used to carry ONE EXTRA leading `HL` each (`HL, B, HL, L, D` where main
    # passes `HL, B, L, D`, and likewise for dkdv/dq), so every argument after it was shifted and
    # the kernels read the wrong scalars. Nothing caught it: this file's kernels have no row in
    # registry.csv, so no driver and no checker ever launches this path, even though
    # modules/triangle_attention/bidirectional.py:175 imports it.
    """One direction backward. dq/dk/dv [B,H,L,L,D] views into the concat-grad buffer. Returns
    dbias [B,H,L,L] (in this direction's frame)."""
    B, H, L, _, D = q.shape
    HL = H * L
    m_m = rearrange(m, "B H L L2 -> B (H L) L2")
    delta = torch.empty_like(m_m)
    grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * HL, 1]
    _attn_bwd_preprocess[grid](
        out, dout, delta, *out.stride(), *dout.stride(), HL, B, L, D,
        HEAD_DIM_PAD=triton.next_power_of_2(D), shape_key=token_key(L),
    )
    dbias = torch.empty(B, HL, L, L, device=q.device, dtype=bias.dtype)
    grid_kv = lambda META: [triton.cdiv(L, META["BLOCK_M2"]), 1, B * HL]
    _attn_bwd_dkdv[grid_kv](
        q, k, v, bias, sm_scale, dout, dk, dv, dbias, m_m, delta,
        *q.stride(), *dk.stride(), *dout.stride(), *bias.stride(), L * L, L, HL, D,
        HEAD_DIM_PAD=triton.next_power_of_2(D), shape_key=token_key(L),
    )
    grid_q = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), 1, B * HL]
    _attn_bwd_dq[grid_q](
        q, k, v, bias, sm_scale, dout, dq, m_m, delta,
        *q.stride(), *dq.stride(), *dout.stride(), *bias.stride(), L, HL, D,
        HEAD_DIM_PAD=triton.next_power_of_2(D), shape_key=token_key(L),
    )
    return reduce(dbias, "B (H L3) L L2 -> B H L L2", "sum", L3=L)


def _split_dirs(t, H):
    """[B,L,L,2*d_hidden] -> ([B,H,L,L,D] starting native, [B,H,L,L,D] ending transposed view)."""
    g = rearrange(t, "B L L2 (G D) -> B G L L2 D", G=2 * H)
    return g[:, :H], g[:, H:].transpose(2, 3)


def _bidir_fwd_fake(q, k, v, bs, be, n_head):
    """``(out, m_s, m_e)``: ``out`` is the packed ``(B, L, L, 2*d_hidden)`` concat buffer both
    directions write into; ``m_s``/``m_e`` are the ``(B, n_head, L, L)`` fp32 per-row logsumexps,
    ``m_e`` in the ending (row=j) frame its direction ran in.
    """
    B, L, _, D2 = q.shape
    f32 = torch.float32
    return (
        q.new_empty((B, L, L, D2)),
        q.new_empty((B, n_head, L, L), dtype=f32),
        q.new_empty((B, n_head, L, L), dtype=f32),
    )


@opaque(fake=_bidir_fwd_fake, name="triangle_attention_bidir_fwd")
def _bidir_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bs: torch.Tensor,
               be: torch.Tensor, n_head: int,
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both directions' forward launches -> ``(out, m_s, m_e)``.

    ``bs``/``be`` arrive already sliced out of the packed bias AND contiguous: they are saved for
    the backward, and an op may not return a view of its own input, so the caller has to be the
    one that makes them. See ``kernels._compile``.
    """
    B, L, _, D2 = q.shape
    H = n_head
    dh = D2 // 2
    D = dh // H
    sm_scale = D**-0.5

    qs, qe = _split_dirs(q, H)
    ks, ke = _split_dirs(k, H)
    vs, ve = _split_dirs(v, H)

    out = torch.empty(B, L, L, D2, device=q.device, dtype=q.dtype)
    out_s = rearrange(out[..., :dh], "B L L2 (H D) -> B H L L2 D", H=H)
    out_e = rearrange(out[..., dh:], "B L L2 (H D) -> B H L L2 D", H=H).transpose(2, 3)
    m_s = torch.empty(B, H, L, L, device=q.device, dtype=torch.float32)
    m_e = torch.empty(B, H, L, L, device=q.device, dtype=torch.float32)

    _fwd_dir(qs, ks, vs, bs, out_s, m_s, sm_scale)
    _fwd_dir(qe, ke, ve, be, out_e, m_e, sm_scale)
    return out, m_s, m_e


def _bidir_bwd_fake(q, k, v, bs, be, m_s, m_e, out, dout, n_head):
    """``(dq, dk, dv, dbias_s, dbias_e)``: the three input grads are packed
    ``(B, L, L, 2*d_hidden)`` like the doubled projections they mirror; the bias grads are
    per-direction ``(B, n_head, L, L)``, already reduced over the row axis, with ``dbias_e``
    still in the ending (row=j) frame -- the transpose back, the concat and the rearrange to the
    packed bias layout are left to the caller.
    """
    B, L, _, D2 = q.shape
    return (
        v.new_empty((B, L, L, D2)),
        v.new_empty((B, L, L, D2)),
        v.new_empty((B, L, L, D2)),
        bs.new_empty((B, n_head, L, L)),
        bs.new_empty((B, n_head, L, L)),
    )


@opaque(fake=_bidir_bwd_fake, name="triangle_attention_bidir_bwd")
def _bidir_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bs: torch.Tensor,
               be: torch.Tensor, m_s: torch.Tensor, m_e: torch.Tensor, out: torch.Tensor,
               dout: torch.Tensor, n_head: int) -> tuple[
                   torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both directions' backward launches -> ``(dq, dk, dv, dbias_s, dbias_e)``.

    ``dbias_e`` is still in the ending (row=j) frame; the transpose back, the concat and the
    rearrange to the packed layout are plain torch and stay in the caller.
    """
    B, L, _, D2 = q.shape
    H = n_head
    dh = D2 // 2
    D = dh // H
    sm_scale = D**-0.5

    qs, qe = _split_dirs(q, H)
    ks, ke = _split_dirs(k, H)
    vs, ve = _split_dirs(v, H)
    dq = torch.empty(B, L, L, D2, device=q.device, dtype=v.dtype)
    dk = torch.empty(B, L, L, D2, device=q.device, dtype=v.dtype)
    dv = torch.empty(B, L, L, D2, device=q.device, dtype=v.dtype)
    dqs, dqe = _split_dirs(dq, H)
    dks, dke = _split_dirs(dk, H)
    dvs, dve = _split_dirs(dv, H)
    out_s = rearrange(out[..., :dh], "B L L2 (H D) -> B H L L2 D", H=H)
    out_e = rearrange(out[..., dh:], "B L L2 (H D) -> B H L L2 D", H=H).transpose(2, 3)
    dout_s = rearrange(dout[..., :dh], "B L L2 (H D) -> B H L L2 D", H=H)
    dout_e = rearrange(dout[..., dh:], "B L L2 (H D) -> B H L L2 D", H=H).transpose(2, 3)

    dbias_s = _bwd_dir(qs, ks, vs, bs, m_s, out_s, dout_s, dqs, dks, dvs, sm_scale)
    dbias_e = _bwd_dir(qe, ke, ve, be, m_e, out_e, dout_e, dqe, dke, dve, sm_scale)
    return dq, dk, dv, dbias_s, dbias_e


class BidirTriangleAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, n_head):
        H = n_head
        bv = rearrange(bias, "B L L2 G -> B G L L2", G=2 * H)
        # Sliced and made contiguous HERE, not inside the op: these are what the backward saves,
        # and an op cannot return a view of its own input.
        bs = bv[:, :H].contiguous()
        be = bv[:, H:].transpose(2, 3).contiguous()

        out, m_s, m_e = _bidir_fwd(q, k, v, bs, be, H)

        ctx.save_for_backward(q, k, v, bs, be, m_s, m_e, out)
        ctx.H = H
        ctx.dh = q.shape[-1] // 2
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, bs, be, m_s, m_e, out = ctx.saved_tensors
        if dout.dtype != q.dtype:
            dout = dout.to(q.dtype)

        dq, dk, dv, dbias_s, dbias_e = _bidir_bwd(
            q, k, v, bs, be, m_s, m_e, out, dout, ctx.H,
        )
        # dbias_e is in the ending (row=j) frame -> transpose back to native [B,H,i,j]
        dbias = torch.cat([dbias_s, dbias_e.transpose(2, 3)], dim=1)  # [B,2H,i,j]
        dbias = rearrange(dbias, "B G L L2 -> B L L2 G")
        return dq, dk, dv, dbias, None


bidir_triangle_attention_pair_bias = BidirTriangleAttentionFn.apply

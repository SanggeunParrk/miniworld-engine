"""The fused SWA atom DiT block's ``autograd.Function`` (team-gm ``SWABlockFn``, commit 14f2c73).

It owns what the backward needs: the forward op runs with ``save=True`` and its 13 intermediates are saved alongside the
inputs, in the order :func:`~miniworld_engine.kernels.swa_dit.dispatch.swa_dit_block_bwd` takes them. The two launches are
opaque ops (``kernels._compile``); Dynamo traces through this Function and stops only at them.

Differentiable inputs: ``q``, ``mod`` (the hoisted modulation -- its gradient flows on to the conditioning and the adaLN
weight through ``swa_dit_hoist_modulation``) and the five block weights. cos / sin / seqused get no gradient.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.swa_dit.dispatch import (
    swa_dit_block_bwd,
    swa_dit_block_fwd,
    swa_dit_mod_bwd_sm80,
    swa_dit_mod_bwd_sm100,
    swa_dit_mod_fwd_sm80,
    swa_dit_mod_fwd_sm100,
    swa_dit_window_attn_bwd_sm80,
    swa_dit_window_attn_fwd_sm80,
)
from miniworld_engine.kernels.swa_dit.interface import FP32_EPS


class SWADiTBlockFunction(torch.autograd.Function):
    """``swa_dit_block`` with a backward. Called by ``interface.swa_dit_block`` only when a gradient is recorded."""

    @staticmethod
    def forward(ctx, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window):
        # The weights may be an fp32 master with bf16 activations: the kernels get casts made here, outside autograd, and the
        # weights get the backward's fp32 gradients in their own dtype (fp32 ones unrounded).
        ctx.param_dtypes = [w.dtype for w in (wqkv, wg, wo, wu, wd)]
        wqkv, wg, wo, wu, wd = (w.to(q.dtype) for w in (wqkv, wg, wo, wu, wd))
        outputs = swa_dit_block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, FP32_EPS, True)
        ctx.save_for_backward(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, *outputs[1:])
        ctx.meta = (B, half_window)
        return outputs[0]

    @staticmethod
    def backward(ctx, dy):
        B, half_window = ctx.meta
        grads = swa_dit_block_bwd(dy.contiguous(), *ctx.saved_tensors, B, half_window, FP32_EPS)
        dq, dmod, *dw = grads
        dwqkv, dwg, dwo, dwu, dwd = (g.to(dt) for g, dt in zip(dw, ctx.param_dtypes, strict=True))
        return dq, dmod, None, None, None, dwqkv, dwg, dwo, dwu, dwd, None, None


class SWADiTModulationSm80(torch.autograd.Function):
    """``swa_dit_hoist_modulation`` on the sm_80 kernels: rn(silu(c)) Wmod^T (mod_fwd) and its backward (mod_bwd); c [R, C] bf16 contiguous, Wmod [6C, C] bf16."""

    @staticmethod
    def forward(ctx, c, wmod):
        mod, a = swa_dit_mod_fwd_sm80(c, wmod, True)
        ctx.save_for_backward(c, a, wmod)
        return mod

    @staticmethod
    def backward(ctx, g):
        c, a, wmod = ctx.saved_tensors
        dc, dw = swa_dit_mod_bwd_sm80(g.contiguous(), c, a, wmod)
        return dc, dw


class SWADiTWindowAttentionSm80(torch.autograd.Function):
    """``interface.swa_dit_window_attention`` on the sm_80 kernels: the window attention of ``modules/swa_atom_attention`` over [N, S, 4, 32] bf16 q / k / v, differentiable in all three. The forward keeps
    the output and the lse for the backward, so nothing is recomputed there."""

    @staticmethod
    def forward(ctx, q, k, v, seqused):
        out, lse = swa_dit_window_attn_fwd_sm80(q, k, v, seqused)
        ctx.save_for_backward(q, k, v, out, lse, seqused)
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, k, v, out, lse, seqused = ctx.saved_tensors
        dq, dk, dv = swa_dit_window_attn_bwd_sm80(q, k, v, out, d_out.contiguous(), lse, seqused)
        return dq, dk, dv, None


class SWADiTModulationSm100(torch.autograd.Function):
    """``swa_dit_hoist_modulation`` on the sm_100a kernels: silu(c) Wmod^T (mod_fwd) and its backward (mod_bwd); c [R, C]
    bf16 contiguous with R a multiple of 128, Wmod [6C, C] bf16 or fp32 (an fp32 master: cast here, outside autograd, and its
    gradient handed back unrounded)."""

    @staticmethod
    def forward(ctx, c, wmod):
        ctx.wdtype = wmod.dtype
        wmod = wmod.to(c.dtype).contiguous()
        ctx.save_for_backward(c, wmod)
        return swa_dit_mod_fwd_sm100(c, wmod)

    @staticmethod
    def backward(ctx, g):
        c, wmod = ctx.saved_tensors
        dc, dw = swa_dit_mod_bwd_sm100(g.contiguous(), c, wmod)
        return dc, dw.to(ctx.wdtype)

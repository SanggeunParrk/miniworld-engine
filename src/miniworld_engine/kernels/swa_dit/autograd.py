"""The fused SWA atom DiT block's ``autograd.Function`` (team-gm ``SWABlockFn``, commit 14f2c73).

It owns what the backward needs: the forward op runs with ``save=True`` and its 13 intermediates are saved alongside the
inputs, in the order :func:`~miniworld_engine.kernels.swa_dit.dispatch.swa_dit_block_bwd` takes them. The two launches are
opaque ops (``kernels._compile``); Dynamo traces through this Function and stops only at them.

Differentiable inputs: ``q``, ``mod`` (the hoisted modulation -- its gradient flows on to the conditioning and the adaLN
weight through ``swa_dit_hoist_modulation``) and the five block weights. cos / sin / seqused get no gradient.
"""
from __future__ import annotations

import torch

from miniworld_engine.integrations.h100_master import is_h100
from miniworld_engine.integrations.h100_master import pack as pack_master
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.swa_dit.dispatch import (
    swa_dit_block_bwd,
    swa_dit_block_fwd,
    swa_dit_mod_bwd_sm100,
    swa_dit_mod_fwd_sm100,
)
from miniworld_engine.kernels.swa_dit.interface import FP32_EPS


class SWADiTBlockFunction(torch.autograd.Function):
    """``swa_dit_block`` with a backward. Called by ``interface.swa_dit_block`` only when a gradient is recorded."""

    @staticmethod
    def forward(ctx, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window):
        # The weights may be an fp32 master with bf16 activations: the kernels get casts made here, outside autograd, and the
        # weights get the backward's fp32 gradients in their own dtype (fp32 ones unrounded).
        ctx.param_dtypes = [w.dtype for w in (wqkv, wg, wo, wu, wd)]
        raw_weights = (wqkv, wg, wo, wu, wd)
        if q.dtype == torch.bfloat16 and any(w.dtype == torch.float32 for w in raw_weights) and is_h100(q.device):
            wqkv, wg, wo, wu, wd = pack_master(raw_weights)
        else:
            wqkv, wg, wo, wu, wd = (w.to(q.dtype) for w in raw_weights)
        outputs = swa_dit_block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, FP32_EPS, True)
        ctx.save_for_backward(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, *outputs[1:], *raw_weights)
        ctx.meta = (B, half_window)
        return outputs[0]

    @staticmethod
    def backward(ctx, dy):
        B, half_window = ctx.meta
        grads = swa_dit_block_bwd(dy.contiguous(), *ctx.saved_tensors[:-5], B, half_window, FP32_EPS)
        dq, dmod, *dw = grads
        dwqkv, dwg, dwo, dwu, dwd = (g.to(dt) for g, dt in zip(dw, ctx.param_dtypes, strict=True))
        return dq, dmod, None, None, None, dwqkv, dwg, dwo, dwu, dwd, None, None


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


def _modulation_linear_fake(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@opaque(fake=_modulation_linear_fake, name="h100_swa_modulation_linear")
def _modulation_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", enabled=False):
        return torch.mm(x, weight.t())


def _modulation_linear_bwd_fake(dy, x, weight):
    return torch.empty_like(x), torch.empty_like(weight)


@opaque(fake=_modulation_linear_bwd_fake, name="h100_swa_modulation_linear_backward")
def _modulation_linear_bwd(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # AOTAutograd's surrounding BF16 autocast must not round the master gradient.
    with torch.autocast("cuda", enabled=False):
        return torch.mm(dy, weight), torch.mm(dy.t(), x)


class H100ModulationLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return _modulation_linear(x, weight)

    @staticmethod
    def backward(ctx, dy):
        return _modulation_linear_bwd(dy, *ctx.saved_tensors)

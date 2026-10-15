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
    native_h100_cast,
    native_h100_enabled,
    native_h100_mod_bwd,
    native_h100_mod_fwd,
    swa_dit_block_bwd,
    swa_dit_block_fwd,
    swa_dit_mod_bwd_sm90,
    swa_dit_mod_bwd_sm100,
    swa_dit_mod_fwd_sm90,
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
        ctx.native_h100 = native_h100_enabled(q, True)
        if ctx.native_h100:
            wqkv, wg, wo, wu, wd = native_h100_cast(list(raw_weights), [torch.empty_like(w, dtype=q.dtype) for w in raw_weights])
        elif q.dtype == torch.bfloat16 and any(w.dtype == torch.float32 for w in raw_weights) and is_h100(q.device):
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
        if ctx.native_h100 and not dy.is_contiguous():
            raise ValueError("native H100 SWA requires contiguous upstream gradient")
        grads = swa_dit_block_bwd(dy.contiguous(), *ctx.saved_tensors[:-5], B, half_window, FP32_EPS)
        dq, dmod, *dw = grads
        if ctx.native_h100:
            dwqkv, dwg, dwo, dwu, dwd = native_h100_cast(dw, list(ctx.saved_tensors[-5:]))
        else:
            dwqkv, dwg, dwo, dwu, dwd = (g.to(dt) for g, dt in zip(dw, ctx.param_dtypes, strict=True))
        return dq, dmod, None, None, None, dwqkv, dwg, dwo, dwu, dwd, None, None


class SWADiTModulationH100(torch.autograd.Function):
    """Native SiLU and dtype boundaries around the FP32 cuBLAS modulation."""
    @staticmethod
    def forward(ctx, c, w):
        out, a, w32 = native_h100_mod_fwd(c, w)
        ctx.save_for_backward(c, a, w32, w)
        return out

    @staticmethod
    def backward(ctx, dy):
        if not dy.is_contiguous():
            raise ValueError("native SWA modulation requires contiguous upstream gradient")
        return native_h100_mod_bwd(dy, *ctx.saved_tensors)


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


def _modulation_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[..., out_features] in x's dtype and device."""
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@opaque(fake=_modulation_linear_fake, name="swa_dit_h100_modulation_linear")
def _modulation_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` with autocast off, so an fp32 master weight is not rounded to bf16."""
    with torch.autocast("cuda", enabled=False):
        return torch.mm(x, weight.t())


def _modulation_linear_bwd_fake(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """dx like x, dweight like weight."""
    return torch.empty_like(x), torch.empty_like(weight)


@opaque(fake=_modulation_linear_bwd_fake, name="swa_dit_h100_modulation_linear_backward")
def _modulation_linear_bwd(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(dx, dweight) of :func:`_modulation_linear`, with autocast off."""
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


class SWADiTModulationSm90(torch.autograd.Function):
    """``swa_dit_hoist_modulation`` on the sm_90a kernels (``cuda/swa_mod.cu``): c [R, C] and Wmod [6C, C], both bf16 or both
    fp32, contiguous; the modulation comes back fp32 and its gradient flows to c (c's dtype) and Wmod (Wmod's dtype)."""

    @staticmethod
    def forward(ctx, c, wmod):
        mod, a = swa_dit_mod_fwd_sm90(c, wmod, True)
        ctx.save_for_backward(c, a, wmod)
        return mod

    @staticmethod
    def backward(ctx, g):
        c, a, wmod = ctx.saved_tensors
        dc, dw = swa_dit_mod_bwd_sm90(g.contiguous(), c, a, wmod)
        return dc, dw

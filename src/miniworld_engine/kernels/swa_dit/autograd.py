"""The fused SWA atom DiT block's ``autograd.Function`` (team-gm ``SWABlockFn``, commit 14f2c73).

It owns what the backward needs: the forward op runs with ``save=True`` and its 13 intermediates are saved alongside the
inputs, in the order :func:`~miniworld_engine.kernels.swa_dit.dispatch.swa_dit_block_bwd` takes them. The two launches are
opaque ops (``kernels._compile``); Dynamo traces through this Function and stops only at them.

Differentiable inputs: ``q``, ``mod`` (the hoisted modulation -- its gradient flows on to the conditioning and the adaLN
weight through ``swa_dit_hoist_modulation``) and the five block weights. cos / sin / seqused get no gradient.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels._h100_master import is_h100
from miniworld_engine.kernels._h100_master import pack as pack_master
from miniworld_engine.kernels.swa_dit._common import FP32_EPS
from miniworld_engine.kernels.swa_dit.dispatch import (
    _h100_global,
    native_h100_cast,
    native_h100_mod_bwd,
    native_h100_mod_bwd_bf16,
    native_h100_mod_fwd,
    native_h100_mod_fwd_bf16,
    native_h100_stages,
    swa_dit_block_bwd,
    swa_dit_block_fwd,
)


class SWADiTBlockFunction(torch.autograd.Function):
    """``swa_dit_block`` with a backward. Called by ``interface.swa_dit_block`` only when a gradient is recorded."""

    @staticmethod
    def forward(ctx, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, portable=False):
        # portable: every stage on its Triton kernel (dispatch._PORTABLE; the "triton" backend), the weights plain casts
        # The weights may be an fp32 master with bf16 activations: the kernels get casts made here, outside autograd, and the
        # weights get the backward's fp32 gradients in their own dtype (fp32 ones unrounded).
        ctx.param_dtypes = [w.dtype for w in (wqkv, wg, wo, wu, wd)]
        raw_weights = (wqkv, wg, wo, wu, wd)
        # The kernels take an fp32 modulation; a bf16 one (a conditioning shared by the samples, projected under autocast) is
        # widened here and its gradient handed back in its own dtype.
        ctx.mod_dtype = mod.dtype
        ctx.native_h100 = not portable and (native_h100_stages(q) or _h100_global(q, half_window))
        # ... except a bf16 table under the composed H100 stages, whose slab kernels read it (and write its gradient) in bf16
        keep_bf16 = mod.dtype is torch.bfloat16 and q.dtype is torch.bfloat16 and not portable and not ctx.native_h100 and is_h100(q.device)
        mod = mod.contiguous() if mod.dtype is torch.float32 or keep_bf16 else mod.float().contiguous()
        if ctx.native_h100:
            wqkv, wg, wo, wu, wd = native_h100_cast(list(raw_weights), [torch.empty_like(w, dtype=q.dtype) for w in raw_weights])
        elif not portable and q.dtype == torch.bfloat16 and any(w.dtype == torch.float32 for w in raw_weights) and is_h100(q.device):
            wqkv, wg, wo, wu, wd = pack_master(raw_weights)
        else:
            wqkv, wg, wo, wu, wd = (w.to(q.dtype).contiguous() for w in raw_weights)
        outputs = swa_dit_block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, FP32_EPS, True, portable)
        ctx.save_for_backward(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, *outputs[1:], *raw_weights)
        ctx.meta = (B, half_window, portable)
        return outputs[0]

    @staticmethod
    def backward(ctx, dy):
        B, half_window, portable = ctx.meta
        grads = swa_dit_block_bwd(dy.contiguous(), *ctx.saved_tensors[:-5], B, half_window, FP32_EPS, portable)  # pyright: ignore[reportCallIssue]  -- the saved tensors (save_for_backward above, without the 5 raw weights) are exactly the leading parameters
        dq, dmod, *dw = grads
        if ctx.native_h100:
            dwqkv, dwg, dwo, dwu, dwd = native_h100_cast(dw, list(ctx.saved_tensors[-5:]))
        else:
            dwqkv, dwg, dwo, dwu, dwd = (g.to(dt) for g, dt in zip(dw, ctx.param_dtypes, strict=True))
        return dq, dmod.to(ctx.mod_dtype), None, None, None, dwqkv, dwg, dwo, dwu, dwd, None, None, None


class SWADiTModulationH100(torch.autograd.Function):
    """Native SiLU and dtype boundaries around the cuBLAS modulation: fp32 table (``bf16_table`` false), or a bf16 one -- of a bf16
    block whose composed stages read it as it is (``native_h100_mod_fwd_bf16``)."""

    @staticmethod
    def forward(ctx, c, w, bf16_table=False):
        ctx.bf16_table = bf16_table
        if bf16_table:
            out, a, wb = native_h100_mod_fwd_bf16(c, w)
            ctx.save_for_backward(c, a, wb, w)
            return out
        out, a, w32 = native_h100_mod_fwd(c, w)
        ctx.save_for_backward(c, a, w32, w)
        return out

    @staticmethod
    def backward(ctx, dy):
        bwd = native_h100_mod_bwd_bf16 if ctx.bf16_table else native_h100_mod_bwd
        return (*bwd(dy.contiguous(), *ctx.saved_tensors), None)  # e.g. the expanded gradient of a sum

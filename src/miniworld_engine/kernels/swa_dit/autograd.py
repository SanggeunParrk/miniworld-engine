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
)
from miniworld_engine.kernels.swa_dit.interface import FP32_EPS


class SWADiTBlockFunction(torch.autograd.Function):
    """``swa_dit_block`` with a backward. Called by ``interface.swa_dit_block`` only when a gradient is recorded."""

    @staticmethod
    def forward(ctx, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window):
        outputs = swa_dit_block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, FP32_EPS, True)
        ctx.save_for_backward(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, *outputs[1:])
        ctx.meta = (B, half_window)
        return outputs[0]

    @staticmethod
    def backward(ctx, dy):
        B, half_window = ctx.meta
        grads = swa_dit_block_bwd(dy.contiguous(), *ctx.saved_tensors, B, half_window, FP32_EPS)
        dq, dmod, dwqkv, dwg, dwo, dwu, dwd = grads
        return dq, dmod, None, None, None, dwqkv, dwg, dwo, dwu, dwd, None, None

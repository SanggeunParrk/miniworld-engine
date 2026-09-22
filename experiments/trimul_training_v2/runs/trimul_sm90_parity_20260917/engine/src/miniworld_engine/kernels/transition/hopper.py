"""Existing Hopper Transition implementations with residual in both epilogues.

Small widths reuse the hand-CUDA b2b forward. Wide widths reuse the CuTe
LN-folded expand followed by a rounded squeeze/residual WGMMA epilogue.
Large D128/256 training saves the native b2b's normalized operand and reuses it
in stacked backward. Wide CuTe retains its recompute algorithm. Both fold the
identity gradient into LN dx; it never contributes to gamma/beta gradients.
"""
from __future__ import annotations

import torch
from miniworld_engine import settings
from miniworld_engine.autotune.shape_key import both_key, rows_of
from miniworld_engine.kernels._compile import opaque


def supported(x, n):
    """Static dispatch contract; unsupported shapes retain the Triton split path."""
    from miniworld_engine.modules.dispatch import is_sm90
    return (x.is_cuda and x.dtype == torch.bfloat16 and is_sm90(x.device)
            and n == 4 and x.shape[-1] in (128, 256, 384, 512, 768)
            and (x.numel() // x.shape[-1]) % 128 == 0)


def enabled(x, n):
    policy = settings.current()
    return (policy.engine_backend != "triton" and not policy.transition_force_split
            and policy.transition_h100_residual and supported(x, n)
            and (x.shape[-1] > 256 or policy.transition_cuda_b2b)
            # Wide token workloads historically favor the Triton split path.
            # Explicit CuTe still exposes the native path for measurement.
            and (x.shape[-1] <= 256 or x.numel() // x.shape[-1] >= 16384))


def _forward_fake(x, gamma, beta, wa, wb, ws, eps, shape_key, use_b2b, save_xn):
    return (torch.empty_like(x), x.new_empty(x.shape[0], dtype=torch.float32),
            x.new_empty(x.shape[0], dtype=torch.float32),
            torch.empty_like(x) if save_xn else x.new_empty(0))


@opaque(fake=_forward_fake, name="transition_residual_sm90_fwd")
def _forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
             wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor,
             eps: float, shape_key: int, use_b2b: bool, save_xn: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
    rstd, c1 = stats_triton(x, eps, shape_key=shape_key)
    if use_b2b and x.shape[-1] <= 256:
        from miniworld_engine.kernels.transition.cuda import transition_b2b_fwd, transition_b2b_fwd_saved
        if save_xn:
            out, xn = transition_b2b_fwd_saved(x, rstd, c1, gamma, beta, wa, wb, ws)
        else:
            out = transition_b2b_fwd(x, rstd, c1, gamma, beta, wa, wb, ws)
            xn = x.new_empty(0)
    else:
        from miniworld_engine.kernels.transition.cute.gemm_transition_swiglu import transition_expand_swiglu_cute
        from miniworld_engine.kernels.transition.cute.squeeze_residual import squeeze_residual
        h = transition_expand_swiglu_cute(x, gamma, beta, wa, wb, eps, stats=(rstd, c1))
        out = squeeze_residual(h, ws, x)
        xn = x.new_empty(0)
    return out, rstd, c1, xn


class HopperResidualTransition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps, use_b2b, save_xn):
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        gamma, beta, wa, wb, ws = [t.to(x.dtype).contiguous() for t in (gamma, beta, wa, wb, ws)]
        key = both_key(rows_of(shape))
        y, rstd, c1, xn = _forward(x, gamma, beta, wa, wb, ws, eps, key, use_b2b, save_xn)
        ctx.save_for_backward(x, rstd, c1, gamma, beta, wa, wb, ws, xn)
        ctx.has_xn = save_xn
        ctx.shape, ctx.key, ctx.eps = shape, key, eps
        ctx.b2b = use_b2b and shape[-1] <= 256
        ctx.backward_backend = settings.current().transition_large_d_training or settings.current().transition_cute_backward
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        from miniworld_engine.kernels.transition.triton.fused import _fused_bwd
        x, rstd, c1, gamma, beta, wa, wb, ws, xn = ctx.saved_tensors
        if ctx.b2b:
            grads = _fused_bwd(dy.contiguous(), x, rstd, c1, gamma, beta, wa, wb, ws,
                               xn if ctx.has_xn else None, ctx.eps, ctx.has_xn, list(ctx.shape), ctx.key, True)
        else:
            from miniworld_engine.kernels.transition.cute.fused import _cute_bwd
            grads = _cute_bwd(dy.contiguous(), x, rstd, c1, gamma, beta, wa, wb, ws,
                             ctx.eps, ctx.backward_backend, list(ctx.shape), ctx.key, True)
        return (*grads, None, None, None)


def transition_residual_hopper(x, gamma, beta, wa, wb, ws, eps=1e-5, *, use_b2b=True, save_xn=None):
    if not supported(x, wa.shape[0] // x.shape[-1]):
        raise ValueError("Hopper Transition requires aligned BF16 n=4 supported-width inputs")
    if save_xn is None:
        save_xn = (settings.current().transition_h100_save_xn and use_b2b
                   and x.shape[-1] in (128, 256) and x.numel() // x.shape[-1] >= 16384
                   and torch.is_grad_enabled()
                   and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws)))
    if save_xn and (not use_b2b or x.shape[-1] > 256):
        raise ValueError("saved xn currently requires the native b2b D128/256 path")
    return HopperResidualTransition.apply(x, gamma, beta, wa, wb, ws, eps, use_b2b, save_xn)

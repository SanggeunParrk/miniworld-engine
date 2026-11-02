"""H100 (sm_90) CUDA wiring of ``layer_norm_linear`` for the pair bias of the token DiT: d_pair 128 -> 16 heads, no biases, a pair of side * side rows.

Every function answers ``None`` / ``False`` when no CUDA kernel serves the call (another card, a dtype / width / layout outside the kernel's contract,
a CPU tensor, a ``torch.compile`` trace) and the caller runs the PyTorch statements. Nothing here launches Triton.
"""

from __future__ import annotations

import math

import torch

from miniworld_engine.kernels._device import on_h100


class _LayerNormProject(torch.autograd.Function):
    """``Linear(LayerNorm(x) * w)`` for d_pair 128 -> 16 heads on the token DiT's CUDA pair-bias kernels (``token_dit_train_rows.cu``): one
    pass reads the pair rows (LayerNorm statistics, the projection on TF32 tensor cores, the head-major result written once, LN(x) never
    stored); the backward is one pass too (the LayerNorm backward and the weight-gradient partials), summed in a fixed order."""

    @staticmethod
    def forward(ctx, x2, ln_weight, proj_weight, eps, side):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

        R, d = x2.shape
        h = proj_weight.shape[0]
        wf = (proj_weight.float() * ln_weight.float()[None, :]).contiguous()
        bias = torch.empty(h, side, side, device=x2.device, dtype=x2.dtype)
        pst = torch.empty(R, 2, device=x2.device, dtype=torch.float32)
        ext().pair_bias(x2, wf, None, bias, pst, side, float(eps), None, 1.0)
        ctx.save_for_backward(x2, pst, wf, ln_weight, proj_weight)
        ctx.side = side
        return bias.view(h, R).t()

    @staticmethod
    def backward(ctx, dout):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

        x2, pst, wf, ln_weight, proj_weight = ctx.saved_tensors
        T = ext()
        R, d = x2.shape
        h = proj_weight.shape[0]
        db = dout.t().contiguous()  # [h, R]
        dz = torch.empty_like(x2)
        part = torch.empty(T.pair_grid(R), h * d, device=x2.device, dtype=torch.float32)
        T.pair_bias_bwd(db, x2, pst, wf, dz, part)
        dwf = part.sum(0).view(h, d)
        dproj = dwf * ln_weight.float()[None, :]
        dln = (dwf * proj_weight.float()).sum(0)
        return dz, dln.to(ln_weight.dtype), dproj.to(proj_weight.dtype), None, None


def layer_norm_linear_serves(x, ln_weight, proj_weight) -> bool:
    """d_pair 128 -> 16 heads, bf16 / fp32, rows forming a square (one pair of side * side rows), sm_90."""
    if (
        x.shape[-1] != 128 or proj_weight.shape != (16, 128) or ln_weight.shape != (128,)
        or x.dtype not in (torch.bfloat16, torch.float32) or not on_h100(x, ln_weight, proj_weight)
    ):
        return False
    rows = x.numel() // 128
    return math.isqrt(rows) ** 2 == rows and x.stride(-1) == 1


def layer_norm_linear(x, ln_weight, proj_weight, eps):
    """``Linear(LayerNorm(x))`` (d_pair 128 -> 16, no biases, a pair of side * side rows) on the CUDA pair-bias kernels, or None."""
    if not layer_norm_linear_serves(x, ln_weight, proj_weight):
        return None
    rows = x.numel() // 128
    side = math.isqrt(rows)
    x2 = x.reshape(rows, 128)
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (x, ln_weight, proj_weight))
    if not grad and x.dtype is torch.bfloat16 and side % 16 == 0:
        # no gradient, bf16: the pair-bias kernel of the AttentionPairBias stack (``apb_rows.cu``: LayerNorm + the bf16 projection with the LayerNorm
        # weight folded into the projection weight, head-major result)
        from miniworld_engine.kernels.augmented_attention.cuda import apb

        wf = (proj_weight.float() * ln_weight.float()[None, :]).to(torch.bfloat16).contiguous()
        bias = apb.pair_bias(x2 if x2.is_contiguous() else x2.contiguous(), wf, None, side, float(eps), 0.0)
        return bias.view(16, rows).t().view(*x.shape[:-1], 16)
    out = _LayerNormProject.apply(x2, ln_weight, proj_weight, eps, side)
    return out.view(*x.shape[:-1], 16)

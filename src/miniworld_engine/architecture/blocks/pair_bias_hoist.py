"""Every token-DiT block's pair bias from ONE LayerNorm and ONE GEMM, forward and backward.

Each ``AugmentedAttentionPairBias`` computes ``bias_b = to_bias_b(ln_pair_b(pair))``. ``ln_pair_b`` has no offset,
so ``ln_pair_b(pair) = LN0(pair) * gamma_b`` with ``LN0`` the plain normalization, and

    bias_b = LN0(pair) @ (gamma_b * W_b)^T

The normalization is the same for every block (only gamma differs), so all blocks' biases are one GEMM of
``LN0(pair)`` against the stacked, gamma-folded weights. Per block and training step that removes a LayerNorm over the
whole pair tensor (twice under per-block checkpointing), a 128 -> H projection, their backward, and the accumulation of
every block's pair gradient into ``pair.grad`` -- 24 full-size adds of a [L, L, 128] fp32 tensor.

The backward receives each block's bias gradient separately (one autograd output per block, so nothing is scattered
through views), then does one GEMM for d LN0, one for the folded weights' gradient, one LayerNorm backward, and
recovers d gamma_b and d W_b from the folded gradient by the product rule:

    dW_b[h, c] = dWf_b[h, c] * gamma_b[c]        dgamma_b[c] = sum_h dWf_b[h, c] * W_b[h, c]

Same arithmetic as the per-block path (LayerNorm, then a matmul, in the tensors' own dtype), so results agree to
rounding: the GEMM sums the same products, with gamma applied to the weight instead of the activation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class _PairBiasAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, pair: Tensor, eps: float, n_head: int, head_major: bool, *params: Tensor
    ):
        gammas, weights = params[0::2], params[1::2]
        dp = pair.shape[-1]
        x = pair.reshape(-1, dp)
        ln0 = F.layer_norm(x, (dp,), None, None, eps)
        wf = torch.cat(
            [w * g.to(w.dtype)[None, :] for g, w in zip(gammas, weights, strict=False)],
            0,
        )  # [nb * H, dp]
        lead = pair.shape[:-1]
        if head_major:  # [nb * H, N]: views, no copy
            b = wf.to(ln0.dtype) @ ln0.t()
            outs = tuple(
                b[i * n_head : (i + 1) * n_head].view(n_head, *lead)
                for i in range(len(weights))
            )
        else:
            b = ln0 @ wf.t().to(ln0.dtype)  # [N, nb * H]
            outs = tuple(
                b[:, i * n_head : (i + 1) * n_head].contiguous().view(*lead, n_head)
                for i in range(len(weights))
            )
        ctx.save_for_backward(pair, *params)
        ctx.eps, ctx.n_head, ctx.head_major = eps, n_head, head_major
        return outs

    @staticmethod
    def backward(ctx, *grads: Tensor | None):
        pair, *params = ctx.saved_tensors
        gammas, weights = params[0::2], params[1::2]
        eps, h = ctx.eps, ctx.n_head
        dp = pair.shape[-1]
        x = pair.reshape(-1, dp)
        n = x.shape[0]
        if ctx.head_major:  # [nb * H, N]
            g = torch.cat(
                [
                    gr.reshape(h, n) if gr is not None else x.new_zeros(h, n)
                    for gr in grads
                ],
                0,
            )
        else:  # [N, nb * H]
            g = torch.cat(
                [
                    gr.reshape(n, h) if gr is not None else x.new_zeros(n, h)
                    for gr in grads
                ],
                1,
            )
        var, mean = torch.var_mean(x, dim=-1, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        ln0 = (x - mean) * rstd
        wf = torch.cat(
            [
                w * gm.to(w.dtype)[None, :]
                for gm, w in zip(gammas, weights, strict=False)
            ],
            0,
        )
        if ctx.head_major:
            dln0 = g.t() @ wf.to(g.dtype)  # [N, dp]
            dwf = (g @ ln0).to(wf.dtype)  # [nb * H, dp]
        else:
            dln0 = g @ wf.to(g.dtype)
            dwf = (g.t() @ ln0).to(wf.dtype)
        # LayerNorm without affine: dx = rstd (dy - mean(dy) - y mean(dy y))
        dx = rstd * (
            dln0
            - dln0.mean(-1, keepdim=True)
            - ln0 * (dln0 * ln0).mean(-1, keepdim=True)
        )
        dparams: list[Tensor | None] = []
        for i, (gm, w) in enumerate(zip(gammas, weights, strict=False)):
            d = dwf[i * h : (i + 1) * h]
            dparams.append(
                (d * w).sum(0).to(gm.dtype) if ctx.needs_input_grad[4 + 2 * i] else None
            )
            dparams.append(
                (d * gm.to(d.dtype)[None, :]).to(w.dtype)
                if ctx.needs_input_grad[5 + 2 * i]
                else None
            )
        return (
            dx.view_as(pair) if ctx.needs_input_grad[0] else None,
            None,
            None,
            None,
            *dparams,
        )


def pair_bias_all(
    attention_modules, pair: Tensor, head_major: bool = False
) -> tuple[Tensor, ...]:
    """``(to_bias(ln_pair(pair)) for each module)`` from one LayerNorm and one GEMM, differentiable in ``pair`` and in
    every module's ``ln_pair.weight`` and ``to_bias.weight``. The modules must share ``eps`` and head count.
    ``head_major``: each bias as [H, *pair.shape[:-1]] (a view of the one GEMM output) instead of [..., H]."""
    mods = list(attention_modules)
    eps = mods[0].ln_pair.eps
    n_head = mods[0].to_bias.weight.shape[0]
    for m in mods:
        if (
            m.ln_pair.eps != eps
            or m.to_bias.weight.shape[0] != n_head
            or getattr(m.ln_pair, "bias", None) is not None
        ):
            raise ValueError(
                "pair_bias_all: blocks must share eps and head count, and ln_pair must have no offset"
            )
    params: list[Tensor] = []
    for m in mods:
        params += [m.ln_pair.weight, m.to_bias.weight]
    return _PairBiasAll.apply(pair, eps, n_head, head_major, *params)

"""Every bias-only DiT block's pair bias at once: ``pair_bias_all(blocks, pair)``.

Each ``BiasOnlyAttention`` computes ``bias_b = to_bias_b(ln_pair_b(pair))``. ``ln_pair`` has no offset, so with LN0 the LayerNorm
without affine and the fold W'_b = W_b diag(gamma_b)

    bias_b = LN0(pair) W'_b^T

-- the normalization is the same for every block, only gamma differs: every block's bias is one LayerNorm and one GEMM against the
stacked folded weights. Pass block b its bias (``block(single, cond, pair, mask, bias=biases[b])``) and the block touches neither
``ln_pair`` nor ``to_bias``; the gradients reach them, and the pair, through this op. Same parameters, the same result up to
rounding (the GEMM sums the same products, gamma applied to the weight instead of the activation). Under activation checkpointing
call it OUTSIDE the checkpointed blocks.

On B200 with an engine implementation (bf16, or fp32 on TF32 tensor cores) this is ``integrations.bias_only_dit_hoist`` (one
LayerNorm rows pass, cuBLAS GEMMs, the blocks' bias gradients written into one buffer); elsewhere the PyTorch fold below
(autograd differentiates it), with the same API.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.integrations import bias_only_dit_hoist as _fused
from miniworld_engine.modules.exceptions import ImplementationType


def _attentions(blocks):
    return [m.attention if hasattr(m, "attention") else m for m in blocks]


def _reference(attns, pair, eps):
    """The fold in PyTorch: LN0 in fp32 (as the engine's LayerNorm computes), one matmul against W'_all, per-block head-major views
    [B, H, L, L] of the [B, L, L, N] product."""
    H, dp = attns[0].to_bias.weight.shape[0], pair.shape[-1]
    ln0 = F.layer_norm(pair.float(), (dp,), None, None, eps).to(pair.dtype)
    wf = torch.cat([a.to_bias.weight * a.ln_pair.weight.to(a.to_bias.weight.dtype)[None, :] for a in attns], 0)   # [N, dp]
    b = torch.matmul(ln0, wf.t().to(ln0.dtype))                                                                   # [B, L, L, N]
    return tuple(b[..., i * H:(i + 1) * H].permute(0, 3, 1, 2) for i in range(len(attns)))


def pair_bias_all(blocks, pair: torch.Tensor, *, implementation: ImplementationType | None = None) -> tuple[torch.Tensor, ...]:
    """``tuple(to_bias(ln_pair(pair)).permute(0, 3, 1, 2) for each block)`` -- [B, H, L, L] each -- from one LayerNorm and one GEMM,
    differentiable in ``pair`` and in every block's ``ln_pair.weight`` / ``to_bias.weight``.

    ``blocks``: ``BiasOnlyDiTBlock``s (or their ``attention`` modules); they must share the LayerNorm eps and the head count, and
    ``ln_pair`` must have no offset. ``implementation`` defaults to the first block's (an attention module alone: PYTORCH)."""
    mods = list(blocks)
    if not mods:
        raise ValueError("pair_bias_all: no blocks")
    attns = _attentions(mods)
    impl = implementation if implementation is not None else getattr(mods[0], "implementation", ImplementationType.PYTORCH)
    eps, H = attns[0].ln_pair.eps, attns[0].to_bias.weight.shape[0]
    for a in attns:
        if (a.ln_pair.eps != eps or a.to_bias.weight.shape[0] != H or getattr(a.ln_pair, "bias", None) is not None
                or getattr(a.to_bias, "bias", None) is not None or a.ln_pair.weight is None):
            raise ValueError("pair_bias_all: the blocks must share eps and head count, ln_pair must have a weight and no offset, "
                             "to_bias no bias")
    if _fused.serves(ImplementationType(impl), attns, pair):
        return _fused.pair_bias_all(attns, pair)
    return _reference(attns, pair, eps)


__all__ = ["pair_bias_all"]

"""Public entry points for the bias-only attention family.

Bias-only attention is the softmax-over-pair-bias attention of
``modules/attention_pair_bias.py``: the logits ARE the pair bias, so there is no query-key
product and the kernel reads only ``v`` and ``bias``. Alongside it live the two gate epilogues
that follow the attention -- the fused gate+to_out GEMM and the standalone one-pass
sigmoid-multiply -- because ``dispatch.py`` picks between them per GPU and per shape.

``bias_only_attention`` is the family's default dispatch: on an A100 the hand-CUDA path of
``cuda/sm80.py`` where it serves the call (bf16, head width 32 / 48 / 64, L a multiple of 128 up to
1024), the Triton kernels otherwise (``triton_bias_only_attention`` is always the Triton one).

This module is the family's public door: importers name it rather than the ``triton/`` layout.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.bias_only_attention.triton.main import (
    triton_bias_only_attention,
)

# Re-exported, not owned: the kernels behind them are gated_projection's (its dispatch picks the A100 hand-CUDA kernels where they
# apply, else the Triton ones: `fused_gate_out`'s Triton kernel is this family's `triton/gate_out.py`). This family names them
# because `dispatch.py` here is what picks between the two gate epilogues.
from miniworld_engine.kernels.gated_projection.interface import (
    fused_gate_out,
    sigmoid_gate_fused,
)


def bias_only_attention(v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``softmax(bias) v`` over ``v`` ``(B, H, L, L, D)`` and ``bias`` ``(B, H, L, L)`` (the ``t`` axis of ``v`` shares one set of weights): the A100 hand-CUDA path where it serves the
    call and wins (not a training call at L = 128, where Triton is faster), else the Triton kernels. Differentiable in both."""
    from miniworld_engine.kernels.bias_only_attention.cuda import sm80

    if sm80.wanted(v, bias):
        return sm80.bias_only_attention_sm80(v, bias)
    return triton_bias_only_attention(v, bias)


__all__ = [
    "bias_only_attention",
    "fused_gate_out",
    "sigmoid_gate_fused",
    "triton_bias_only_attention",
]

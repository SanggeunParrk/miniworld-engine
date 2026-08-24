"""Public entry points for the bias-only attention family.

Bias-only attention is the softmax-over-pair-bias attention of
``modules/attention_pair_bias.py``: the logits ARE the pair bias, so there is no query-key
product and the kernel reads only ``v`` and ``bias``. Alongside it live the two gate epilogues
that follow the attention -- the fused gate+to_out GEMM and the standalone one-pass
sigmoid-multiply -- because ``dispatch.py`` picks between them per GPU and per shape.

This module is the family's public door: importers name it rather than the ``triton/`` layout.
"""

from __future__ import annotations

from miniworld_engine.kernels.bias_only_attention.triton.gate_out import (
    fused_gate_out,
    sigmoid_gate_fused,
)
from miniworld_engine.kernels.bias_only_attention.triton.main import (
    triton_bias_only_attention,
)

__all__ = [
    "fused_gate_out",
    "sigmoid_gate_fused",
    "triton_bias_only_attention",
]

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
)
from miniworld_engine.kernels.bias_only_attention.triton.main import (
    triton_bias_only_attention,
)

# Re-exported, not owned: the kernels behind it are gated_projection's. This family names it
# because `dispatch.py` here is what picks between the two gate epilogues.
from miniworld_engine.kernels.gated_projection.interface import (
    sigmoid_gate_fused,
)

__all__ = [
    "fused_gate_out",
    "sigmoid_gate_fused",
    "triton_bias_only_attention",
]

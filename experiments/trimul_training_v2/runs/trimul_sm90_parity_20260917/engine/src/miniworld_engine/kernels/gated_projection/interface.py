"""Public entry point for the gated-projection family.

Gated projection is ``sigmoid(gate) * x`` followed by the output projection, fused so the
gated activation never round-trips through HBM; the exported name is the autograd-aware
entry (fwd+bwd), not the ``torch.autograd.Function`` behind it.

``sigmoid_gate_fused`` is the standalone one-pass gate: the autograd-aware entry over the
``_sigmul_fwd``/``_sigmul_bwd`` kernels this family owns. ``bias_only_attention`` re-exports
it, because its ``dispatch.py`` is what chooses between it and ``fused_gate_out``.
"""

from __future__ import annotations

from miniworld_engine.kernels.gated_projection.triton.main import (
    sigmoid_gate_fused,
    triton_gated_projection,
)

__all__ = ["sigmoid_gate_fused", "triton_gated_projection"]

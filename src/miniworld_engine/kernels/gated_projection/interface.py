"""Public entry point for the gated-projection family.

Gated projection is ``sigmoid(gate) * x`` followed by the output projection, fused so the
gated activation never round-trips through HBM; the exported name is the autograd-aware
entry (fwd+bwd), not the ``torch.autograd.Function`` behind it.

Nothing outside the family imports this yet -- the flat ``kernels`` bridge does not name it --
but every family gets an ``interface.py`` so the layout rule holds without exceptions.
"""

from __future__ import annotations

from .triton.main import triton_gated_projection

__all__ = ["triton_gated_projection"]

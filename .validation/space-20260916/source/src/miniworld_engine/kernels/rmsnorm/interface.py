"""Public entry point for the rmsnorm family.

RMSNorm is ``x / sqrt(mean(x^2) + eps) * weight`` over the last axis -- LayerNorm with the mean
removed. Two call sites in this repo were reaching for ``F.rms_norm`` directly, which is three
HBM passes and holds the normalized activation for the backward:
``modules/swa_atom_attention`` (q and k, no learnable weight) and
``kernels/triangle_attention/whole_op.py`` (with one). The exported name is the autograd-aware
entry, not the ``torch.autograd.Function`` behind it.
"""

from __future__ import annotations

from miniworld_engine.kernels.rmsnorm.triton.main import (
    triton_rmsnorm,
    triton_rmsnorm_adamod,
)

__all__ = ["triton_rmsnorm", "triton_rmsnorm_adamod"]

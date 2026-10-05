"""Public entry point for the rmsnorm family.

RMSNorm is ``x / sqrt(mean(x^2) + eps) * weight`` over the last axis -- LayerNorm with the mean
removed. Two call sites in this repo were reaching for ``F.rms_norm`` directly, which is three
HBM passes and holds the normalized activation for the backward:
``modules/swa_atom_attention`` (q and k, no learnable weight) and
``kernels/triangle_attention/whole_op.py`` (with one). The exported name is the autograd-aware
entry, not the ``torch.autograd.Function`` behind it.

``rms_norm_modulation`` is the dispatching entry of the RMSNorm + adaLN modulation (``ops.rms_norm_modulation``): on an A100 the hand-CUDA kernels of
``cuda/sm80.py`` where they serve the call (atom width 128, bf16), the Triton kernels elsewhere and for everything they refuse (``MINIWORLD_NORMS_SM80=0``
keeps Triton).  ``triton_rmsnorm`` / ``triton_rmsnorm_adamod`` are the Triton entries by name.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.rmsnorm.cuda import sm80 as _sm80
from miniworld_engine.kernels.rmsnorm.triton.main import (
    triton_rmsnorm,
    triton_rmsnorm_adamod,
)

__all__ = ["rms_norm_modulation", "triton_rmsnorm", "triton_rmsnorm_adamod"]


def rms_norm_modulation(
    q: torch.Tensor,
    c: torch.Tensor,
    w_scale: torch.Tensor,
    w_shift: torch.Tensor,
    w_gate: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rmsnorm(q) * (1 + c @ w_scale^T) + c @ w_shift^T, c @ w_gate^T)``, forward and backward; ``c`` is the activated conditioning (see ``triton_rmsnorm_adamod``)."""
    if _sm80.supports_adamod(q, c, w_scale, w_shift, w_gate, weight):
        return _sm80.rmsnorm_adamod(q, c, w_scale, w_shift, w_gate, weight, eps)
    return triton_rmsnorm_adamod(q, c, w_scale, w_shift, w_gate, weight, eps)

"""Whole-op wrapper for the SwiGLU Transition layer.

Exposes :func:`transition` — the full layer op (``LN_in → SwiGLU(expand_a, expand_b)
→ squeeze``), weights-as-args and autograd-transparent, with the fused SwiGLU kernel
inside. A model layer holds the weights as ``nn.Parameter`` and makes one call.

Weight convention mirrors ``nn.Linear`` (``weight`` is ``(out, in)``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def transition(
    x: torch.Tensor,                     # (..., d_hidden)
    *,
    ln_in_weight: torch.Tensor,          # (d_hidden,)
    ln_in_bias: torch.Tensor,            # (d_hidden,)
    expand_a_weight: torch.Tensor,       # (n*d_hidden, d_hidden)
    expand_b_weight: torch.Tensor,       # (n*d_hidden, d_hidden)
    squeeze_weight: torch.Tensor,        # (d_hidden, n*d_hidden)
    n: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused SwiGLU transition — whole-op call. Returns the same shape as ``x``.

    Autograd-transparent: back-prop produces gradients for ``x`` and every weight.
    """
    from .triton.main import triton_transition

    x_n = F.layer_norm(x, (x.shape[-1],), ln_in_weight, ln_in_bias, eps)
    return triton_transition(x_n, expand_a_weight, expand_b_weight, squeeze_weight, n)

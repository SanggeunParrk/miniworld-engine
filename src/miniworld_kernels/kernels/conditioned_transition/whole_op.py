"""Whole-op wrapper for the conditioned SwiGLU transition tail.

Exposes :func:`conditioned_transition` — the post-normalization op ``SwiGLU(expand_a,
expand_b) → squeeze → sigmoid(to_scale(cond))·out``, weights-as-args and
autograd-transparent, with the fused SwiGLU kernel inside.

Boundary note: the **adaptive** normalization (AdaLN: rotation / MP research variants)
is a reusable *model* layer with no kernel content and stays in the model repo. This op
takes the ALREADY-normalized ``x`` plus ``cond`` and owns only the kernel-backed tail +
the conditioning gate — mirroring the scope of the dedicated conditioned_transition
kernel, but composed safely from the (bf16-correct) transition kernel + a torch gate.

Weight convention mirrors ``nn.Linear`` (``weight`` is ``(out, in)``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def conditioned_transition(
    x: torch.Tensor,                     # (..., d_hidden) — already (adaptive-)normalized
    cond: torch.Tensor,                  # (..., d_cond)
    *,
    expand_a_weight: torch.Tensor,       # (n*d_hidden, d_hidden)
    expand_b_weight: torch.Tensor,       # (n*d_hidden, d_hidden)
    squeeze_weight: torch.Tensor,        # (d_hidden, n*d_hidden)
    to_scale_weight: torch.Tensor,       # (d_hidden, d_cond)
    to_scale_bias: torch.Tensor,         # (d_hidden,)
    n: int,
) -> torch.Tensor:
    """Conditioned SwiGLU tail — whole-op call. Returns ``(..., d_hidden)``.

    Autograd-transparent: back-prop produces gradients for ``x``, ``cond`` and every
    weight/bias.
    """
    from miniworld_kernels.kernels.transition.triton.main import triton_transition

    out = triton_transition(x, expand_a_weight, expand_b_weight, squeeze_weight, n)
    scale = F.linear(cond, to_scale_weight, to_scale_bias)
    return torch.sigmoid(scale) * out

"""PyTorch reference for the ConditionedTransition tail kernel.

Mirrors ``kernels.cond_transition_train`` (``ConditionedTransitionTailFunction``, and
its forward-only twin ``cond_transition_inference_dispatch``), i.e. the post-AdaLN tail
of AF3 Algorithm 25. Same weights-as-args order, same layout, same single output::

    a     = x @ expand_a_weight.T                  # (M, n*D)
    b     = x @ expand_b_weight.T                  # (M, n*D)
    h     = silu(a) * b = a * sigmoid(a) * b       # SwiGLU, (M, n*D)
    out   = h @ squeeze_weight.T                   # (M, D)
    scale = cond @ to_scale_weight.T + to_scale_bias   # (M, D)
    y     = sigmoid(scale) * out                   # conditioning gate, (M, D)

The conditioning gate is an ELEMENTWISE per-feature multiplier: ``cond`` is projected
d_cond -> d_hidden by the biased ``to_scale`` linear, squashed by a sigmoid, and applied
to the squeeze output — it never touches ``a``/``b``.

Where the AdaLN sits: OUTSIDE this op. ``x`` arrives ALREADY adaptively normalised —
``modules.ConditionedTransition`` runs ``x = ada_ln_in(x, cond)`` (Alg. 25 step 1) before
calling the kernel, and the same ``cond`` is then reused here for the output gate. This
reference deliberately stops at the same boundary as the kernel, so it owns no AdaLN
parameters.

Weight convention mirrors ``nn.Linear`` (``weight`` is ``(out, in)``). Parameters default
to fp32 because that is the precision the kernel and its bench run at (fp32 storage,
TF32 tensor cores); pass ``dtype=torch.bfloat16`` to compare against a bf16 kernel run.

The reference is the plain functional form (:func:`conditioned_transition_pytorch`): a kernel is checked against it on both the forward output and the backward gradients.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def conditioned_transition_pytorch(
    x: torch.Tensor,  # (..., D) — already AdaLN-normalised
    cond: torch.Tensor,  # (..., DC)
    expand_a_weight: torch.Tensor,  # (n*D, D)
    expand_b_weight: torch.Tensor,  # (n*D, D)
    squeeze_weight: torch.Tensor,  # (D, n*D)
    to_scale_weight: torch.Tensor,  # (D, DC)
    to_scale_bias: torch.Tensor,  # (D,)
    n: int | None = None,
) -> torch.Tensor:
    """Compute ``sigmoid(to_scale(cond)) * squeeze(silu(a) * b)``. Returns ``(..., D)``.

    ``n`` is accepted for parity with the whole-op signature and is unused: the expansion
    factor is implied by the weight shapes.
    """
    a = F.linear(x, expand_a_weight)
    b = F.linear(x, expand_b_weight)
    h = F.silu(a) * b
    out = F.linear(h, squeeze_weight)
    scale = F.linear(cond, to_scale_weight, to_scale_bias)
    return torch.sigmoid(scale) * out

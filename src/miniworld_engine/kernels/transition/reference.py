"""PyTorch reference for the fused SwiGLU Transition kernel.

Mirrors the whole-op entry point ``kernels.transition.whole_op.transition`` (and the
kernel it wraps, ``kernels.triton_transition_fused``): LayerNorm folded into a SwiGLU
expand pair followed by the squeeze projection, over the last dimension::

    xn = LayerNorm(x; ln_in_weight, ln_in_bias, eps)   # (..., D)
    a  = xn @ expand_a_weight.T                        # (..., n*D)
    b  = xn @ expand_b_weight.T                        # (..., n*D)
    h  = silu(a) * b = a * sigmoid(a) * b              # SwiGLU, (..., n*D)
    y  = h @ squeeze_weight.T                          # (..., D)
    y  = y + x                                         # the residual, unconditional

Weight convention mirrors ``nn.Linear`` (``weight`` is ``(out, in)``), matching the
kernel argument layout. The residual is unconditional here because it is unconditional in
the op: every launcher -- ``modules.Transition``, ``ops.transition``, the fused kernel's own
epilogue -- returns ``x + transition(x)``. The bare no-LayerNorm, no-residual SwiGLU FFN is a
different op with a different reference.

The reference is the plain functional form (:func:`transition_pytorch`): a kernel is checked against it on both the forward output and the backward gradients.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def transition_pytorch(
    x: torch.Tensor,  # (..., D)
    ln_in_weight: torch.Tensor,  # (D,)
    ln_in_bias: torch.Tensor,  # (D,)
    expand_a_weight: torch.Tensor,  # (n*D, D)
    expand_b_weight: torch.Tensor,  # (n*D, D)
    squeeze_weight: torch.Tensor,  # (D, n*D)
    n: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute ``x + squeeze(silu(a) * b)`` over ``LayerNorm(x)``. Returns ``x``'s shape.

    ``n`` is carried only for signature parity with the kernel (the expansion factor is
    already implied by the weight shapes). The LayerNorm affine params are cast to the
    activation dtype: the module keeps them fp32-pinned while the trunk runs bf16, and
    the kernel likewise upcasts internally and normalises back to ``x.dtype``.
    """
    xn = F.layer_norm(
        x,
        (x.shape[-1],),
        ln_in_weight.to(x.dtype),
        ln_in_bias.to(x.dtype),
        eps,
    )
    a = F.linear(xn, expand_a_weight)
    b = F.linear(xn, expand_b_weight)
    h = F.silu(a) * b
    out = F.linear(h, squeeze_weight)
    return out + x

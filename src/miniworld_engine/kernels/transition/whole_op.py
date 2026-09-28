"""Whole-op wrapper for the SwiGLU Transition layer.

Exposes :func:`transition` — the full layer op (``LN_in → SwiGLU(expand_a, expand_b)
→ squeeze``), weights-as-args and autograd-transparent, with the fused SwiGLU kernel
inside. A model layer holds the weights as ``nn.Parameter`` and makes one call.
``n`` is part of the weight shapes (``expand_*_weight`` is ``(n*d, d)``); it is kept in the
signature for callers and checked against them.

Weight convention mirrors ``nn.Linear`` (``weight`` is ``(out, in)``).
"""

from __future__ import annotations

import torch


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
    """Fused SwiGLU transition — whole-op call. Returns ``x + transition(x)``, ``x``'s shape.

    The residual is part of the op, not a flag on it: ``modules.Transition``, this facade and
    the fused kernel's own epilogue all define Transition as ``x + squeeze(SwiGLU(LN(x)))``, and
    on the b2b paths the add is free (D == K, so the kernel reloads its own input tile). The
    bare ``squeeze(SwiGLU(x @ Wa^T, x @ Wb^T))`` with no LayerNorm and no residual is a
    different op -- ``kernels.triton_swiglu_ffn``.

    Autograd-transparent: back-prop produces gradients for ``x`` and every weight.

    Dispatch is the one ``modules.Transition`` uses: the hand-CUDA sm_90 kernels
    (``fused_sm90a`` at d=128, ``fused_wide_sm90a`` at d=64/256/384/512; n=4, bf16) where they
    apply, else the shape-general Triton residual path. LayerNorm is folded into the kernel on
    every path, never a separate native ``F.layer_norm``.
    """
    from miniworld_engine import settings

    if expand_a_weight.shape[0] != n * x.shape[-1]:
        msg = (f"transition: expand_a_weight has {expand_a_weight.shape[0]} rows, expected "
               f"n * d_hidden = {n} * {x.shape[-1]}")
        raise ValueError(msg)
    if settings.current().transition_fused_sm90a and settings.current().engine_backend != "triton":
        from miniworld_engine.kernels.transition.cuda import (
            fused_sm90a,
            fused_wide_sm90a,
        )

        if fused_sm90a.available(x, expand_a_weight, squeeze_weight):
            return fused_sm90a.transition_fused_sm90a(
                x, ln_in_weight, ln_in_bias, expand_a_weight, expand_b_weight, squeeze_weight, eps)
        if fused_wide_sm90a.available(x, expand_a_weight, squeeze_weight):
            return fused_wide_sm90a.transition_wide_sm90a(
                x, ln_in_weight, ln_in_bias, expand_a_weight, expand_b_weight, squeeze_weight, eps)

    from miniworld_engine.kernels.transition.triton.residual import transition_residual

    return transition_residual(
        x, ln_in_weight, ln_in_bias, expand_a_weight, expand_b_weight, squeeze_weight, eps,
    )

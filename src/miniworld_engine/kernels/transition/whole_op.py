"""Whole-op wrapper for the SwiGLU Transition layer.

Exposes :func:`transition` — the full layer op (``LN_in → SwiGLU(expand_a, expand_b)
→ squeeze``), weights-as-args and autograd-transparent, with the fused SwiGLU kernel
inside. A model layer holds the weights as ``nn.Parameter`` and makes one call.

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
    """Fused SwiGLU transition — whole-op call. Returns the same shape as ``x``.

    Autograd-transparent: back-prop produces gradients for ``x`` and every weight.

    LayerNorm is FOLDED INTO the fused kernel on every path (matching
    ``modules.Transition``'s arch dispatch) — never a separate native ``F.layer_norm``
    (which autocast runs in fp32 and whose GammaBeta backward is pathologically slow).
    The wide-``d`` split fallback still normalises via the fused Triton LN, not aten.
    """
    from miniworld_engine import kernels
    from miniworld_engine.modules import dispatch as _dispatch

    d_hidden = x.shape[-1]

    # H100 (sm_90) wide d: quack cute WGMMA fused expand — LN folded into the cute prologue.
    if d_hidden >= 256 and _dispatch.is_sm90(x.device):
        return kernels.cute_transition_fused(
            x, ln_in_weight, ln_in_bias,
            expand_a_weight, expand_b_weight, squeeze_weight, n, eps,
        )

    # Pre-Hopper (sm_80 / A100) wide d: the shape-general split GEMM wins, but keep the
    # input LayerNorm on our fused Triton LN (never native fp32 F.layer_norm).
    if d_hidden >= 256 and not _dispatch.is_sm90plus(x.device):
        from ..layernorm.triton.main import triton_layernorm
        from .triton.main import triton_transition

        x_n = triton_layernorm(
            x.reshape(-1, d_hidden), ln_in_weight, ln_in_bias, eps,
        ).reshape(x.shape)
        return triton_transition(
            x_n, expand_a_weight, expand_b_weight, squeeze_weight, n,
        )

    # B200 (sm_100, every d, via cute b2b_fwd_sm100) + d<=128 on any arch (the AF3 shape):
    # the fused Triton entry (LN folded, backward recomputes xn from saved LN stats).
    return kernels.triton_transition_fused(
        x, ln_in_weight, ln_in_bias,
        expand_a_weight, expand_b_weight, squeeze_weight, n, eps,
        save_xn=False,
    )

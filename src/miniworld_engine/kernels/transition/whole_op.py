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

    # Pre-Hopper wide d: the shape-general split GEMM wins, but keep the input LayerNorm on our
    # fused Triton LN (never native fp32 F.layer_norm).
    #
    # The `>= 256` and the `is_sm86` beside it are NOT free-standing numbers: they mirror
    # `modules/transition/module.py`, which is where the measurements are. A100 (sm_80),
    # cudagraph-manual: the fused path costs 2.28 ms at d=256 against the split's 1.40, and
    # 10.9 vs 4.8 at d=512, while d=128 (the AF3 shape) is where fused wins. sm_86
    # (RTX A5000/A6000) is different enough to need its own answer -- there the fused path
    # loses at EVERY d, 0.88-0.95x across both the L and d sweeps -- so on sm_86 all widths
    # take the split. This wrapper had only the `>= 256` half and would send d=128 to the
    # fused path on an A5000, against that measurement.
    if not _dispatch.is_sm90plus(x.device) and (d_hidden >= 256 or _dispatch.is_sm86(x.device)):
        from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
        from miniworld_engine.kernels.transition.triton.main import (
            triton_transition,
        )

        # x un-flattened. `triton_layernorm` flattens internally and reads the shape it was
        # GIVEN to build its autotune key (`rows_of` refuses an already-flat shape, by design --
        # it cannot tell a token (B, L, D) from a pair (B, L, L, D) once they are 2-D). Handing
        # it `x.reshape(-1, d_hidden)` raised ValueError on every call, so this branch had never
        # run: on a non-Hopper card `ops.transition` was dead at every d_hidden it selects.
        x_n = triton_layernorm(x, ln_in_weight, ln_in_bias, eps)
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

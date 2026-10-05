"""``ops.layer_norm_linear``: the fused ``Linear(LayerNorm(x))`` to a few outputs, one entry for every backend.

On an A100 (sm_80) bf16 activations run the hand-CUDA kernels (``cuda/sm80.py``; inference and training, ``MINIWORLD_LNLINEAR_SM80=0`` turns them off); everything else -- other
GPUs, fp32 activations, widths or output counts the kernels do not instantiate, an engine backend forced to Triton -- runs the Triton op (``triton/pair_bias.py``) unchanged.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.layernorm_linear.triton.pair_bias import (
    triton_layer_norm_linear,
)


def layer_norm_linear(x: torch.Tensor, ln_weight: torch.Tensor, proj_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """``Linear(LayerNorm(x))`` over the last dim: ``x`` ``[..., d]``, ``ln_weight`` the LayerNorm scale ``[d]`` (no beta), ``proj_weight`` ``[n_head, d]`` (no bias);
    returns ``[..., n_head]``.  Autograd-transparent (``dx``, the scale's and the projection's gradients)."""
    from miniworld_engine.kernels.layernorm_linear.cuda import sm80

    if sm80.serves(x, ln_weight, proj_weight):
        return sm80.layer_norm_linear(x, ln_weight, proj_weight, eps)
    return triton_layer_norm_linear(x, ln_weight, proj_weight, eps)

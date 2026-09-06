"""The ATOM-track diffusion transformer block (ESMFold2 sliding-window attention).

    x = x + SWA3DRoPEAttention(AdaptiveLayerNorm(x, s), attention_params)
    x = x + ConditionedTransition(x, s)

A different algorithm from the token track's, which is why it is a different folder. The
attention here is WINDOWED (half_window, so cost is linear in the atom length, not quadratic),
positional information comes from 3D RoPE rather than a learned pair bias, and there is no pair
representation in the block at all. The token block is in ``modules/dit``.

The adaLN lives in THIS block, unlike the token one where both parts build their own:
``SWA3DRoPEAttention`` is the attention core and takes ``(x, attention_params)`` -- it knows
nothing about conditioning. Its own bench docstring says the modulate and the FFN "live in the
consumer's SWAAtomBlock, not here". This is that consumer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from miniworld_engine.modules.adaptive_layernorm import AdaptiveLayerNorm
from miniworld_engine.modules.conditioned_transition import ConditionedTransition
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.swa_atom_attention import SWA3DRoPEAttention


class SWADiTBlock(nn.Module):
    """Atom-track DiT block: adaLN modulate -> windowed 3D-RoPE attention -> transition.

    ``forward(x, cond, attention_params)`` -> ``x``'s shape, with ``x`` at the ATOM length.
    ``attention_params`` is the tuple ``SWA3DRoPEAttention`` takes,
    ``(cos, sin, seqused, cu_seqlens, max_seqlen, valid)``; build it with
    ``modules.swa_atom_attention.build_attention_params``.
    """

    def __init__(
        self,
        d_atom: int = 128,
        d_cond: int = 128,
        n_head: int = 4,
        n: int = 2,
        half_window: int = 64,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.ada_ln = AdaptiveLayerNorm(
            d_hidden=d_atom, d_cond=d_cond, implementation=implementation,
        )
        self.attention = SWA3DRoPEAttention(d_atom, n_head, half_window=half_window)
        self.transition = ConditionedTransition(
            d_hidden=d_atom, d_cond=d_cond, n=n, implementation=implementation,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "N S d_atom"],
        cond: Float[torch.Tensor, "N S d_cond"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d_atom"]:
        """Both residuals explicit, as on the token track."""
        x = x + self.attention(self.ada_ln(x, cond), attention_params)
        return x + self.transition(x, cond)

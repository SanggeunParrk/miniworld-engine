"""The full pair-bias diffusion transformer block (AF3 Alg. 23).

    a = AugmentedAttentionPairBias(a, s, z, mask)     # a + attention(a, s, z)
    a = ConditionedTransition(a, s)                  # a + transition(a, s)

Token and ordinary atom DiT share this algorithm, with different widths and head counts.
Attention covers the full sequence with a pair bias, and owns its AdaLN conditioning.
The separate ``modules/swa_dit`` block uses windowed 3D-RoPE attention without a pair term.

WHY A BLOCK IS BENCHED AT ALL, when its parts already are: a per-part result does not compose.
Every kernel here is an opaque ``custom_op``, so each launch carries CPU overhead that a per-part
bench pays once and a block pays once per part -- and, the other way, `torch.compile` fuses
ACROSS parts in the reference but cannot fuse across our opaque ops. Measured on adaLN alone:
0.74x without CUDA graphs, 1.09x with them, same kernels. A block is where both effects land.

Each part owns its residual, like every ``x = x + f(x)`` module in the engine:
``AugmentedAttentionPairBias`` and ``ConditionedTransition`` return ``x + f(x)``, so the block
only chains them. The fused block paths (``integrations.token_dit`` / ``token_dit_train`` at token
widths, ``integrations.atom_dit`` at atom widths on B200) fold both residuals into their kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from miniworld_engine.integrations import atom_dit as _atom
from miniworld_engine.integrations import token_dit as _h100
from miniworld_engine.integrations import token_dit_train as _train
from miniworld_engine.modules.augmented_attention import AugmentedAttentionPairBias
from miniworld_engine.modules.conditioned_transition import ConditionedTransition
from miniworld_engine.modules.exceptions import ImplementationType


def _shared_key_mask(mask: torch.Tensor | None) -> tuple[torch.Tensor | None, bool]:
    """(the [B, L] key mask the samples share, True), or (``mask``, False) when an [A, B, L] mask may differ per sample.

    A diffusion module hands the token DiT its token mask expanded over the augmented samples
    (``token_mask.unsqueeze(0).expand(A, -1, -1)``): the same [B, L] row for every sample, recognised by its stride 0 (no
    device read, so it holds under torch.compile and CUDA-graph capture). A materialized [A, B, L] mask is not inspected."""
    if mask is None or mask.ndim == 2:
        return mask, True
    if mask.ndim == 3 and (mask.shape[0] == 1 or mask.stride(0) == 0):
        return mask[0], True
    return mask, False


class DiTBlock(nn.Module):
    """Token or ordinary atom DiT: pair-bias attention, then conditioned transition.

    ``forward(single, cond, pair, mask)`` -> ``single``'s shape. ``single`` and ``cond`` carry the
    augmentation axis (``A, B, L, d``); ``pair`` does not (``B, L, L, d_pair``), because the pair
    bias is shared across augmentations.

    The adaLN conditioning is not applied here: both parts build their own ``AdaptiveLayerNorm``
    from their ``(d_hidden, d_cond)``.
    """

    def __init__(
        self,
        d_single: int = 768,
        d_cond: int = 384,
        d_pair: int = 128,
        n_head: int = 16,
        n: int = 2,
        *,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.attention = AugmentedAttentionPairBias(
            d_single=d_single, d_cond=d_cond, d_pair=d_pair, n_head=n_head,
            use_qk_norm=use_qk_norm, implementation=implementation,
        )
        self.transition = ConditionedTransition(
            d_hidden=d_single, d_cond=d_cond, n=n, implementation=implementation,
        )

    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | Bool[torch.Tensor, "A B L"] | None = None,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """AF3 Alg. 23; each part returns its residual output (stream in, stream out).

        ``mask`` is a key mask: [B, L], or [A, B, L] (the attention's own contract). The fused token DiT paths take one key
        mask shared by the samples, so an [A, B, L] mask that is one -- a [B, L] mask expanded over A (stride 0), or A == 1 --
        reaches them as its [B, L] row; a mask that differs per sample keeps the module path."""
        key_mask, shared = _shared_key_mask(mask)
        if shared and _h100.serves(self, single, cond, pair, compute_dtype):
            return _h100.update(self, single, cond, pair, key_mask)
        if shared and _train.serves(self, single, cond, pair, key_mask, compute_dtype):
            return _train.block(self, single, cond, pair, key_mask, compute_dtype)
        if _atom.serves(self, single, cond, pair, mask, compute_dtype):
            return _atom.block(self, single, cond, pair, mask)
        kw = {"compute_dtype": compute_dtype} if compute_dtype is not None else {}
        single = self.attention(single, cond, pair, mask, **kw)
        return self.transition(single, cond)

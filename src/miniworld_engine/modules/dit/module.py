"""The TOKEN-track diffusion transformer block (AF3 Alg. 23).

    a = a + AugmentedAttentionPairBias(a, s, z, mask)
    a = a + ConditionedTransition(a, s)

A different algorithm from the atom track's, which is why it is a different folder: this one
attends over the whole token sequence with a PAIR BIAS (the trunk's z feeding the logits), and
its attention owns the adaLN conditioning internally. The atom track is windowed, 3D-RoPE, and
has no pair term at all -- see ``modules/swa_dit``.

WHY A BLOCK IS BENCHED AT ALL, when its parts already are: a per-part result does not compose.
Every kernel here is an opaque ``custom_op``, so each launch carries CPU overhead that a per-part
bench pays once and a block pays once per part -- and, the other way, `torch.compile` fuses
ACROSS parts in the reference but cannot fuse across our opaque ops. Measured on adaLN alone:
0.74x without CUDA graphs, 1.09x with them, same kernels. A block is where both effects land.

The residual is EXPLICIT here, and it belongs here: neither ``AugmentedAttentionPairBias`` nor
``ConditionedTransition`` adds one -- they return the update and the block owns the stream.
(Contrast the pairformer's trimul and transition, which fuse their residual into the kernel
epilogue because there it is one operand of a store the kernel is already making.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from miniworld_engine.modules.augmented_attention import AugmentedAttentionPairBias
from miniworld_engine.modules.conditioned_transition import ConditionedTransition
from miniworld_engine.modules.exceptions import ImplementationType


class DiTBlock(nn.Module):
    """Token-track DiT block: augmented attention with pair bias, then conditioned transition.

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
        mask: Bool[torch.Tensor, "B L"] | None = None,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """AF3 Alg. 23, both residuals explicit (the parts return updates, not streams)."""
        kw = {"compute_dtype": compute_dtype} if compute_dtype is not None else {}
        single = single + self.attention(single, cond, pair, mask, **kw)
        return single + self.transition(single, cond)

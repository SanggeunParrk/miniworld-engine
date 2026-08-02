# vendored from team-gm psk/benchmark : team_gm/modules/pairformer.py
"""Pairformer stack — a thin *shell* over the pair-track fused-kernel modules.

The Pairformer/PairformerBlock here do **no** backend dispatch of their own. The
``implementation`` selector is passed straight through to each sub-module
(:class:`TriangleMultiplication`, :class:`TriangleAttention`, :class:`Transition`),
and *those* resolve it to the best concrete kernel for the running GPU:

    * ``pytorch``        -> reference torch ops
    * ``cuequivariance`` -> cuequivariance kernels where a vendor kernel exists
                            (trimul, triangle-attention); modules without one
                            (transition) fall back to their own auto path
    * ``miniworld``      -> ours: each module auto-routes to its fastest internal
                            impl (triton / cute / hand-CUDA) per shape and arch

This mirrors the team-gm Pairformer but keeps only the **pair track** (the part
with our kernels). The single track (OuterProduct / AttentionPairBias / single
Transition) is intentionally omitted — it always ran pytorch upstream and is not
part of the kernel comparison. Enable-single support can be layered on later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.primitives import Dropout
from miniworld_engine.modules.transition import Transition
from miniworld_engine.modules.triangle_attention import TriangleAttention
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
    TriangleMultiplication,
)


@dataclass
class PairformerConfig:
    """Pair-track Pairformer hyper-parameters (AF3 defaults)."""

    d_pair: int = 128
    d_hidden_tri_multi: int = 128
    d_hidden_tri_attention: int = 32
    n_head_tri_attention: int = 4
    p_drop: float = 0.25
    use_self_attention: bool = True
    n_block: int = 4
    # When False, the block drops both triangle-attention updates and keeps only
    # the triangle-multiplication update + transition — the "bidir trimul only" variant.
    use_triangle_attention: bool = True
    # When True, the two directional updates are replaced by a single fused
    # BidirectionalTriangleMultiplication (ours = the developed fused-bidir b200
    # kernel); used by the "trimul_only" variant.
    bidirectional_trimul: bool = False


class PairformerBlock(nn.Module):
    """A single Pairformer block over the pair representation.

    Wires two triangle-multiplication updates (outgoing/incoming), two
    triangle-attention updates (starting/ending) and a pair transition, exactly
    as team-gm's ``PairformerBlock``. ``implementation`` is forwarded verbatim to
    every sub-module — the block itself never branches on it.
    """

    def __init__(
        self,
        config: PairformerConfig,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.config = config

        if config.bidirectional_trimul:
            # Single fused bidirectional update (ours = the developed fused-bidir
            # b200 kernel; cuequiv = two directional calls; pytorch = fused ref).
            self.tri_multi = BidirectionalTriangleMultiplication(
                d_pair=config.d_pair,
                d_hidden=config.d_hidden_tri_multi,
                implementation=implementation,
                p_drop=config.p_drop,  # drop_row owned by the module (residual is unconditional)
            )
            self.tri_multi_outgoing = None
            self.tri_multi_incoming = None
        else:
            self.tri_multi = None
            self.tri_multi_outgoing = TriangleMultiplication(
                d_pair=config.d_pair,
                d_hidden=config.d_hidden_tri_multi,
                outgoing=True,
                implementation=implementation,
                p_drop=config.p_drop,  # drop_row owned by the module (residual is unconditional)
            )
            self.tri_multi_incoming = TriangleMultiplication(
                d_pair=config.d_pair,
                d_hidden=config.d_hidden_tri_multi,
                outgoing=False,
                implementation=implementation,
                p_drop=config.p_drop,  # drop_row owned by the module (residual is unconditional)
            )
        if config.use_triangle_attention:
            # team-gm's ``d_hidden_tri_attention`` is the PER-HEAD channel
            # (projections are ``d_hidden * n_head``); miniworld's TriangleAttention
            # takes the TOTAL hidden dim, so scale by n_head. Per-head must be >=16
            # for the triton flash kernel's tl.dot (K>=16); AF3's 32 satisfies this.
            d_hidden_attn_total = (
                config.d_hidden_tri_attention * config.n_head_tri_attention
            )
            self.tri_atten_starting = TriangleAttention(
                d_pair=config.d_pair,
                n_head=config.n_head_tri_attention,
                d_hidden=d_hidden_attn_total,
                starting=True,
                use_self_attention=config.use_self_attention,
                implementation=implementation,
                p_drop=config.p_drop,  # drop_row owned by the module (residual unconditional)
            )
            self.tri_atten_ending = TriangleAttention(
                d_pair=config.d_pair,
                n_head=config.n_head_tri_attention,
                d_hidden=d_hidden_attn_total,
                starting=False,
                use_self_attention=config.use_self_attention,
                implementation=implementation,
                p_drop=config.p_drop,  # drop_col owned by the module (residual unconditional)
            )
        else:
            self.tri_atten_starting = None
            self.tri_atten_ending = None
        self.drop_row = Dropout(broadcast_dim=1, p_drop=config.p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=config.p_drop)
        self.transition_pair = Transition(
            config.d_pair, implementation=implementation
        )

    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass (residual updates, pair track only)."""
        # trimul & transition OWN their residual (unconditional) + drop_row (module p_drop, fused
        # into the kernel epilogue for speed) — the block just calls ``module(pair, mask)``; there
        # is no external ``pair + drop_row(...)`` here. See the modules' constructor comments.
        if self.tri_multi is not None:
            pair = self.tri_multi(pair, mask)  # y = pair + drop_row(bidir_trimul(pair))
        else:
            pair = self.tri_multi_outgoing(pair, mask)  # y = pair + drop_row(trimul_out(pair))
            pair = self.tri_multi_incoming(pair, mask)  # y = pair + drop_row(trimul_in(pair))
        if self.tri_atten_starting is not None:
            # Triangle attention now OWNS its residual + dropout too (drop_row for starting,
            # drop_col for ending) — same contract as trimul. It is not kernel-fused yet (explicit
            # add inside the module), but the block just calls ``module(pair, mask)``.
            pair = self.tri_atten_starting(pair, mask)  # y = pair + drop_row(attn_start(pair))
            pair = self.tri_atten_ending(pair, mask)    # y = pair + drop_col(attn_end(pair))
        pair = self.transition_pair(pair)  # y = pair + transition(pair) (residual fused, no dropout)
        return pair


class Pairformer(nn.Module):
    """The Pairformer stack: ``n_block`` :class:`PairformerBlock` in sequence.

    Pure shell — its only job is to hold the blocks and forward ``implementation``
    down at construction. All kernel dispatch happens inside the sub-modules.

    Parameters
    ----------
    config : PairformerConfig
        Stack hyper-parameters.
    implementation : ImplementationType | str
        Backend family passed to every sub-module (``pytorch`` / ``cuequivariance``
        / ``miniworld``). Coerced from a string for convenience.
    """

    def __init__(
        self,
        config: PairformerConfig | None = None,
        implementation: ImplementationType | str = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.config = config or PairformerConfig()
        self.implementation = ImplementationType(implementation)

        self.pairformer_blocks = nn.ModuleList(
            [
                PairformerBlock(self.config, implementation=self.implementation)
                for _ in range(self.config.n_block)
            ]
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Run the pair representation through every block."""
        for block in self.pairformer_blocks:
            pair = block(pair, mask)
        return pair

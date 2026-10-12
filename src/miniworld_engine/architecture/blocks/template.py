from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from miniworld_engine.architecture.blocks._engine_impl import to_engine_impl
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.architecture.primitives import LayerNorm, Linear
from miniworld_engine.architecture.typing import typecheck
from miniworld_engine.modules import (
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)

if TYPE_CHECKING:
    from jaxtyping import Bool, Float


class TemplatePairformerBlock(nn.Module):
    """Single Pairformer block for template processing (pair-only)."""

    def __init__(
        self,
        d_pair: int,
        d_hidden: int,
        n_head_tri_attention: int,
        p_drop: float,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        use_self_attention: bool = True,
    ) -> None:
        super().__init__()

        # residual + drop_row/drop_col owned by the miniworld_engine modules; dropout passed via
        # p_drop, applied in-module only during self.training (trimul & tri_atten_starting use
        # drop_row dim=1, tri_atten_ending uses drop_col dim=2). No external `pair + drop(...)`.
        self.tri_multi_outgoing = TriangleMultiplication(
            d_pair=d_hidden,
            outgoing=True,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop,
        )
        self.tri_multi_incoming = TriangleMultiplication(
            d_pair=d_hidden,
            outgoing=False,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop,
        )
        self.tri_atten_starting = TriangleAttention(
            d_pair=d_hidden,
            n_head=n_head_tri_attention,
            starting=True,
            implementation=to_engine_impl(implementation),
            use_self_attention=use_self_attention,
            p_drop=p_drop,
        )
        self.tri_atten_ending = TriangleAttention(
            d_pair=d_hidden,
            n_head=n_head_tri_attention,
            starting=False,
            implementation=to_engine_impl(implementation),
            use_self_attention=use_self_attention,
            p_drop=p_drop,
        )
        # Transition has no cuequivariance backend: implementation=CUEQUIVARIANCE
        # errors here (honest dispatch — no silent reroute to miniworld_engine).
        self.transition_pair = Transition(
            d_pair,
            n=2,
            implementation=to_engine_impl(implementation),
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass. residual + dropout are owned by the miniworld_engine modules (see __init__)."""
        pair = self.tri_multi_outgoing(pair, mask)
        pair = self.tri_multi_incoming(pair, mask)
        pair = self.tri_atten_starting(pair, mask)
        pair = self.tri_atten_ending(pair, mask)
        return self.transition_pair(pair)


class TemplatePairformer(nn.Module):
    """Stack of TemplatePairformerBlocks."""

    class Config(BaseModel):
        """Configuration for TemplatePairformer."""

        d_pair_template_input: int = 4
        d_pair: int = 128
        d_hidden: int = 128
        n_head_tri_attention: int = 4
        p_drop: float = 0.25
        use_self_attention: bool = True
        implementation: ImplementationType = ImplementationType.PYTORCH
        n_block: int = 2
        dropout_prob: float = 0.25

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TemplatePairformerBlock(
                    d_pair=config.d_pair,
                    d_hidden=config.d_hidden,
                    n_head_tri_attention=config.n_head_tri_attention,
                    p_drop=config.p_drop,
                    implementation=config.implementation,
                    use_self_attention=config.use_self_attention,
                )
                for _ in range(config.n_block)
            ],
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        for block in self.blocks:
            pair = block(pair, mask)
        return pair


class TemplateEmbedder(nn.Module):
    """Template embedder (Algorithm 16)."""

    def __init__(
        self,
        d_pair: int,
        d_pair_template: int,
        template_pairformer_config: TemplatePairformer.Config,
    ) -> None:
        super().__init__()
        self.d_pair = d_pair
        self.d_pair_template = d_pair_template

        self.ln_pair = LayerNorm(d_pair)
        self.proj_pair = Linear(d_pair, d_pair_template, bias=False)

        self.proj_template = Linear(
            template_pairformer_config.d_pair_template_input,
            d_pair_template,
            bias=False,
        )

        self.template_pairformer = TemplatePairformer(template_pairformer_config)
        self.ln_template = LayerNorm(d_pair_template)

        self.proj_out = Linear(d_pair_template, d_pair, bias=False)

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        template_feat: Float[torch.Tensor, "B L L d_pair_template_input"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        pair = self.proj_pair(self.ln_pair(pair)) + self.proj_template(template_feat)

        pair = self.template_pairformer(pair, mask=mask)
        pair = self.ln_template(pair)

        return self.proj_out(F.relu(pair))

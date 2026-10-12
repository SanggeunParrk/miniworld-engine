import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, model_validator
from torch import nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from typing_extensions import Self

from miniworld_engine.architecture.blocks._engine_impl import to_engine_impl
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.architecture.primitives import Linear
from miniworld_engine.architecture.typing import typecheck
from miniworld_engine.modules import (
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)


class MSAPairformerBlock(nn.Module):
    """A single MSA-Pairformer block."""

    def __init__(
        self,
        d_msa: int = 64,
        d_pair: int = 128,
        d_hidden: int = 32,
        n_head_msa: int = 8,
        n_head_tri_attention: int = 4,
        p_drop_msa: float = 0.15,
        p_drop_pair: float = 0.25,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        is_last_block: bool = False,
    ) -> None:
        super().__init__()
        self.is_last_block = is_last_block

        # Communication
        self.outer_product_mean = OuterProductMean(
            d_msa=d_msa,
            d_pair=d_pair,
            d_hidden=d_hidden,
        )

        # MSA stack (skipped in last block)
        if not is_last_block:
            self.msa_pair_weighted_avg = MSAPairWeightedAveraging(
                d_msa=d_msa,
                d_pair=d_pair,
                n_head=n_head_msa,
                p_drop=p_drop_msa,  # residual + drop_msa now owned by the layer
            )
            self.transition_msa = Transition(d_msa)

        # Pair stack. residual + drop_row/drop_col are OWNED by the miniworld_engine modules now
        # (unconditional residual, fused into the kernel epilogue for trimul); dropout is passed via
        # p_drop and applied in-module only during self.training (trimul & tri_atten_starting use
        # drop_row dim=1, tri_atten_ending uses drop_col dim=2). No external `pair + drop(...)`.
        self.tri_multi_outgoing = TriangleMultiplication(
            d_pair=d_pair,
            outgoing=True,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop_pair,
        )
        self.tri_multi_incoming = TriangleMultiplication(
            d_pair=d_pair,
            outgoing=False,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop_pair,
        )
        self.tri_atten_starting = TriangleAttention(
            d_pair=d_pair,
            n_head=n_head_tri_attention,
            starting=True,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop_pair,
        )
        self.tri_atten_ending = TriangleAttention(
            d_pair=d_pair,
            n_head=n_head_tri_attention,
            starting=False,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop_pair,
        )
        # Transition has no cuequivariance backend: implementation=CUEQUIVARIANCE
        # errors here (honest dispatch — no silent reroute to miniworld_engine).
        self.transition_pair = Transition(
            d_pair,
            implementation=to_engine_impl(implementation),
        )

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B M L d_msa"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
        msa_mask: Bool[torch.Tensor, "B M L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B M L d_msa"],
        Float[torch.Tensor, "B L L d_pair"],
    ]:
        """Forward pass."""
        # Communication
        pair = self.outer_product_mean(msa, msa_mask, residual=pair)

        # MSA stack
        if not self.is_last_block:
            msa = self.msa_pair_weighted_avg(
                msa, pair, mask
            )  # owns residual + drop_msa
            msa = self.transition_msa(msa)

        # Pair stack
        pair = self.tri_multi_outgoing(pair, mask)
        pair = self.tri_multi_incoming(pair, mask)
        pair = self.tri_atten_starting(pair, mask)
        pair = self.tri_atten_ending(pair, mask)
        pair = self.transition_pair(pair)

        return msa, pair


class MSAPairformer(nn.Module):
    """MSA-Pairformer stack consisting of multiple MSA-Pairformer blocks."""

    class Config(BaseModel):
        """Configuration for MSAPairformer module.

        Parameters
        ----------
        d_msa : int
            Dimension of MSA representation.
        d_pair : int
            Dimension of pair representation.
        d_hidden : int
            Hidden dimension for OuterProductMean.
        n_head_msa : int
            Number of heads for MSAPairWeightedAveraging.
        n_head_tri_attention : int
            Number of heads for TriangleAttention.
        p_drop_msa : float
            Dropout probability for MSA stack.
        p_drop_pair : float
            Dropout probability for pair stack.
        implementation : ImplementationType
            Implementation to use for miniworld_engine-supported layers.
        n_block : int
            Number of blocks.
        n_checkpoint_segments : int, optional
            Number of segments for gradient checkpointing.

        """

        d_msa: int = 64
        d_pair: int = 128
        d_hidden: int = 32
        n_head_msa: int = 8
        n_head_tri_attention: int = 4
        p_drop_msa: float = 0.15
        p_drop_pair: float = 0.25
        implementation: ImplementationType = ImplementationType.PYTORCH
        n_block: int = 4
        n_checkpoint_segments: int | None = None
        use_checkpoint: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList(
            [
                MSAPairformerBlock(
                    d_msa=config.d_msa,
                    d_pair=config.d_pair,
                    d_hidden=config.d_hidden,
                    n_head_msa=config.n_head_msa,
                    n_head_tri_attention=config.n_head_tri_attention,
                    p_drop_msa=config.p_drop_msa,
                    p_drop_pair=config.p_drop_pair,
                    implementation=config.implementation,
                    is_last_block=(i + 1 == config.n_block),
                )
                for i in range(config.n_block)
            ],
        )

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B M L d_msa"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
        msa_mask: Bool[torch.Tensor, "B M L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        if self.config.use_checkpoint and torch.is_grad_enabled():
            for block in self.blocks:
                msa, pair = checkpoint(
                    block, msa, pair, mask, msa_mask, use_reentrant=False
                )
        elif self.config.n_checkpoint_segments is None or not torch.is_grad_enabled():
            for block in self.blocks:
                msa, pair = block(msa, pair, mask, msa_mask)
        else:

            def run_module(module):
                def forward(inputs):
                    m, p = inputs
                    return module(m, p, mask, msa_mask)

                return forward

            msa, pair = checkpoint_sequential(
                [run_module(b) for b in self.blocks],
                segments=self.config.n_checkpoint_segments,
                use_reentrant=False,
                input=(msa, pair),
            )

        return pair


class MSAModuleBlock(nn.Module):
    """MSA update and pair update module block."""

    def __init__(
        self,
        d_msa: int = 64,
        d_pair: int = 128,
        d_hidden_msa: int = 32,
        n_head_tri_attention: int = 4,
        p_drop_msa: float = 0.15,
        p_drop: float = 0.25,
        *,
        use_self_attention: bool = True,
        use_triangle_attention: bool = True,
        mask_interchain: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        last_block: bool = False,
    ) -> None:
        super().__init__()

        self.last_block = last_block
        self.use_triangle_attention = use_triangle_attention

        self.outer_product_mean = OuterProductMean(
            d_msa=d_msa,
            d_pair=d_pair,
            d_hidden=d_hidden_msa,
            mask_interchain=mask_interchain,
        )
        if not last_block:
            self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
                d_msa=d_msa,
                d_pair=d_pair,
                d_hidden=d_hidden_msa,
                p_drop=p_drop_msa,  # residual + drop_msa now owned by the layer
            )
            self.transition_msa = Transition(
                d_msa,
                implementation=to_engine_impl(implementation),
            )

        # residual + drop_row/drop_col owned by the miniworld_engine modules; dropout passed via
        # p_drop, applied in-module only during self.training. No external `pair + drop(...)`.
        self.tri_multi_outgoing = TriangleMultiplication(
            d_pair=d_pair,
            outgoing=True,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop,
        )
        self.tri_multi_incoming = TriangleMultiplication(
            d_pair=d_pair,
            outgoing=False,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop,
        )

        if use_triangle_attention:
            self.tri_atten_starting = TriangleAttention(
                d_pair=d_pair,
                n_head=n_head_tri_attention,
                starting=True,
                use_self_attention=use_self_attention,
                implementation=to_engine_impl(implementation),
                p_drop=p_drop,
            )
            self.tri_atten_ending = TriangleAttention(
                d_pair=d_pair,
                n_head=n_head_tri_attention,
                starting=False,
                use_self_attention=use_self_attention,
                implementation=to_engine_impl(implementation),
                p_drop=p_drop,
            )
        self.transition_pair = Transition(
            d_pair,
            implementation=to_engine_impl(implementation),
        )

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B N L d_msa"],
        msa_mask: Bool[torch.Tensor, "B N"] | None,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
        token_asym_id: Int[torch.Tensor, "B L"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B N L d_msa"], Float[torch.Tensor, "B L L d_pair"]]:
        """Forward pass."""
        opm_mask = None
        if msa_mask is not None:
            opm_mask = msa_mask[:, :, None].expand(-1, -1, msa.shape[2])
            if mask is not None:
                opm_mask = opm_mask & mask[:, None, :]

        pair = self.outer_product_mean(msa, opm_mask, token_asym_id, residual=pair)

        if not self.last_block:
            msa = self.msa_pair_weighted_averaging(
                msa, pair, mask
            )  # owns residual + drop_msa
            msa = self.transition_msa(msa)

        pair = self.tri_multi_outgoing(pair, mask)
        pair = self.tri_multi_incoming(pair, mask)
        if self.use_triangle_attention:
            pair = self.tri_atten_starting(pair, mask)
            pair = self.tri_atten_ending(pair, mask)
        pair = self.transition_pair(pair)

        return msa, pair


class MSAModule(nn.Module):
    """MSA module with multiple blocks."""

    class Config(BaseModel):
        """Configuration for MSAModule."""

        d_msa: int = 64
        d_pair: int = 128
        d_single_token_input: int = 441
        d_hidden_msa: int = 32
        d_hidden_tri_multi: int = 128
        d_hidden_tri_attention: int = 32
        n_head_tri_attention: int = 4
        n_head_attention: int = 16
        p_drop_msa: float = 0.15
        p_drop: float = 0.25
        use_self_attention: bool = True
        use_triangle_attention: bool = True
        mask_interchain: bool = False
        implementation: ImplementationType = ImplementationType.PYTORCH
        num_res_class: int = 32

        n_block: int = 4
        n_checkpoint_segments: int | None = None

        @model_validator(mode="after")
        def check_checkpoint_segments(self) -> Self:
            """Check n_checkpoint_segments is valid."""
            if self.n_checkpoint_segments is None:
                return self
            if (
                self.n_checkpoint_segments > self.n_block
                or self.n_checkpoint_segments < 1
            ):
                msg = (
                    "n_checkpoint_segments must be between 1 and n_block. "
                    f"Got n_checkpoint_segments={self.n_checkpoint_segments} "
                    f"and n_block={self.n_block}."
                )
                raise ValueError(msg)
            return self

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_res_class = config.num_res_class

        self.embed_msa = Linear(self.num_res_class + 2, config.d_msa, bias=False)
        self.single_to_msa = Linear(
            config.d_single_token_input,
            config.d_msa,
            bias=False,
        )

        self.blocks = nn.ModuleList(
            [
                MSAModuleBlock(
                    d_msa=config.d_msa,
                    d_pair=config.d_pair,
                    d_hidden_msa=config.d_hidden_msa,
                    n_head_tri_attention=config.n_head_tri_attention,
                    p_drop_msa=config.p_drop_msa,
                    p_drop=config.p_drop,
                    use_self_attention=config.use_self_attention,
                    use_triangle_attention=config.use_triangle_attention,
                    mask_interchain=config.mask_interchain,
                    implementation=config.implementation,
                )
                for _ in range(config.n_block - 1)
            ],
        )
        self.blocks.append(
            MSAModuleBlock(
                d_msa=config.d_msa,
                d_pair=config.d_pair,
                d_hidden_msa=config.d_hidden_msa,
                n_head_tri_attention=config.n_head_tri_attention,
                p_drop_msa=config.p_drop_msa,
                p_drop=config.p_drop,
                use_self_attention=config.use_self_attention,
                use_triangle_attention=config.use_triangle_attention,
                mask_interchain=config.mask_interchain,
                implementation=config.implementation,
                last_block=True,
            )
        )

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B N L d_msa"],
        msa_mask: Bool[torch.Tensor, "B N"] | None,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
        token_asym_id: Int[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        msa = msa.to(pair.dtype)
        msa = self.embed_msa(msa)
        if single is not None:
            msa = msa + self.single_to_msa(single).unsqueeze(1)

        for block in self.blocks:
            msa, pair = block(msa, msa_mask, pair, mask, token_asym_id)
        return pair

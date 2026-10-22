import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel, model_validator
from torch import nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from typing_extensions import Self

from miniworld_engine.architecture.blocks._engine_impl import to_engine_impl
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.architecture.typing import typecheck
from miniworld_engine.modules import (
    AttentionPairBias,
    BidirectionalTriangleMultiplication,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)


class PairformerBlock(nn.Module):
    """A single Pairformer block."""

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 128,
        n_head_tri_attention: int = 4,
        n_head_attention: int = 16,
        p_drop: float = 0.25,
        *,
        use_single: bool = True,
        use_self_attention: bool = True,
        use_triangle_attention: bool = True,
        use_bidirectional_trimul: bool = False,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.use_single = use_single
        self.use_triangle_attention = use_triangle_attention
        self.use_bidirectional_trimul = use_bidirectional_trimul

        if use_bidirectional_trimul:
            # Single block computing outgoing+incoming together. bidir has no
            # cuequivariance backend, so implementation=CUEQUIVARIANCE will error
            # here (honest dispatch — no silent reroute to miniworld_engine).
            self.tri_multi = BidirectionalTriangleMultiplication(
                d_pair=d_pair,
                implementation=to_engine_impl(implementation),
                p_drop=p_drop,
            )
        else:
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
                use_qk_norm=use_qk_norm,
                implementation=to_engine_impl(implementation),
                p_drop=p_drop,
            )
            self.tri_atten_ending = TriangleAttention(
                d_pair=d_pair,
                n_head=n_head_tri_attention,
                starting=False,
                use_self_attention=use_self_attention,
                use_qk_norm=use_qk_norm,
                implementation=to_engine_impl(implementation),
                p_drop=p_drop,
            )
        # residual + drop_row/drop_col are OWNED by the miniworld_engine modules now (unconditional
        # residual fused into the kernel epilogue for trimul; explicit residual for triangle
        # attention). Dropout is passed via p_drop and applied in-module only during self.training —
        # trimul & tri_atten_starting use drop_row (dim=1), tri_atten_ending uses drop_col (dim=2).
        # The block just calls module(pair, mask); no external `pair + drop(...)` wrapper (that would
        # un-fuse the residual and lose the speed win).
        # Transition has no cuequivariance backend: implementation=CUEQUIVARIANCE
        # errors here (honest dispatch — no silent reroute to miniworld_engine).
        self.transition_pair = Transition(
            d_pair,
            implementation=to_engine_impl(implementation),
        )

        if use_single:
            self.attention_pair_bias = AttentionPairBias(
                d_single=d_single,
                d_pair=d_pair,
                n_head=n_head_attention,
                use_qk_norm=use_qk_norm,
            )
            self.transition_single = Transition(
                d_single, implementation=to_engine_impl(implementation)
            )

    @classmethod
    def from_components(
        cls,
        *,
        triangle_outgoing: nn.Module,
        triangle_incoming: nn.Module,
        attention_starting: nn.Module | None,
        attention_ending: nn.Module | None,
        transition_pair: nn.Module,
        attention_single: nn.Module | None = None,
        transition_single: nn.Module | None = None,
        parallel: bool = False,
    ) -> "PairformerBlock":
        """Compose checkpoint-compatible ops with the standard residual contract.

        Model adapters may provide different parameter layouts or mask semantics.
        Every supplied self-residual op returns its complete residual output;
        this shared block owns only ordering. All tensors retain a batch axis.

        ``parallel`` (Chai-1, RoseTTAFold3): every part reads the block input, ``pair + sum_i f_i(pair)`` (the single attention reads
        the pair entering the block, the single transition the single entering it), summed from the parts' residual outputs by
        ``kernels.dit_variants.cuda.parallel_residual``.
        """
        if (attention_single is None) != (transition_single is None):
            raise ValueError(
                "Single attention and transition must be supplied together"
            )
        if (attention_starting is None) != (attention_ending is None):
            raise ValueError("Both pair attentions are present, or neither is")
        block = cls.__new__(cls)
        nn.Module.__init__(block)
        block.use_single = attention_single is not None
        # A family that folds from a language model keeps only the triangle
        # multiplications and the transition; building the attentions it has no
        # weights for would leave them at random init.
        block.use_triangle_attention = attention_starting is not None
        block.use_bidirectional_trimul = False
        block.tri_multi_outgoing = triangle_outgoing
        block.tri_multi_incoming = triangle_incoming
        block.tri_atten_starting = attention_starting
        block.tri_atten_ending = attention_ending
        block.transition_pair = transition_pair
        block.parallel = parallel
        if block.use_single:
            block.attention_pair_bias = attention_single
            block.transition_single = transition_single
        return block

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
        *,
        pair_mask: Bool[torch.Tensor, "B L L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B L L d_pair"],
        Float[torch.Tensor, "B L d_single"] | None,
    ]:
        """Forward pass."""
        triangle_mask = pair_mask if pair_mask is not None else mask
        if getattr(self, "parallel", False):
            return self._parallel_forward(pair, single, mask, triangle_mask)
        if self.use_bidirectional_trimul:
            pair = self.tri_multi(pair, triangle_mask)
        else:
            pair = self.tri_multi_outgoing(pair, triangle_mask)
            pair = self.tri_multi_incoming(pair, triangle_mask)
        if self.use_triangle_attention:
            pair = self.tri_atten_starting(pair, triangle_mask)
            pair = self.tri_atten_ending(pair, triangle_mask)
        pair = self.transition_pair(pair)

        if self.use_single:
            if single is None:
                msg = "single must be provided when use_single=True"
                raise ValueError(msg)
            single = self.attention_pair_bias(single, pair, mask)  # owns residual
            single = self.transition_single(single)
        return pair, single

    def _parallel_forward(self, pair, single, mask, triangle_mask):
        """Every part on the block input; the parts own their residuals, so the update is ``pair + sum_i (y_i - pair)``."""
        if self.use_bidirectional_trimul:
            outs = [self.tri_multi(pair, triangle_mask)]
        else:
            outs = [self.tri_multi_outgoing(pair, triangle_mask), self.tri_multi_incoming(pair, triangle_mask)]
        if self.use_triangle_attention:
            outs += [self.tri_atten_starting(pair, triangle_mask), self.tri_atten_ending(pair, triangle_mask)]
        outs.append(self.transition_pair(pair))
        new_pair = _parallel_residual(pair, outs)
        if self.use_single:
            if single is None:
                msg = "single must be provided when use_single=True"
                raise ValueError(msg)
            single = _parallel_residual(single, [self.attention_pair_bias(single, pair, mask), self.transition_single(single)])
        return new_pair, single


def _parallel_residual(x: torch.Tensor, outs: list[torch.Tensor]) -> torch.Tensor:
    """x + sum_i (y_i - x): the CUDA sum (fp32, one rounding) on H100, else the same in torch."""
    if (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float32)
        and torch.cuda.get_device_capability(x.device) == (9, 0)
        and x.numel() % 4 == 0
        and len(outs) <= 8
    ):
        from miniworld_engine.kernels.dit_variants import cuda

        x = x.contiguous()
        return cuda.parallel_residual(x, [y.contiguous() for y in outs])
    acc = x.to(torch.promote_types(x.dtype, torch.float32))
    total = acc + sum(y.to(acc.dtype) - acc for y in outs)
    return total.to(x.dtype)


class Pairformer(nn.Module):
    """The Pairformer stack consisting of multiple Pairformer blocks."""

    class Config(BaseModel):
        """Configuration for Pairformer module."""

        d_single: int = 384
        d_pair: int = 128
        n_head_tri_attention: int = 4
        n_head_attention: int = 16
        p_drop: float = 0.25
        use_single: bool = True
        use_self_attention: bool = True
        use_triangle_attention: bool = True
        use_bidirectional_trimul: bool = False
        use_qk_norm: bool = False
        implementation: ImplementationType = ImplementationType.PYTORCH
        n_block: int = 4
        n_checkpoint_segments: int | None = None
        use_checkpoint: bool = False

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
        self.config = config

        self.blocks = nn.ModuleList(
            [
                PairformerBlock(
                    d_single=config.d_single,
                    d_pair=config.d_pair,
                    n_head_tri_attention=config.n_head_tri_attention,
                    n_head_attention=config.n_head_attention,
                    p_drop=config.p_drop,
                    use_single=config.use_single,
                    use_self_attention=config.use_self_attention,
                    use_triangle_attention=config.use_triangle_attention,
                    use_bidirectional_trimul=config.use_bidirectional_trimul,
                    use_qk_norm=config.use_qk_norm,
                    implementation=config.implementation,
                )
                for _ in range(config.n_block)
            ],
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B L L d_pair"],
        Float[torch.Tensor, "B L d_single"] | None,
    ]:
        """Forward pass."""
        if self.config.use_checkpoint and torch.is_grad_enabled():
            for block in self.blocks:
                pair, single = checkpoint(
                    block, pair, single, mask, use_reentrant=False
                )
        elif self.config.n_checkpoint_segments is None or not torch.is_grad_enabled():
            for block in self.blocks:
                pair, single = block(pair, single, mask)
        else:

            def run_module(module):
                def forward(inputs):
                    p, s = inputs
                    return module(p, s, mask)

                return forward

            pair, single = checkpoint_sequential(
                [run_module(b) for b in self.blocks],
                segments=self.config.n_checkpoint_segments,
                use_reentrant=False,
                input=(pair, single),
            )

        return pair, single

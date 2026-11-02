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
                            impl (triton / hand-CUDA) per shape and arch

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

from miniworld_engine.modules.attention_pair_bias import AttentionPairBias
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.primitives import Dropout
from miniworld_engine.modules.transition import Transition
from miniworld_engine.modules.triangle_attention import TriangleAttention
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
    TriangleMultiplication,
)
from miniworld_engine.utils.typecheck import typecheck


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
    # ---- the released AF3-family block (every default is today's pair-track block) ----
    # Single track (AF3 Alg. 17 lines 9-10): ``d_single > 0`` adds ``AttentionPairBias`` over the pair logits and a single
    # ``Transition``; ``forward(pair, mask, single)`` then returns ``(pair, single)``. Its mask is ``mask``.
    d_single: int = 0
    n_head_single: int = 16
    single_gate_bias: float = 0.0  # Chai-1: a constant on the single attention's gate logits
    pair_transition_n: int = 4
    single_transition_n: int = 4
    # ``parallel``: every update of a track reads the representation ENTERING the block and the deltas are summed into it (the
    # single's pair logits read the pair entering too), where AF3 threads each update through the running activation.
    parallel: bool = False
    # RoseTTAFold3: the triangle multiplications divide the contraction by the real sequence length.
    divide_by_length: bool = False
    # AlphaFold 3's released code projects the ending node's pair bias from the pair as it is: ``transposed_bias=False``.
    transposed_bias: bool = True
    triangle_attention_gate_bias: bool = False
    triangle_attention_out_bias: bool = False
    untransposed_output: bool = False
    dual_output: bool = False


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
                divide_by_length=config.divide_by_length,
            )
            self.tri_multi_incoming = TriangleMultiplication(
                d_pair=config.d_pair,
                d_hidden=config.d_hidden_tri_multi,
                outgoing=False,
                implementation=implementation,
                p_drop=config.p_drop,  # drop_row owned by the module (residual is unconditional)
                divide_by_length=config.divide_by_length,
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
                transposed_bias=config.transposed_bias,
                gate_bias=config.triangle_attention_gate_bias,
                out_bias=config.triangle_attention_out_bias,
                untransposed_output=config.untransposed_output,
                dual_output=config.dual_output,
                implementation=implementation,
                p_drop=config.p_drop,  # drop_row owned by the module (residual unconditional)
            )
            self.tri_atten_ending = TriangleAttention(
                d_pair=config.d_pair,
                n_head=config.n_head_tri_attention,
                d_hidden=d_hidden_attn_total,
                starting=False,
                use_self_attention=config.use_self_attention,
                transposed_bias=config.transposed_bias,
                gate_bias=config.triangle_attention_gate_bias,
                out_bias=config.triangle_attention_out_bias,
                untransposed_output=config.untransposed_output,
                dual_output=config.dual_output,
                implementation=implementation,
                p_drop=config.p_drop,  # drop_col owned by the module (residual unconditional)
            )
        else:
            self.tri_atten_starting = None
            self.tri_atten_ending = None
        self.drop_row = Dropout(broadcast_dim=1, p_drop=config.p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=config.p_drop)
        self.transition_pair = Transition(
            config.d_pair, config.pair_transition_n, implementation=implementation
        )
        if config.d_single:
            self.single_attention = AttentionPairBias(
                config.d_single, config.d_pair, config.n_head_single,
                gate_bias=config.single_gate_bias, implementation=implementation,
            )
            self.single_transition = Transition(
                config.d_single, config.single_transition_n, implementation=implementation
            )

    @classmethod
    def from_released(cls, state_dict, *, config: PairformerConfig, prefix: str = "", dtype: torch.dtype | None = None,
                      norm_eps: float = 1e-5, implementation: ImplementationType = ImplementationType.PYTORCH) -> PairformerBlock:
        """The block from a released ``PairformerBlock`` state (names in ``modules.released``); ``config`` carries what the weights do
        not say (head counts, ``parallel``, ``divide_by_length``, ``transposed_bias`` ...)."""
        from miniworld_engine.modules import released

        block = cls(config, implementation=implementation)
        return released.load(block, released.pairformer_block_from_released(state_dict, prefix=prefix), dtype=dtype, norm_eps=norm_eps)

    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
        single: Float[torch.Tensor, "B L d_single"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"] | tuple[Float[torch.Tensor, "B L L d_pair"], Float[torch.Tensor, "B L d_single"]]:
        """Forward pass (residual updates). Pair track only: returns ``pair``. With ``config.d_single`` the single track is
        added and ``(pair, single)`` is returned (``single`` is then required)."""
        if self.config.d_single and single is None:
            raise ValueError("a block with a single track (config.d_single) needs the single representation")
        if not self.config.d_single and single is not None:
            raise ValueError("this block has no single track (config.d_single == 0)")
        pair_in = pair
        if self.config.parallel:
            pair = pair_in + self._pair_updates(pair_in, mask)
        else:
            pair = self._sequential_pair(pair, mask)
        if not self.config.d_single:
            return pair
        # A parallel block's single reads the pair ENTERING it; a sequential one the pair its own track just produced.
        logits_pair = pair_in if self.config.parallel else pair
        if self.config.parallel:
            single = (single + (self.single_attention(single, logits_pair, mask) - single)
                      + (self.single_transition(single) - single))
        else:
            single = self.single_transition(self.single_attention(single, logits_pair, mask))
        return pair, single

    def _sequential_pair(self, pair, mask):
        # trimul & transition OWN their residual (unconditional) + drop_row (module p_drop, fused
        # into the kernel epilogue for speed) — the block just calls ``module(pair, mask)``; there
        # is no external ``pair + drop_row(...)`` here. See the modules' constructor comments.
        if self.tri_multi is not None:
            pair = self.tri_multi(pair, mask)  # y = pair + drop_row(bidir_trimul(pair))
        else:
            # The constructor sets these two together, in the branch where tri_multi is None --
            # an invariant nothing in the types carries, so a checker reads both calls as calling
            # `None | TriangleMultiplication`. Asserting it states the contract where it is relied
            # on and turns a would-be `NoneType is not callable` into a named failure.
            assert self.tri_multi_outgoing is not None, "tri_multi is None but outgoing is unset"
            assert self.tri_multi_incoming is not None, "tri_multi is None but incoming is unset"
            pair = self.tri_multi_outgoing(pair, mask)  # y = pair + drop_row(trimul_out(pair))
            pair = self.tri_multi_incoming(pair, mask)  # y = pair + drop_row(trimul_in(pair))
        if self.tri_atten_starting is not None:
            # Triangle attention now OWNS its residual + dropout too (drop_row for starting,
            # drop_col for ending) — same contract as trimul. It is not kernel-fused yet (explicit
            # add inside the module), but the block just calls ``module(pair, mask)``.
            # Both halves are built together or not at all, same as the trimul pair above.
            assert self.tri_atten_ending is not None, "tri_atten starting is set but ending is not"
            pair = self.tri_atten_starting(pair, mask)  # y = pair + drop_row(attn_start(pair))
            pair = self.tri_atten_ending(pair, mask)    # y = pair + drop_col(attn_end(pair))
        # y = pair + transition(pair) (residual fused, no dropout)
        return self.transition_pair(pair)

    def _pair_updates(self, pair_in, mask):
        """The sum of every update's DELTA on ``pair_in`` (the ``parallel`` schedule). The modules own their residual, so a delta is
        ``module(x) - x`` (exact in fp32; in bf16 it carries one extra rounding of ``x + delta``)."""
        modules = ([self.tri_multi] if self.tri_multi is not None else [self.tri_multi_outgoing, self.tri_multi_incoming])
        if self.tri_atten_starting is not None:
            modules += [self.tri_atten_starting, self.tri_atten_ending]
        modules.append(self.transition_pair)
        total = None
        for module in modules:
            delta = (module(pair_in, mask) if module is not self.transition_pair else module(pair_in)) - pair_in
            total = delta if total is None else total + delta
        return total


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
        single: Float[torch.Tensor, "B L d_single"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"] | tuple[Float[torch.Tensor, "B L L d_pair"], Float[torch.Tensor, "B L d_single"]]:
        """Run the pair representation (and the single, with ``config.d_single``) through every block."""
        if single is None:
            for block in self.pairformer_blocks:
                pair = block(pair, mask)
            return pair
        for block in self.pairformer_blocks:
            pair, single = block(pair, mask, single)
        return pair, single

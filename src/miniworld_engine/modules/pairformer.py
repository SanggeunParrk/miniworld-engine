"""Pairformer block and stack: triangle multiplications, triangle attentions and a pair transition, each owning its residual; optionally
the single track (``AttentionPairBias`` over the pair logits + a single ``Transition``).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn

from miniworld_engine.modules.primitives import BackendModule, Dropout, MasterWeights, pair_inputs, random_input
from miniworld_engine.modules.transition import Transition
from miniworld_engine.modules.triangle import BidirectionalTriangleMultiplication, TriangleAttention, TriangleMultiplication


def _per_sample_call(pair: torch.Tensor) -> bool:
    """A :class:`PairformerBlock` call the block may run one sample at a time: ``B > 1`` and no autograd recorded (FoldForge R19)."""
    return pair.shape[0] > 1 and not torch.is_grad_enabled()


@dataclass
class PairformerConfig:
    """Pairformer hyper-parameters (AF3 defaults)."""

    d_pair: int = 128
    d_hidden_tri_multi: int = 128
    #: PER-HEAD channels of the triangle attention (its total hidden width is this times ``n_head_tri_attention``).
    d_hidden_tri_attention: int = 32
    n_head_tri_attention: int = 4
    p_drop: float = 0.25
    use_self_attention: bool = True
    n_block: int = 4
    # When False, the block drops both triangle-attention updates and keeps only the triangle-multiplication update + transition.
    use_triangle_attention: bool = True
    # When True, the two directional updates are replaced by one BidirectionalTriangleMultiplication.
    bidirectional_trimul: bool = False
    # ---- the released AF3-family block (every default is the pair-track block) ----
    #: The model whose triangle attention / multiplication variants the block builds (``TriangleAttention(model=...)``,
    #: ``TriangleMultiplication(model=...)``): e.g. ``"af3"`` (AF3 ending-bias orientation), ``"rf3"`` (gate / out biases, the trimul
    #: divided by the sequence length), ``"chai1"`` / ``"chai1_confidence"`` (Chai-1's output forms).
    model: str = "miniworld"
    #: Separate keys for the two kinds of triangle update (None: ``model``). RoseTTAFold3's fast mode, for example, is
    #: ``model="rf3", trimul_model="miniworld"``: the RF3 triangle attention with the plain multiplication (no division by the length).
    triattn_model: str | None = None
    trimul_model: str | None = None
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


class PairformerBlock(BackendModule, MasterWeights):
    """One Pairformer block: triangle multiplication outgoing / incoming (or one bidirectional update), triangle attention starting /
    ending, the pair transition; with ``config.d_single`` the single track after it."""

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 2,  # 2026-11-13: its sub-modules' anthropic paths (TriAttn folded block + cu129 core prebuilts, Transition folded residual)
        "anthropic:a100_sm80": 3,  # 2026-11-13: the A100 kit stack (its TriMul / TriAttn sub-modules on the kits' sm_80 native rows)
        "cuda:a100_sm80": 6,  # 6 (2026-11-14): TriMul D64 k3s; 5: TriAttn rf core
        "cuda:h100_sm90": 7,
    }  # 2026-11-11: the A100 sub-modules' inference weight forms read by captures; cuda 3 / 2 (2026-11-11): B > 1 without autograd runs
    # batched (the cuda AttentionPairBias takes B > 1: no refusal-driven per-sample path)
    # cuda:a100_sm80 4 (2026-11-12): its AttentionPairBias's generic sm_80 pair-bias backward faster (the same values)
    # cuda:h100_sm90 3 (2026-11-12): its TriMul / TriangleAttention training dropout scale by the H100 Philox kernel, TriMul small-L tiles
    # cuda:h100_sm90 4 (2026-11-13): its TriMul / Transition inference cast fp32-master weights once per parameter version
    # cuda:h100_sm90 5 (2026-11-13): its TriangleAttention at d_pair 128 x 4 heads projects the gate in the inference tail (fp32), d_pair
    # 256 x 8 heads keeps the wide front's CTA groups in step
    # cuda:h100_sm90 6 (2026-11-13): its TriangleAttention: d_pair 256 x 8 heads LayerNorm folded into the wide front's pack, d_pair 128
    # ending-node tail reads o transposed (bitwise the same), AF3 d_pair 64 one-kernel front up to L2048
    # cuda:h100_sm90 7 (2026-11-13): its TriangleAttention inference skips the dead trailing key tiles of a padded mask (bitwise the same)

    def __init__(self, config: PairformerConfig, master_weights: bool = False, *, backend: str = "pytorch") -> None:
        super().__init__()
        self.config = config
        trimul_model = config.model if config.trimul_model is None else config.trimul_model
        triattn_model = config.model if config.triattn_model is None else config.triattn_model

        if config.bidirectional_trimul:
            self.tri_multi = BidirectionalTriangleMultiplication(
                d_pair=config.d_pair,
                d_hidden=config.d_hidden_tri_multi,
                p_drop=config.p_drop,
            )
            self.tri_multi_outgoing = None
            self.tri_multi_incoming = None
        else:
            self.tri_multi = None
            self.tri_multi_outgoing = TriangleMultiplication(
                d_pair=config.d_pair,
                d_hidden=config.d_hidden_tri_multi,
                outgoing=True,
                p_drop=config.p_drop,
                model=trimul_model,
            )
            self.tri_multi_incoming = TriangleMultiplication(
                d_pair=config.d_pair,
                d_hidden=config.d_hidden_tri_multi,
                outgoing=False,
                p_drop=config.p_drop,
                model=trimul_model,
            )
        if config.use_triangle_attention:
            # ``d_hidden_tri_attention`` is the PER-HEAD width; TriangleAttention takes the TOTAL hidden width.
            d_hidden_attn_total = config.d_hidden_tri_attention * config.n_head_tri_attention
            attention = {
                "d_pair": config.d_pair,
                "n_head": config.n_head_tri_attention,
                "d_hidden": d_hidden_attn_total,
                "p_drop": config.p_drop,
                "model": triattn_model,
            }
            if not config.use_self_attention:  # bias-only: only TriangleAttention has it (a model variant refuses the argument)
                attention["use_self_attention"] = False
            self.tri_atten_starting = TriangleAttention(starting=True, **attention)
            self.tri_atten_ending = TriangleAttention(starting=False, **attention)
        else:
            self.tri_atten_starting = None
            self.tri_atten_ending = None
        # Kept for the module tree (no parameters): the updates own their dropout.
        self.drop_row = Dropout(broadcast_dim=1, p_drop=config.p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=config.p_drop)
        self.transition_pair = Transition(config.d_pair, config.pair_transition_n)
        if config.d_single:
            from miniworld_engine.modules.attention import AttentionPairBias

            self.single_attention = AttentionPairBias(
                config.d_single,
                config.d_pair,
                config.n_head_single,
                gate_bias=config.single_gate_bias,
            )
            self.single_transition = Transition(config.d_single, config.single_transition_n)
        if master_weights:
            self.set_master_weights()
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        single: torch.Tensor | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``pair`` ``[B, L, L, d_pair]``, ``mask`` ``[B, L]`` (bool). Pair track only: returns ``pair``. With ``config.d_single`` the
        single track is added and ``(pair, single)`` is returned (``single`` ``[B, L, d_single]`` is then required). ``extra_pair_bias``
        ``[B, n_head_single, L, L]`` is added to the single attention's pair-bias logits; it needs the single track.

        Workspace (FoldForge R19): a call without autograd (``torch.no_grad`` / ``inference_mode``) and ``B > 1`` on ``"pytorch"`` runs
        the block one sample at a time into the batched output (every statement acts per sample: the same math), so its temporaries
        are one sample's (the reference composition's are about 8-10 pair-sized tensors) plus the output, not ``B`` times that. The
        outputs equal the batched composition's up to GEMM rounding (a GEMM over one sample's rows may run another cuBLAS kernel). A
        call that records autograd keeps the batched composition (its saved activations are the batch's anyway). The kernel backends
        run the batched composition (every sub-module takes ``B > 1``; the ``cuda`` ``AttentionPairBias`` too, since the FoldForge 5I28
        confidence-head change)."""
        if pair.is_cuda and not torch.compiler.is_compiling() and not torch.cuda.is_current_stream_capturing():
            # a finished item's hoisted tables (a token DiT's pair bias: 11.3 GiB at OpenDDE's 3968 structural tokens) go before this
            # block's pair-sized temporaries are allocated, not at the next item's diffusion (FoldForge 5I28 OpenDDE n = 16)
            from miniworld_engine.kernels import _capture

            _capture.release_dead_inputs()
        if _per_sample_call(pair) and self.backend == "pytorch":
            return self._per_sample(pair, mask, single, extra_pair_bias)
        return self._block(pair, mask, single, extra_pair_bias)

    def _per_sample(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
        single: torch.Tensor | None,
        extra_pair_bias: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """:meth:`_block` on each sample ``b`` of ``pair[b : b + 1]`` (a size-1 leading dim of ``mask`` / ``single`` / ``extra_pair_bias``
        broadcasts), written into one ``[B, ...]`` output per track as it is produced."""
        count = pair.shape[0]

        def pick(t: torch.Tensor | None, b: int) -> torch.Tensor | None:
            return t if t is None or t.shape[0] == 1 else t[b : b + 1]

        out_pair: torch.Tensor | None = None
        out_single: torch.Tensor | None = None
        for b in range(count):
            res = self._block(pair[b : b + 1], pick(mask, b), pick(single, b), pick(extra_pair_bias, b))
            p, s = res if isinstance(res, tuple) else (res, None)
            if out_pair is None:
                out_pair = p.new_empty((count, *p.shape[1:]))
            out_pair[b : b + 1] = p
            if s is not None:
                if out_single is None:
                    out_single = s.new_empty((count, *s.shape[1:]))
                out_single[b : b + 1] = s
            del res, p, s
        assert out_pair is not None  # count > 1
        return out_pair if out_single is None else (out_pair, out_single)

    def _block(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        single: torch.Tensor | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The block's composition on the whole batch (the reference statements; see :meth:`_forward_pytorch`)."""
        if self.config.d_single and single is None:
            raise ValueError("a block with a single track (config.d_single) needs the single representation")
        if not self.config.d_single and single is not None:
            raise ValueError("this block has no single track (config.d_single == 0)")
        if extra_pair_bias is not None and not self.config.d_single:
            raise ValueError("extra_pair_bias biases the single attention: the block needs the single track (config.d_single)")
        pair_in = pair
        pair = pair_in + self._pair_updates(pair_in, mask) if self.config.parallel else self._sequential_pair(pair, mask)
        if not self.config.d_single:
            return pair
        # A parallel block's single reads the pair ENTERING it; a sequential one the pair its own track just produced.
        logits_pair = pair_in if self.config.parallel else pair
        if self.config.parallel:
            single = (
                single
                + (self.single_attention(single, logits_pair, mask, extra_pair_bias) - single)
                + (self.single_transition(single) - single)
            )
        else:
            single = self.single_transition(self.single_attention(single, logits_pair, mask, extra_pair_bias))
        return pair, cast(torch.Tensor, single)  # the single track: given with config.d_single

    # ------------------------------------------------------------------------------------------------------------- "cuda" / "triton"
    # The block owns no kernel of its own: ``set_backend`` hands the name to every sub-module (the triangle multiplications, the triangle
    # attentions, the pair transition, and with the single track the AttentionPairBias and the single Transition), and ``forward`` is the
    # reference composition, so each update runs on ITS backend. A sub-module without that backend stays on "pytorch" for the module tree,
    # but ``unsupported`` then refuses the whole call naming it: the block never runs a mixed schedule silently.
    # ``"anthropic"``: the release's fused pairformer layers are drivers of the same pieces, not one kernel -- the boltz2 kit's pairfuse
    # (``bz_pairfuse.pfm_forward``, fast row ``trimul=core.fast, triatt=core.fast, transition=core.fast``: the core's TriMul face, pair_fused's
    # tri_attn_block, the core's transition provider, in the stock layer order) and protenix_v2's BLK2 (c_z 256 only) -- and both patch their
    # stock model classes (boltz / protenix must be installed). The block's anthropic backend is that composition on the sub-modules' own
    # anthropic backends (TriangleMultiplication: core TriMul fast; TriangleAttention: tri_attn_block with core tier:fast; Transition: the
    # core provider), the residuals added by each update instead of folded into the kernels' epilogues.
    def _backend_children(self) -> list[tuple[str, BackendModule]]:
        names = [
            "tri_multi",
            "tri_multi_outgoing",
            "tri_multi_incoming",
            "tri_atten_starting",
            "tri_atten_ending",
            "transition_pair",
            "single_attention",
            "single_transition",
        ]
        return [(n, getattr(self, n)) for n in names if getattr(self, n, None) is not None]

    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        for _, child in self._backend_children():
            child.set_backend(backend if backend in child.backends else "pytorch")
        return self

    def unsupported(self, pair, mask=None, single=None, extra_pair_bias=None) -> str | None:
        """The first sub-module that has no such backend, or whose range does not take this call (its own reason); None if all run (a
        kernel backend runs the call batched: :meth:`_forward_pytorch`)."""
        if self.backend == "pytorch":
            return None
        return self._refusal(pair, mask, single, extra_pair_bias)

    def _refusal(self, pair, mask=None, single=None, extra_pair_bias=None) -> str | None:
        """The call's refusal on the selected backend (None: every sub-module takes it)."""
        calls = {"single_attention": (single, pair, mask, extra_pair_bias), "single_transition": (single,), "transition_pair": (pair,)}
        for name, child in self._backend_children():
            if self.backend not in child.backends:
                return f"{name} ({type(child).__name__}) has no {self.backend!r} backend (available: {', '.join(child.backends)})"
            why = child.unsupported(*calls.get(name, (pair, mask)))
            if why is not None:
                return f"{name} ({type(child).__name__}): {why}"
        return None

    _forward_anthropic = _forward_cuda = _forward_triton = _forward_pytorch

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        inputs = pair_inputs(self.config.d_pair, length=length, samples=samples, dtype=dtype, device=device, generator=generator)
        if self.config.d_single:
            inputs["single"] = random_input((samples, length, self.config.d_single), dtype=dtype, device=device, generator=generator)
        return inputs

    def _sequential_pair(self, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # every update owns its residual + dropout: the block just calls ``module(pair, mask)``
        if self.tri_multi is not None:
            pair = self.tri_multi(pair, mask)
        else:
            assert self.tri_multi_outgoing is not None, "tri_multi is None but outgoing is unset"
            assert self.tri_multi_incoming is not None, "tri_multi is None but incoming is unset"
            pair = self.tri_multi_outgoing(pair, mask)
            pair = self.tri_multi_incoming(pair, mask)
        if self.tri_atten_starting is not None:
            assert self.tri_atten_ending is not None, "tri_atten starting is set but ending is not"
            pair = self.tri_atten_starting(pair, mask)
            pair = self.tri_atten_ending(pair, mask)
        return self.transition_pair(pair)

    def _pair_updates(self, pair_in: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """The sum of every update's DELTA on ``pair_in`` (the ``parallel`` schedule). The modules own their residual, so a delta is
        ``module(x) - x`` (exact in fp32; in bf16 it carries one extra rounding of ``x + delta``)."""
        modules: list[Any] = [self.tri_multi] if self.tri_multi is not None else [self.tri_multi_outgoing, self.tri_multi_incoming]
        if self.tri_atten_starting is not None:
            modules += [self.tri_atten_starting, self.tri_atten_ending]
        modules.append(self.transition_pair)
        total = None
        for module in modules:
            delta = (module(pair_in, mask) if module is not self.transition_pair else module(pair_in)) - pair_in
            total = delta if total is None else total + delta
        assert total is not None  # modules always holds transition_pair
        return total


class Pairformer(BackendModule, MasterWeights):
    """The Pairformer stack: ``config.n_block`` :class:`PairformerBlock` in sequence."""

    backends = ("pytorch", "anthropic", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {"anthropic": 2, "anthropic:a100_sm80": 3}  # 2026-11-13: the blocks' anthropic paths (see PairformerBlock)

    def __init__(self, config: PairformerConfig | None = None, master_weights: bool = False, *, backend: str = "pytorch") -> None:
        super().__init__()
        self.config = config or PairformerConfig()
        self.pairformer_blocks = nn.ModuleList([PairformerBlock(self.config) for _ in range(self.config.n_block)])
        if master_weights:
            self.set_master_weights()
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        single: torch.Tensor | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the pair representation (and the single, with ``config.d_single``) through every block; ``extra_pair_bias`` (single track
        only) is added to every block's single-attention logits."""
        if single is None:
            if extra_pair_bias is not None:
                raise ValueError("extra_pair_bias biases the single attention: pass the single representation")
            for block in self.pairformer_blocks:
                pair = block(pair, mask)
            return pair
        for block in self.pairformer_blocks:
            pair, single = block(pair, mask, single, extra_pair_bias)
        return pair, cast(torch.Tensor, single)

    # "cuda" / "triton": every block takes the backend (see :class:`PairformerBlock`); the blocks are alike, so the first one states the range.
    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        for block in cast("list[PairformerBlock]", self.pairformer_blocks):
            block.set_backend(backend)
        return self

    def unsupported(self, pair, mask=None, single=None, extra_pair_bias=None) -> str | None:
        return (
            cast(PairformerBlock, self.pairformer_blocks[0]).unsupported(pair, mask, single, extra_pair_bias)
            if len(self.pairformer_blocks)
            else None
        )

    _forward_anthropic = _forward_cuda = _forward_triton = _forward_pytorch

"""ESMFold2 ``SWAAtomTransformer`` (Algorithm 8) — DiT adaLN-Zero blocks with
sliding-window 3D-RoPE attention and no atom-pair tensor.

Each block is a canonical DiT adaLN-Zero block (Peebles & Xie, arXiv:2212.09748):
the conditioning predicts ``shift / scale / gate`` for both the attention and the
FFN sub-layer; the residual branches are gated by a zero-initialised modulation
so they start as the identity. This is intentionally *not* the AF3 conditioning
(``sigmoid``-scale AdaLN + ``bias=-2`` output gate) used by
:class:`~miniworld_engine.architecture.layers.AugmentedAttentionPairBias`; it mirrors ESMFold2,
which reverts the atom transformer to the original DiT block while keeping AF3's
``AttentionPairBias`` only at the token level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import BaseModel
from torch.utils.checkpoint import checkpoint_sequential

from miniworld_engine import kernels, settings
from miniworld_engine.architecture._engine_impl import (
    to_engine_impl_pointwise,
    torch_pointwise,
)
from miniworld_engine.architecture.blocks._engine_impl import to_engine_impl
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.architecture.layers.ops import mp_sum, mp_swish_gate
from miniworld_engine.architecture.primitives import Linear, MPLinear
from miniworld_engine.architecture.typing import typecheck
from miniworld_engine.modules import RMSNorm
from miniworld_engine.modules.dispatch import fallback_autocast, restricted_card
from miniworld_engine.modules.swa_atom_attention.module import (
    SWA3DRoPEAttention,
    build_3d_rope,
)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN with hidden size rounded up to a multiple of 256."""

    def __init__(
        self,
        d_model: int,
        expansion_ratio: int = 2,
        *,
        magnitude_preserving: bool = False,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        hidden = ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256
        self.magnitude_preserving = magnitude_preserving
        self.implementation = implementation
        self._pointwise = torch_pointwise()
        dense = MPLinear if magnitude_preserving else Linear
        self.w_up = dense(d_model, 2 * hidden, bias=False, init="normal")
        self.w_down = dense(hidden, d_model, bias=False, init="normal")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.implementation == ImplementationType.MINIWORLD_ENGINE
            and x.is_cuda
            and not self.magnitude_preserving
            and not self._pointwise
            and not restricted_card(x.device)  # H100 / B200 have no Triton fallback: the PyTorch statements below run
        ):
            # Cast through autograd, retaining the original FP32 master weights.
            dtype = (
                torch.get_autocast_dtype("cuda")
                if torch.is_autocast_enabled("cuda")
                else x.dtype
            )
            wa, wb = self.w_up.weight.to(dtype).chunk(2, dim=0)
            with torch.autocast("cuda", enabled=False):
                return kernels.triton_swiglu_ffn(
                    x.to(dtype), wa, wb, self.w_down.weight.to(dtype)
                )
        # H100 / B200 land here (no Triton fallback): fp32 master weights under a bf16 activation run under bf16 autocast
        with fallback_autocast(x, self, x.is_cuda and restricted_card(x.device)):
            x1, x2 = self.w_up(x).chunk(2, dim=-1)
            x = mp_swish_gate(x1, x2) if self.magnitude_preserving else F.silu(x1) * x2
            return self.w_down(x)


class SWAAtomBlock(nn.Module):
    """adaLN-Zero + sliding-window 3D-RoPE attention + SwiGLU FFN."""

    def __init__(
        self,
        d_atom: int,
        d_cond: int,
        n_head: int,
        *,
        half_window: int = 64,
        expansion_ratio: int = 2,
        magnitude_preserving: bool = False,
        mp_full: bool = False,
        mp_residual: bool = False,
        residual_t: float = 0.3,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        magnitude_preserving = magnitude_preserving or mp_full
        self.mp_residual = mp_residual or mp_full
        self.residual_t = residual_t
        self.implementation = implementation
        self.mp_full = mp_full
        self._pointwise = torch_pointwise()
        self.attn_norm = RMSNorm(
            d_atom,
            elementwise_affine=False,
            implementation=to_engine_impl_pointwise(implementation),
        )
        self.ffn_norm = RMSNorm(
            d_atom,
            elementwise_affine=False,
            implementation=to_engine_impl_pointwise(implementation),
        )
        # Plain ESMFold2 uses adaLN-Zero. Under full MP, this modulation is an
        # active MP projection; residual magnitude is controlled by mp_sum.
        mod_cls = MPLinear if mp_full else Linear
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            mod_cls(
                d_cond, 6 * d_atom, bias=False, init="normal" if mp_full else "zero"
            ),
        )
        self.attn = SWA3DRoPEAttention(
            d_atom,
            n_head,
            half_window=half_window,
            magnitude_preserving=magnitude_preserving,
            mp_full=mp_full,
            implementation=to_engine_impl(implementation),
        )
        # the engine attention keeps its flash core; its q/k-norm + RoPE and
        # output gate are PyTorch ops under the B200 policy
        self.attn.torch_pointwise = self._pointwise
        self.ffn = SwiGLUFFN(
            d_atom,
            expansion_ratio,
            magnitude_preserving=magnitude_preserving,
            implementation=implementation,
        )

    def forward(
        self,
        q: Float[torch.Tensor, "N S d_atom"],
        c: Float[torch.Tensor, "N S d_cond"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d_atom"]:
        if (
            self.implementation == ImplementationType.MINIWORLD_ENGINE
            and q.is_cuda
            and not self.mp_full
            and not self._pointwise
            and not restricted_card(q.device)  # H100 / B200 have no Triton fallback: the PyTorch statements below run
        ):
            # Keep the six-chunk checkpoint layout and the adaLN-Zero algorithm.
            # The engine SWADiTBlock implements different conditioning, so it
            # cannot replace this block without changing the trained model.
            dtype = (
                torch.get_autocast_dtype("cuda")
                if torch.is_autocast_enabled("cuda")
                else q.dtype
            )
            activated = F.silu(c).to(dtype)
            weights = self.adaln_modulation[1].weight.to(dtype).chunk(6, dim=0)
            eps = self.attn_norm.effective_eps(q.dtype)
            with torch.autocast("cuda", enabled=False):
                attn_in, gate_a = kernels.triton_rmsnorm_adamod(
                    q.to(dtype), activated, weights[1], weights[0], weights[2], eps=eps
                )
            attn = gate_a * self.attn(attn_in, attention_params)
            q = mp_sum(q, attn, self.residual_t) if self.mp_residual else q + attn
            with torch.autocast("cuda", enabled=False):
                ffn_in, gate_f = kernels.triton_rmsnorm_adamod(
                    q.to(dtype),
                    activated,
                    weights[4],
                    weights[3],
                    weights[5],
                    eps=self.ffn_norm.effective_eps(q.dtype),
                )
            ffn = gate_f * self.ffn(ffn_in)
            return mp_sum(q, ffn, self.residual_t) if self.mp_residual else q + ffn
        with fallback_autocast(q, self, q.is_cuda and restricted_card(q.device)):
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln_modulation(
                c
            ).chunk(6, dim=-1)
            attn_in = self.attn_norm(q) * (1 + scale_a) + shift_a
            attn = gate_a * self.attn(attn_in, attention_params)
            q = mp_sum(q, attn, self.residual_t) if self.mp_residual else q + attn
            ffn_in = self.ffn_norm(q) * (1 + scale_f) + shift_f
            ffn = gate_f * self.ffn(ffn_in)
            q = mp_sum(q, ffn, self.residual_t) if self.mp_residual else q + ffn
            return q


def _pads_atoms(q: torch.Tensor, s: int) -> bool:
    """Whether the fused block call pads the atom axis: bf16 on B200 (the engine's sm_100a block wants S % 128 == 0)."""
    return (
        s % 128 != 0
        and q.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(q.device) == (10, 0)
    )


class SWAAtomTransformer(nn.Module):
    """Stack of :class:`SWAAtomBlock` (ESMFold2 Algorithm 8)."""

    class Config(BaseModel):
        """Configuration for the SWA atom transformer."""

        d_atom: int = 128
        d_cond: int = 128
        n_block: int = 3
        # Activation-checkpoint the atom blocks (None = off; 1..n_block). The atom DiT
        # activations (A*B*L_atom x d_atom) dominate the fp32 diffusion memory at large
        # num_augment, so this is the primary lever for fitting big crops.
        n_checkpoint_segments: int | None = None
        n_head: int = 4
        swa_window_size: int = 128
        expansion_ratio: int = 2
        # "esmfold2": ESMFold2 SWAAtomBlock (adaLN-Zero + SwiGLU).
        # "af3": keep MiniWorld's AF3 atom block (AdaLN + sigmoid gate + to_scale
        #   + ConditionedTransition), swap ONLY the attention core to 3D-RoPE
        #   sliding-window (no atom pair). See RoPESWAAF3Transformer.
        block_style: str = "esmfold2"
        # global_attn: with block_style "af3", use GLOBAL self-attention (all
        # atoms attend to all) instead of the sliding window — keeps 3D RoPE and
        # no atom pair. O(L_atom^2); for training-crop-size experiments only.
        global_attn: bool = False
        # local_structure_attn: with block_style "af3", attend to explicit
        # neighbors = nearest atoms in sequence order plus nearest atoms by the
        # current noisy coordinates x_t, excluding sequence neighbors. This uses
        # a gather-based sparse attention path because FA window kernels cannot
        # express arbitrary dynamic neighbors.
        local_structure_attn: bool = False
        seq_neighbors: int = 128
        structure_neighbors: int = 128
        structure_query_chunk_size: int = 128
        sparse_attention_query_chunk_size: int = 64
        n_spatial_rope_pairs_per_axis: int = 2
        spatial_rope_base_frequency: float = 20.0
        n_uid_rope_pairs: int = 10
        uid_rope_base_frequency: float = 10000.0
        magnitude_preserving: bool = False
        use_rotation: bool = False
        mp_full: bool = False
        mp_residual: bool = False
        residual_t: float = 0.3
        implementation: ImplementationType | None = None
        # Fused engine path (miniworld_engine.kernels.swa_dit; bf16 or fp32 activations, the weights in that dtype or fp32 (an
        # fp32 master over bf16 activations), d_atom 128 / 4 heads / window 128 /
        # SwiGLU hidden 256, no MP options): three kernels per block forward (bf16: CUDA wgmma on H100, Triton elsewhere;
        # fp32: Triton), the adaLN modulation hoisted to the augment-invariant conditioning. Same parameters (checkpoint
        # compatible).
        # None -> env settings.swa_fused ("1" on).
        fused_triton: bool | None = None

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.d_atom // config.n_head
        self.blocks = nn.ModuleList(
            [
                SWAAtomBlock(
                    config.d_atom,
                    config.d_cond,
                    config.n_head,
                    half_window=config.swa_window_size // 2,
                    expansion_ratio=config.expansion_ratio,
                    magnitude_preserving=config.magnitude_preserving,
                    mp_full=config.mp_full,
                    mp_residual=config.mp_residual,
                    residual_t=config.residual_t,
                    implementation=config.implementation
                    or ImplementationType.MINIWORLD_ENGINE,
                )
                for _ in range(config.n_block)
            ],
        )

    @typecheck
    def build_rope(
        self,
        ref_pos: Float[torch.Tensor, "B N 3"],
        ref_space_uid: Int[torch.Tensor, "B N"],
    ) -> tuple[Float[torch.Tensor, "B N half"], Float[torch.Tensor, "B N half"]]:
        """Build the (cos, sin) 3D RoPE tensors for one batch (no augment axis)."""
        return build_3d_rope(
            ref_pos,
            ref_space_uid,
            self.head_dim,
            n_spatial_per_axis=self.config.n_spatial_rope_pairs_per_axis,
            n_uid_pairs=self.config.n_uid_rope_pairs,
            spatial_base_freq=self.config.spatial_rope_base_frequency,
            uid_base_freq=self.config.uid_rope_base_frequency,
        )

    def _global_cast(self, q: torch.Tensor) -> bool:
        """Whether an fp32 caller runs the global-attention block through a bf16 cast.

        Global attention (a huge ``swa_window_size``) runs the engine's sm_100a
        block, which is bf16 only: the fp32 caller gets it through a bf16 cast of
        the stream and the weights (autograd keeps the fp32 master weights; the
        output comes back in the input dtype). That is a precision change, so it
        is opt-in with ``settings.swa_global_bf16=1``.
        """
        return (
            q.dtype == torch.float32
            and settings.current().swa_global_bf16
            and self._is_global()
        )

    def _is_global(self) -> bool:
        try:
            from miniworld_engine.kernels.swa_dit.interface import is_global
        except ImportError:  # an engine without the global-attention block
            return False
        return is_global(self.config.swa_window_size // 2)

    def _fused_ok(
        self, q: torch.Tensor, c: torch.Tensor, attention_params: tuple
    ) -> bool:
        cfg = self.config
        want = (
            cfg.fused_triton
            if cfg.fused_triton is not None
            else settings.current().swa_fused
        )
        if not want or not q.is_cuda:
            return False
        from miniworld_engine.kernels.swa_dit.interface import (
            highest_fp32_on_hopper_blackwell,
        )

        if highest_fp32_on_hopper_blackwell(q, self.config.swa_window_size // 2):
            return False  # fp32 under "highest" off the native H100 stages: the per-block PyTorch statements (no Triton on H100 / B200)
        blk = self.blocks[0]
        is_global = self._is_global()
        cast = self._global_cast(q)
        hopper = torch.cuda.get_device_capability(q.device) == (9, 0)
        if is_global and not (q.dtype == torch.bfloat16 or cast or hopper):
            return False  # global attention on B200: bf16 stages only (see _global_cast); H100's native stages take fp32 too
        if is_global and torch.cuda.get_device_capability(q.device) not in ((10, 0), (9, 0)):
            return False
        # The engine block runs bf16 or fp32 activations and conditioning, with the
        # weights in that dtype or fp32 (an fp32 master over bf16 activations: the
        # engine casts them for its kernels outside autograd); the global cast makes
        # the fp32 caller's operands bf16.
        return (
            q.dtype in (torch.bfloat16, torch.float32)
            and c.dtype == q.dtype
            and cfg.d_atom == 128
            and cfg.d_cond == 128
            and cfg.n_head == 4
            and (cfg.swa_window_size == 128 or is_global)
            and not (
                cfg.magnitude_preserving
                or cfg.mp_full
                or cfg.mp_residual
                or cfg.use_rotation
            )
            and blk.ffn.w_down.weight.shape[1] == 256
            and len(attention_params) == 6
            and all(
                w.dtype in (q.dtype, torch.float32) or cast
                for b in self.blocks
                for w in (
                    b.adaln_modulation[1].weight,
                    b.attn.Wqkv.weight,
                    b.attn.gate_proj.weight,
                    b.attn.out_proj.weight,
                    b.ffn.w_up.weight,
                    b.ffn.w_down.weight,
                )
            )
        )

    def _forward_fused(self, q, c, attention_params, c_base):
        # The fused block lives in the engine (kernels/swa_dit, moved from team-gm 14f2c73).
        from miniworld_engine.kernels.swa_dit.interface import (
            swa_dit_block,
            swa_dit_hoist_modulation,
        )

        cos, sin, seqused, _cu, _max, _valid = attention_params
        _, s, _ = q.shape
        if (
            c_base is None
        ):  # no augment structure known: one modulation row per (n, atom)
            c_base = c
        out_dtype = q.dtype
        cast = self._global_cast(q)
        if cast:  # fp32 caller, global attention: the bf16 block (_global_cast)
            q, c_base = q.to(torch.bfloat16), c_base.to(torch.bfloat16)
        b = c_base.shape[0]
        cos, sin = cos[:b], sin[:b]
        s_in = s
        if _pads_atoms(q, s):
            # the engine's B200 block takes a multiple of 128 atoms (callers pad; it refuses otherwise): zero rows appended,
            # seqused unchanged, so the padded atoms are never keys and the first s rows come back unchanged
            s = (s + 127) // 128 * 128
            q, c_base, cos, sin = (
                torch.nn.functional.pad(t, (0, 0, 0, s - s_in))
                for t in (q, c_base, cos, sin)
            )
        cos_b = (
            cos.reshape(b * s, -1).contiguous()
        )  # build_attention_params repeats the per-batch RoPE over augments
        sin_b = sin.reshape(b * s, -1).contiguous()
        hw = self.config.swa_window_size // 2

        def run(block, x):
            ws = [
                block.adaln_modulation[1].weight,
                block.attn.Wqkv.weight,
                block.attn.gate_proj.weight,
                block.attn.out_proj.weight,
                block.ffn.w_up.weight,
                block.ffn.w_down.weight,
            ]
            if cast:
                ws = [w.to(torch.bfloat16) for w in ws]
            mod = swa_dit_hoist_modulation(c_base, ws[0])
            return swa_dit_block(x, mod, cos_b, sin_b, seqused, *ws[1:], b, hw)

        if self.config.n_checkpoint_segments is None or not torch.is_grad_enabled():
            for block in self.blocks:
                q = run(block, q.contiguous())
        else:
            from torch.utils.checkpoint import checkpoint

            for block in (
                self.blocks
            ):  # per-block activation checkpointing (the n_checkpoint_segments knob)
                q = checkpoint(run, block, q.contiguous(), use_reentrant=False)
        q = q if s == s_in else q[:, :s_in].contiguous()
        return q.to(out_dtype) if cast else q

    @typecheck
    def forward(
        self,
        q: Float[torch.Tensor, "N S d_atom"],
        c: Float[torch.Tensor, "N S d_cond"],
        attention_params: tuple,
        c_base: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "N S d_atom"]:
        """Forward pass over the flattened ``N = A * B`` atom sequence.  ``c_base`` [B, S, d_cond]: the augment-invariant
        conditioning that ``c`` repeats (``c[a*B + b] == c_base[b]``); lets the fused path compute the modulation once per atom."""
        if self._fused_ok(q, c, attention_params):
            return self._forward_fused(q, c, attention_params, c_base)
        if self.config.n_checkpoint_segments is None:
            for block in self.blocks:
                q = block(q, c, attention_params)
            return q

        # Activation checkpointing over the atom blocks: only segment-boundary q is kept;
        # c / attention_params are per-call constants, closed over here.
        def run_module(module):
            def forward(x):
                return module(x, c, attention_params)

            return forward

        return checkpoint_sequential(
            [run_module(b) for b in self.blocks],
            segments=self.config.n_checkpoint_segments,
            use_reentrant=False,
            input=q,
        )

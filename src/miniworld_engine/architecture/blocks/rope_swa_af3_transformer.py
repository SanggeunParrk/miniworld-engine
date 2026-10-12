"""AF3 atom block with a 3D-RoPE sliding-window attention core (no atom pair).

Variant "2": keep MiniWorld's AF3 atom transformer block — AdaptiveLayerNorm
input modulation, the AF3 sigmoid value-gate (`to_gate`) + `to_out`, the AF3
``to_scale`` output gate (bias=-2), and the AF3 ``ConditionedTransition`` FFN —
and change ONLY the attention core: the block-local QK + O(L^2) atom-pair bias
is replaced by sliding-window attention with 3D RoPE on q/k (no pair tensor).

This is intentionally NOT the ESMFold2 ``SWAAtomTransformer`` block (adaLN-Zero
+ SwiGLU). It reuses the SWA RoPE / varlen plumbing (``build_3d_rope`` and
``build_attention_params``) and exposes the same interface as
``SWAAtomTransformer`` (``Config`` / ``build_rope`` / ``forward(q, c, params)``
on a flattened ``[N=A*B, S, d]`` sequence), so the SWA atom encoder/decoder can
instantiate it interchangeably.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from miniworld_engine.architecture.blocks._engine_impl import to_engine_impl
from miniworld_engine.architecture.blocks.swa_atom_transformer import SWAAtomTransformer
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.architecture.layers.ops import mp_sum, sigmoid_gate
from miniworld_engine.architecture.primitives import Linear, MPLinear
from miniworld_engine.architecture.typing import typecheck
from miniworld_engine.modules import AdaptiveLayerNorm, ConditionedTransition, RMSNorm
from miniworld_engine.modules.swa_atom_attention import (
    apply_rotary_emb_3d,
    build_3d_rope,
    flash_window_seqused,
    sparse_neighbor_attention,
)
from miniworld_engine.modules.swa_atom_attention.module import _flash_backend


class RoPESWAAttentionAF3(nn.Module):
    """AF3-style attention (AdaLN in, sigmoid gate, to_scale out-gate) whose core
    is sliding-window attention with 3D RoPE on q/k — no atom-pair bias.

    Mirrors ``AugmentedAttentionPairBias`` exactly except: there is no
    ``ln_pair``/``to_bias`` pair branch, and the QK-softmax core is a
    sliding-window FlashAttention with rotary q/k instead of dense QK + bias.
    Operates on a flattened ``[N, S, d]`` sequence (``N = A * B``).
    """

    def __init__(
        self,
        d_single: int,
        d_cond: int,
        n_head: int,
        *,
        half_window: int = 64,
        use_qk_norm: bool = True,
        magnitude_preserving: bool = False,
        use_rotation: bool = False,
        mp_full: bool = False,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        if d_single % n_head != 0:
            msg = f"d_single ({d_single}) must be divisible by n_head ({n_head})."
            raise ValueError(msg)
        self.n_head = n_head
        self.head_dim = d_single // n_head
        self.scale = self.head_dim**-0.5
        self.half_window = half_window
        self.use_qk_norm = use_qk_norm
        magnitude_preserving = magnitude_preserving or mp_full
        self.magnitude_preserving = magnitude_preserving
        self.mp_full = mp_full
        self.implementation = implementation
        if use_rotation or mp_full:
            raise ValueError(
                "The installed engine does not support legacy rotation/MP AdaLN"
            )

        # AF3 block pieces, with DiT-MP/EDM2 init when mp_full is enabled.
        dense = MPLinear if magnitude_preserving else Linear
        self.ada_ln_in = AdaptiveLayerNorm(
            d_single, d_cond, implementation=to_engine_impl(implementation)
        )
        self.to_query = dense(
            d_single, self.head_dim * n_head, bias=True, init="normal"
        )
        self.to_key = dense(d_single, self.head_dim * n_head, bias=False, init="normal")
        self.to_value = dense(
            d_single, self.head_dim * n_head, bias=False, init="normal"
        )
        if use_qk_norm:
            self.norm_query = RMSNorm(
                self.head_dim, implementation=to_engine_impl(implementation)
            )
            self.norm_key = RMSNorm(
                self.head_dim, implementation=to_engine_impl(implementation)
            )
        gate_cls = MPLinear if mp_full else Linear
        out_cls = MPLinear if mp_full else Linear
        self.to_gate = gate_cls(
            d_single,
            self.head_dim * n_head,
            bias=False,
            init="normal" if mp_full else "gating",
        )
        self.to_out = out_cls(
            self.head_dim * n_head,
            d_single,
            bias=False,
            init="normal" if mp_full else "zero",
        )
        scale_cls = MPLinear if mp_full else Linear
        self.to_scale = scale_cls(
            d_cond, d_single, bias=True, init="normal" if mp_full else "default"
        )
        self.to_scale.bias.data.fill_(-2.0)

    def forward(
        self,
        single: Float[torch.Tensor, "N S d"],
        cond: Float[torch.Tensor, "N S d_cond"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d"]:
        cos, sin, seqused, cu_seqlens, max_seqlen, valid, *extra = attention_params
        neighbor_idx = neighbor_mask = None
        sparse_query_chunk_size = 64
        if extra:
            neighbor_idx, neighbor_mask, *maybe_chunk = extra
            if maybe_chunk:
                sparse_query_chunk_size = maybe_chunk[0]
        n, s = single.shape[:2]

        x = self.ada_ln_in(single, cond)
        q = self.to_query(x).view(n, s, self.n_head, self.head_dim)
        k = self.to_key(x).view(n, s, self.n_head, self.head_dim)
        v = self.to_value(x).view(n, s, self.n_head, self.head_dim)
        if self.use_qk_norm:
            q = self.norm_query(q)
            k = self.norm_key(k)
        q = apply_rotary_emb_3d(q, cos, sin)
        k = apply_rotary_emb_3d(k, cos, sin)

        if neighbor_idx is not None:
            out = sparse_neighbor_attention(
                q,
                k,
                v,
                neighbor_idx,
                neighbor_mask,
                valid,
                self.scale,
                query_chunk_size=sparse_query_chunk_size,
            )
        elif self.implementation == ImplementationType.PYTORCH or not single.is_cuda:
            out = self._sdpa_band(q, k, v, valid)
        elif _flash_backend(single.device) is not None:
            out = self._flash_window(
                q, k, v, cu_seqlens, seqused, max_seqlen, valid, n, s
            )
        else:
            raise RuntimeError(
                "Engine AF3 SWA attention requires a usable FlashAttention backend on CUDA"
            )

        out = out.reshape(n, s, -1)
        out = sigmoid_gate(self.to_gate(x), out)
        out = self.to_out(out)
        return sigmoid_gate(self.to_scale(cond), out)

    def _flash_window(self, q, k, v, cu_seqlens, seqused, max_seqlen, valid, n, s):
        return flash_window_seqused(
            q,
            k,
            v,
            cu_seqlens,
            seqused,
            max_seqlen,
            valid,
            n,
            s,
            self.scale,
            self.half_window,
        )

    def _sdpa_band(self, q, k, v, valid):
        """Dense fallback (CPU / no flash). O(S^2) — tests only. half_window<0=global."""
        n, s = q.shape[:2]
        allowed = valid.unsqueeze(1) & valid.unsqueeze(2)
        if self.half_window >= 0:  # sliding window; <0 -> global (valid pairs only)
            rank = torch.cumsum(valid.to(torch.long), dim=1) - 1
            within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= self.half_window
            allowed = allowed & within
        allowed = allowed | torch.eye(s, dtype=torch.bool, device=q.device).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=allowed.unsqueeze(1),
            scale=self.scale,
        ).transpose(1, 2)
        return out * valid.unsqueeze(-1).unsqueeze(-1)


class RoPESWAAF3Block(nn.Module):
    """AF3 atom block: RoPE-SWA attention + AF3 ConditionedTransition FFN."""

    def __init__(
        self,
        d_atom: int,
        d_cond: int,
        n_head: int,
        *,
        half_window: int = 64,
        transition_n: int = 2,
        magnitude_preserving: bool = False,
        use_rotation: bool = False,
        mp_full: bool = False,
        mp_residual: bool = False,
        residual_t: float = 0.3,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        self.mp_residual = mp_residual or mp_full
        if magnitude_preserving:
            raise ValueError(
                "The installed engine does not support legacy MP ConditionedTransition"
            )
        self.residual_t = residual_t
        self.attention = RoPESWAAttentionAF3(
            d_atom,
            d_cond,
            n_head,
            half_window=half_window,
            magnitude_preserving=magnitude_preserving,
            use_rotation=use_rotation,
            mp_full=mp_full,
            implementation=implementation,
        )
        self.transition = ConditionedTransition(
            d_atom,
            d_cond,
            n=transition_n,
            implementation=to_engine_impl(implementation),
        )

    def forward(
        self,
        q: Float[torch.Tensor, "N S d_atom"],
        c: Float[torch.Tensor, "N S d_cond"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d_atom"]:
        attn = self.attention(q, c, attention_params)
        q = mp_sum(q, attn, self.residual_t) if self.mp_residual else q + attn
        # the engine transition returns q + f(q); mp_sum mixes the update itself
        return (
            mp_sum(q, self.transition.delta(q, c), self.residual_t)
            if self.mp_residual
            else self.transition(q, c)
        )


class RoPESWAAF3Transformer(nn.Module):
    """Stack of :class:`RoPESWAAF3Block` — drop-in for ``SWAAtomTransformer``.

    Reuses ``SWAAtomTransformer.Config`` (same fields). ``expansion_ratio`` maps
    to the AF3 ``ConditionedTransition`` expansion ``n``.
    """

    Config = SWAAtomTransformer.Config

    def __init__(self, config: SWAAtomTransformer.Config) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.d_atom // config.n_head
        self.blocks = nn.ModuleList(
            [
                RoPESWAAF3Block(
                    config.d_atom,
                    config.d_cond,
                    config.n_head,
                    # global_attn -> half_window=-1 (full self-attention, no window)
                    half_window=(
                        -1
                        if getattr(config, "global_attn", False)
                        else config.swa_window_size // 2
                    ),
                    transition_n=config.expansion_ratio,
                    magnitude_preserving=config.magnitude_preserving,
                    use_rotation=config.use_rotation,
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
        return build_3d_rope(
            ref_pos,
            ref_space_uid,
            self.head_dim,
            n_spatial_per_axis=self.config.n_spatial_rope_pairs_per_axis,
            n_uid_pairs=self.config.n_uid_rope_pairs,
            spatial_base_freq=self.config.spatial_rope_base_frequency,
            uid_base_freq=self.config.uid_rope_base_frequency,
        )

    @typecheck
    def forward(
        self,
        q: Float[torch.Tensor, "N S d_atom"],
        c: Float[torch.Tensor, "N S d_cond"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d_atom"]:
        for block in self.blocks:
            q = block(q, c, attention_params)
        return q

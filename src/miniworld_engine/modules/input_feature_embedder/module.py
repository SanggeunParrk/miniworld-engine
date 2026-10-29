"""Input feature embedder, ESMFold2 style: the atom encoder that turns reference-conformer atom features into one vector per token.

Per atom: the six reference features (position, mask, element, arcsinh(charge)) -> a ``d_atom`` single -> an ESMFold2
``SWAAtomTransformer`` (adaLN-Zero blocks, sliding-window 3D-RoPE self-attention, SwiGLU FFN; the single conditions itself) -> an
affine-free terminal RMSNorm -> ``Linear + ReLU`` to the token width -> the mean over each token's atoms.

This module is a composition, not a new kernel: every stage is an engine part that already exists. The atom transformer runs the
SWA atom DiT's fused block (H100: the native sm_90a stages with the window kernel, or FlashAttention-4 for global attention; B200: the
sm_100a stages) wherever its contract holds -- d_atom 128, 4 heads, SwiGLU hidden 256 -- and its per-op blocks otherwise; the
projections are cuBLAS and the token mean one scatter-add. On H100 / B200 nothing in it falls back to Triton (docs/gpus/h100/dispatch.md).

Parameter names match MiniWorld's ``ESMFold2InputAtomAttentionEncoder`` (``to_atom_single_cond``, ``atom_transformer``,
``atom_out_norm``, ``atom_single_rep_to_token_single``), so its checkpoints load unchanged. Unlike that encoder this one never builds the
O(L_atom^2) atom-pair features the ESMFold2 path discards.

``features="esmfold2"`` is ESMFold2's own encoder (biohub/esm ``EsmFold2AtomEncoder``) without the atom-name characters: per atom
position, charge, atom mask and the element one-hot (atomic number, 128 classes) -> ``Linear`` -> an affine ``LayerNorm`` (``atom_norm``)
that both starts the atom stream and conditions every block -> the transformer -> ``Linear + ReLU`` with no terminal norm -> the token
mean. The default ``"legacy"`` layout above is the one MiniWorld trained through v2.3.0: its scalar element and un-normalised
condition let the residual stream grow without bound and collapse onto one direction, after which the bias-free ReLU dies (2026-10-22).

``token_pool="linear"`` replaces the terminal ``Linear + ReLU`` with the LLM / ViT read-out: an affine ``LayerNorm`` (``atom_out_norm``)
-> a bias-free ``Linear`` -> the token mean, no activation. The bias-free ReLU sat on a stream whose atoms share one direction, so
most units went silent for every atom at once and never recovered (ESMFold2 layout, v2.3.1: 30 of 384 alive, 0.2% active).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn

from miniworld_engine.architecture.exceptions import ImplementationType

#: Reference features per atom: position (3), mask, element, arcsinh(charge).
N_ATOM_FEATURES = 6
#: Element one-hot width of the ``"esmfold2"`` layout (atomic numbers 0..127, ESMFold2's ``MAX_ATOMIC_NUMBER``).
MAX_ATOMIC_NUMBER = 128
#: ``"esmfold2"`` features per atom: position (3), charge, atom mask, element one-hot.
N_ATOM_FEATURES_ESMFOLD2 = 3 + 1 + 1 + MAX_ATOMIC_NUMBER

AtomFeatures = Literal["legacy", "esmfold2"]
TokenPool = Literal["relu", "linear"]


def atom_single_features(
    pos: Float[torch.Tensor, "B N 3"],
    mask: Float[torch.Tensor, "B N"],
    element: Float[torch.Tensor, "B N"],
    charge: Float[torch.Tensor, "B N"],
) -> Float[torch.Tensor, "B N 6"]:
    """The six per-atom reference features (AF3 Alg. 5 inputs without the atom-pair part), zero on masked atoms."""
    mask = mask.to(pos.dtype)
    features = torch.cat(
        [pos, mask[..., None], element.to(pos.dtype)[..., None], torch.arcsinh(charge.to(pos.dtype))[..., None]], dim=-1
    )
    return features * mask[..., None]


def atom_single_features_esmfold2(
    pos: Float[torch.Tensor, "B N 3"],
    charge: Float[torch.Tensor, "B N"],
    atom_mask: Bool[torch.Tensor, "B N"],
    element: Float[torch.Tensor, "B N"],
) -> Float[torch.Tensor, "B N 133"]:
    """ESMFold2's per-atom features without the atom-name characters: position, charge, atom mask, element one-hot (atomic number),
    zero on masked atoms (ESMFold2 zeroes the one-hots there; positions and charges of padding are zero already)."""
    mask = atom_mask.to(pos.dtype)
    element = F.one_hot(element.long().clamp(0, MAX_ATOMIC_NUMBER - 1), MAX_ATOMIC_NUMBER).to(pos.dtype)
    features = torch.cat([pos, charge.to(pos.dtype)[..., None], mask[..., None], element], dim=-1)
    return features * mask[..., None]


class InputFeatureEmbedder(nn.Module):
    """Atom features -> per-token single (ESMFold2-style input atom encoder).

    ``forward(pos, ref_mask, element, charge, space_uid, atom_mask, atom_to_token, n_token)`` with ``pos`` [B, N, 3], the other atom
    tensors [B, N] and ``atom_to_token`` [B, N] the token index of each atom; returns [B, n_token, d_token].
    """

    def __init__(
        self,
        d_atom: int,
        d_token: int,
        n_head: int,
        n_block: int,
        *,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        spatial_rope_base_frequency: float = 20.0,
        uid_rope_base_frequency: float = 10000.0,
        features: AtomFeatures = "legacy",
        token_pool: TokenPool = "relu",
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        # Imported here: the architecture package imports `miniworld_engine.modules`, which imports this file.
        from miniworld_engine.architecture.blocks._engine_impl import to_engine_impl
        from miniworld_engine.architecture.blocks.swa_atom_transformer import SWAAtomTransformer
        from miniworld_engine.architecture.primitives import LayerNorm, Linear
        from miniworld_engine.modules.primitives import RMSNorm

        if features not in ("legacy", "esmfold2"):
            message = f"features must be 'legacy' or 'esmfold2', got {features!r}"
            raise ValueError(message)
        if token_pool not in ("relu", "linear"):
            message = f"token_pool must be 'relu' or 'linear', got {token_pool!r}"
            raise ValueError(message)
        self.features = features
        self.token_pool = token_pool
        self.d_token = d_token
        n_in = N_ATOM_FEATURES_ESMFOLD2 if features == "esmfold2" else N_ATOM_FEATURES
        self.to_atom_single_cond = Linear(n_in, d_atom, init="default", bias=False)
        if features == "esmfold2":
            # ESMFold2's ``atom_norm``: the projected features are layer-normed once; the result starts the atom stream and is the
            # condition of every block, so the adaLN modulation sees a bounded input. The engine LayerNorm (H100 / B200: the general
            # CUDA LayerNorm) rather than nn.LayerNorm's PyTorch kernels; same parameter names, so checkpoints load unchanged
            self.atom_norm = LayerNorm(d_atom, implementation=implementation)
        self.atom_transformer = SWAAtomTransformer(
            SWAAtomTransformer.Config(
                d_atom=d_atom,
                d_cond=d_atom,
                implementation=implementation,
                n_block=n_block,
                n_head=n_head,
                swa_window_size=swa_window_size,
                expansion_ratio=expansion_ratio,
                n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
                n_uid_rope_pairs=n_uid_rope_pairs,
                spatial_rope_base_frequency=spatial_rope_base_frequency,
                uid_rope_base_frequency=uid_rope_base_frequency,
                block_style="esmfold2",
                # the engine's fused block (H100: the native sm_90a stages, FlashAttention-4 for global attention; B200: the sm_100a
                # stages) wherever its contract holds -- d_atom 128, 4 heads, SwiGLU hidden 256 -- instead of the per-op blocks
                fused_triton=True,
            )
        )
        # Terminal norm of the pre-norm stack (GPT-2 / LLaMA ``ln_f``): affine-free, no parameters. ESMFold2 has none; the linear
        # read-out takes an affine LayerNorm, as an LLM's final norm before its head.
        if token_pool == "linear":
            self.atom_out_norm = LayerNorm(d_atom, implementation=implementation)
        elif features == "esmfold2":
            self.atom_out_norm = nn.Identity()
        else:
            self.atom_out_norm = RMSNorm(d_atom, elementwise_affine=False, implementation=to_engine_impl(implementation))
        self.atom_single_rep_to_token_single = nn.Sequential(
            Linear(d_atom, d_token, init="default", bias=False),
            *([nn.ReLU()] if token_pool == "relu" else []),
        )

    def atom_single(
        self,
        pos: Float[torch.Tensor, "B N 3"],
        ref_mask: Float[torch.Tensor, "B N"],
        element: Float[torch.Tensor, "B N"],
        charge: Float[torch.Tensor, "B N"],
        space_uid: Int[torch.Tensor, "B N"],
        atom_mask: Bool[torch.Tensor, "B N"],
    ) -> Float[torch.Tensor, "B N d_atom"]:
        """The atom stream after the transformer and the terminal norm (none for ``"esmfold2"``)."""
        from miniworld_engine.modules.swa_atom_attention.module import build_attention_params

        if self.features == "esmfold2":
            projected = self.to_atom_single_cond(atom_single_features_esmfold2(pos, charge, atom_mask, element))
            # the LayerNorm runs in fp32 under autocast; hand the transformer the projection's dtype, as the legacy layout does
            single = self.atom_norm(projected).to(projected.dtype)
        else:
            single = self.to_atom_single_cond(atom_single_features(pos, ref_mask, element, charge))
        cos, sin = self.atom_transformer.build_rope(pos, space_uid)
        params = build_attention_params(cos, sin, atom_mask, num_aug=1)
        single = self.atom_transformer(single, single, params)   # the single conditions itself
        return self.atom_out_norm(single).to(single.dtype)   # the LayerNorm of token_pool="linear" runs in fp32 under autocast

    def to_tokens(
        self,
        atom_single: Float[torch.Tensor, "B N d_atom"],
        atom_mask: Bool[torch.Tensor, "B N"],
        atom_to_token: Int[torch.Tensor, "B N"],
        n_token: int,
    ) -> Float[torch.Tensor, "B T d_token"]:
        """``Linear + ReLU`` (``token_pool="linear"``: ``Linear``) per atom, then the mean over each token's atoms: one scatter-add (its
        backward a gather), summed in fp32."""
        per_atom = self.atom_single_rep_to_token_single(atom_single * atom_mask[..., None])
        idx = atom_to_token.long()
        src = per_atom.float()
        total = src.new_zeros(idx.shape[0], n_token, src.shape[-1]).scatter_add(1, idx[..., None].expand_as(src), src)
        count = src.new_zeros(idx.shape[0], n_token).scatter_add(1, idx, atom_mask.to(src.dtype))
        return (total / count[..., None].clamp(min=1.0)).to(per_atom.dtype)

    def forward(
        self,
        pos: Float[torch.Tensor, "B N 3"],
        ref_mask: Float[torch.Tensor, "B N"],
        element: Float[torch.Tensor, "B N"],
        charge: Float[torch.Tensor, "B N"],
        space_uid: Int[torch.Tensor, "B N"],
        atom_mask: Bool[torch.Tensor, "B N"],
        atom_to_token: Int[torch.Tensor, "B N"],
        n_token: int,
    ) -> Float[torch.Tensor, "B T d_token"]:
        single = self.atom_single(pos, ref_mask, element, charge, space_uid, atom_mask)
        return self.to_tokens(single, atom_mask, atom_to_token, n_token)

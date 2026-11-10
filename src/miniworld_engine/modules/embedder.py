"""Input feature embedder, ESMFold2 style: the atom encoder that turns reference-conformer atom features into one vector per token.

Per atom: the six reference features (position, mask, element, arcsinh(charge)) -> a ``d_atom`` single -> an ESMFold2
``SWAAtomTransformer`` (adaLN-Zero blocks, sliding-window 3D-RoPE self-attention, SwiGLU FFN; the single conditions itself) -> an
affine-free terminal RMSNorm -> ``Linear + ReLU`` to the token width -> the mean over each token's atoms.

This module is a composition: the reference features, a projection, the ``SWAAtomTransformer``, a terminal norm, the token read-out and
one scatter-add for the token mean.

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

A :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import nn

from miniworld_engine.modules.atom_dit import SWAAtomTransformer
from miniworld_engine.modules.functional import build_attention_params
from miniworld_engine.modules.norm import LayerNorm, RMSNorm
from miniworld_engine.modules.primitives import BackendModule, Linear, key_mask, random_input, wide

#: Reference features per atom: position (3), mask, element, arcsinh(charge).
N_ATOM_FEATURES = 6
#: Element one-hot width of the ``"esmfold2"`` layout (atomic numbers 0..127, ESMFold2's ``MAX_ATOMIC_NUMBER``).
MAX_ATOMIC_NUMBER = 128
#: ``"esmfold2"`` features per atom: position (3), charge, atom mask, element one-hot.
N_ATOM_FEATURES_ESMFOLD2 = 3 + 1 + 1 + MAX_ATOMIC_NUMBER

AtomFeatures = Literal["legacy", "esmfold2"]
TokenPool = Literal["relu", "linear"]


def atom_single_features(
    pos: torch.Tensor,
    mask: torch.Tensor,
    element: torch.Tensor,
    charge: torch.Tensor,
) -> torch.Tensor:
    """The six per-atom reference features (AF3 Alg. 5 inputs without the atom-pair part), zero on masked atoms."""
    mask = mask.to(pos.dtype)
    features = torch.cat(
        [pos, mask[..., None], element.to(pos.dtype)[..., None], torch.arcsinh(charge.to(pos.dtype))[..., None]], dim=-1
    )
    return features * mask[..., None]


def atom_single_features_esmfold2(
    pos: torch.Tensor,
    charge: torch.Tensor,
    atom_mask: torch.Tensor,
    element: torch.Tensor,
) -> torch.Tensor:
    """ESMFold2's per-atom features without the atom-name characters: position, charge, atom mask, element one-hot (atomic number),
    zero on masked atoms (ESMFold2 zeroes the one-hots there; positions and charges of padding are zero already)."""
    mask = atom_mask.to(pos.dtype)
    element = F.one_hot(element.long().clamp(0, MAX_ATOMIC_NUMBER - 1), MAX_ATOMIC_NUMBER).to(pos.dtype)
    features = torch.cat([pos, charge.to(pos.dtype)[..., None], mask[..., None], element], dim=-1)
    return features * mask[..., None]


class InputFeatureEmbedder(BackendModule):
    """Atom features -> per-token single (ESMFold2-style input atom encoder).

    ``forward(pos, ref_mask, element, charge, space_uid, atom_mask, atom_to_token, n_token)`` with ``pos`` [B, N, 3], the other atom
    tensors [B, N] and ``atom_to_token`` [B, N] the token index of each atom; returns [B, n_token, d_token].

    ``backend="anthropic"`` sets its ``atom_transformer``'s (the SWA attention on ESMFold2's release band core); the rest stays
    ``"pytorch"``'s (no release op has the features, the read-out or the token mean).

    ``backend="cuda"`` / ``"triton"`` (the engine's kernels) sets the same backend on its ``atom_transformer`` and on its norms
    (``atom_norm``, ``atom_out_norm``: the LayerNorm / RMSNorm kernels); the reference features, the two bias-free projections, the ReLU
    and the token mean (one scatter-add) are the composition's own statements, as in ``"pytorch"``. Each child states its own range and
    raises :class:`~miniworld_engine.modules.primitives.BackendUnsupported` outside it; this class refuses a backend its atom transformer
    does not have.
    """
    version = 1   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")
    # 2026-11-10: kits; sm_80 paths; H100 inference (cached weight forms, slim modulation, fp32 global on the fp32 stages, PDL)
    backend_versions: ClassVar[dict[str, int]] = {"anthropic": 2, "cuda:a100_sm80": 3, "cuda:h100_sm90": 2}
    #: The engine backends this composition hands to its children.
    _ENGINE_BACKENDS = ("cuda", "triton")

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
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
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
            # condition of every block, so the adaLN modulation sees a bounded input
            self.atom_norm = LayerNorm(d_atom)
        self.atom_transformer = SWAAtomTransformer(
            SWAAtomTransformer.Config(
                d_atom=d_atom,
                d_cond=d_atom,
                n_block=n_block,
                n_head=n_head,
                swa_window_size=swa_window_size,
                expansion_ratio=expansion_ratio,
                n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
                n_uid_rope_pairs=n_uid_rope_pairs,
                spatial_rope_base_frequency=spatial_rope_base_frequency,
                uid_rope_base_frequency=uid_rope_base_frequency,
                block_style="esmfold2",
            )
        )
        # Terminal norm of the pre-norm stack (GPT-2 / LLaMA ``ln_f``): affine-free, no parameters. ESMFold2 has none; the linear
        # read-out takes an affine LayerNorm, as an LLM's final norm before its head.
        if token_pool == "linear":
            self.atom_out_norm = LayerNorm(d_atom)
        elif features == "esmfold2":
            self.atom_out_norm = nn.Identity()
        else:
            self.atom_out_norm = RMSNorm(d_atom, elementwise_affine=False)
        self.atom_single_rep_to_token_single = nn.Sequential(
            Linear(d_atom, d_token, init="default", bias=False),
            *([nn.ReLU()] if token_pool == "relu" else []),
        )
        self.set_backend(backend)

    def _engine_children(self) -> list[BackendModule]:
        return [m for m in (getattr(self, "atom_norm", None), self.atom_out_norm) if isinstance(m, BackendModule)]

    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        if backend in self._ENGINE_BACKENDS:
            # a child without this backend stays as it is; _supports_<backend> refuses the call when the atom transformer is one of them
            if backend in self.atom_transformer.backends:
                self.atom_transformer.set_backend(backend)
            else:
                self.atom_transformer.set_backend("pytorch")
            for child in self._engine_children():
                child.set_backend(backend if backend in child.backends else "pytorch")
        else:
            self.atom_transformer.set_backend(backend)
            for child in self._engine_children():
                child.set_backend("pytorch")
        return self

    def _forward_anthropic(self, *args, **kwargs) -> torch.Tensor:
        return self._forward_pytorch(*args, **kwargs)   # the atom transformer runs its own backend

    def _supports_engine(self, backend: str, pos: torch.Tensor) -> str | None:
        from miniworld_engine.modules.norm import a100, h100_unsupported, triton_unsupported

        if backend not in self.atom_transformer.backends:
            return f"its atom transformer ({type(self.atom_transformer).__name__}) has no {backend!r} backend"
        missing = [type(c).__name__ for c in self._engine_children() if backend not in c.backends]
        if missing:
            return f"its {', '.join(missing)} has no {backend!r} backend"
        if backend == "cuda":   # the H100 or (the children's sm_80 builds) the A100; each child states its own range when it runs
            return None if a100(pos) else h100_unsupported(pos)
        return triton_unsupported(pos)

    def _supports_cuda(self, pos: torch.Tensor, *args, **kwargs) -> str | None:
        return self._supports_engine("cuda", pos)

    def _supports_triton(self, pos: torch.Tensor, *args, **kwargs) -> str | None:
        return self._supports_engine("triton", pos)

    def _forward_cuda(self, *args, **kwargs) -> torch.Tensor:
        return self._forward_pytorch(*args, **kwargs)   # the composition; the transformer and the norms run their "cuda" backends

    def _forward_triton(self, *args, **kwargs) -> torch.Tensor:
        return self._forward_pytorch(*args, **kwargs)   # the composition; the transformer and the norms run their "triton" backends

    def atom_single(
        self,
        pos: torch.Tensor,
        ref_mask: torch.Tensor,
        element: torch.Tensor,
        charge: torch.Tensor,
        space_uid: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The atom stream after the transformer and the terminal norm (none for ``"esmfold2"``)."""
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
        atom_single: torch.Tensor,
        atom_mask: torch.Tensor,
        atom_to_token: torch.Tensor,
        n_token: int,
    ) -> torch.Tensor:
        """``Linear + ReLU`` (``token_pool="linear"``: ``Linear``) per atom, then the mean over each token's atoms: one scatter-add (its
        backward a gather), summed in fp32."""
        per_atom = self.atom_single_rep_to_token_single(atom_single * atom_mask[..., None])
        idx = atom_to_token.long()
        src = per_atom.to(wide(per_atom))
        total = src.new_zeros(idx.shape[0], n_token, src.shape[-1]).scatter_add(1, idx[..., None].expand_as(src), src)
        count = src.new_zeros(idx.shape[0], n_token).scatter_add(1, idx, atom_mask.to(src.dtype))
        return (total / count[..., None].clamp(min=1.0)).to(per_atom.dtype)

    def _forward_pytorch(
        self,
        pos: torch.Tensor,
        ref_mask: torch.Tensor,
        element: torch.Tensor,
        charge: torch.Tensor,
        space_uid: torch.Tensor,
        atom_mask: torch.Tensor,
        atom_to_token: torch.Tensor,
        n_token: int,
    ) -> torch.Tensor:
        single = self.atom_single(pos, ref_mask, element, charge, space_uid, atom_mask)
        return self.to_tokens(single, atom_mask, atom_to_token, n_token)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``samples`` systems of ``length`` atoms (real atoms first), 8 atoms per token and per conformer, elements 1..19, charges -1..1."""
        mask = key_mask(samples, length, device=device)
        index = (torch.arange(length, device=device) // 8).expand(samples, length).contiguous()
        return {"pos": random_input((samples, length, 3), dtype=dtype, device=device, generator=generator, scale=5.0),
                "ref_mask": mask, "atom_mask": mask, "space_uid": index, "atom_to_token": index, "n_token": (length + 7) // 8,
                "element": torch.randint(1, 20, (samples, length), device=device, generator=generator),
                "charge": torch.randint(-1, 2, (samples, length), device=device, generator=generator)}

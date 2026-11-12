"""Stacks of diffusion-transformer blocks: ``n_blocks`` blocks of one model's ``TokenDiT*`` / ``AtomDiT*`` class run in order, and the
conversion of a released stack's shared pair LayerNorm / projection into the blocks' own (:func:`from_shared_pair`).

    stack = TokenDiTStack("protenix", 24, d_single=768, d_cond=384, d_pair=128, n_head=16)
    out = stack(single, cond, pair, mask)                       # every block in order (each owns its residuals)
    stack.load_shared_pair(norm_weight, projection)             # a released AF3-family checkpoint's shared pair norm / projection

The stacks add no math and no kernel: each block is the model's own class (state-dict keys ``blocks.{i}.<block names>``) and does what it
does alone. What the stack guarantees is the hand-over the blocks' caches rely on: every block gets the SAME ``pair`` / ``cond`` / ``mask``
/ ``uid`` tensor objects, so each block's hoisted pair bias and conditioning tables (keyed on those tensors, ``kernels._capture``) are made
once per fold and reused over the sampler's steps; ``extra_pair_bias`` / ``compute_dtype`` (token) and ``uid`` (Chai-1's atom blocks) are
forwarded to every block.

Layout (the blocks' own): ``single`` / ``cond`` ``[A, B, L, d]`` (A diffusion samples per fold, B folds), token ``pair`` ``[B, L, L, d_pair]``,
atom ``pair`` the trunked ``[B, nwin, 32, 128, d_pair]``, ``mask`` the bool key mask ``[B, L]``. ``B`` is the multi-fold axis (several
trunk_seeds of one input, each with its own pair and conditioning; the multi-fold contract in docs/modules.md).

A released stack (AlphaFold 3, Boltz-2, Chai-1) keeps ONE pair LayerNorm and one projection of ``n_blocks * n_head`` rows (or one per super
block of blocks, whose row-concatenation is that projection) for the whole stack; :func:`from_shared_pair` gives each block its copy of
the norm weight and its ``n_head`` rows. The LayerNorm offset is dropped: it adds ``W_proj beta``, one constant per head for every (query,
key), which the softmax over the keys cancels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from miniworld_engine.modules.atom_dit import ATOM_DIT_MODELS, _LocalAtomDiT
from miniworld_engine.modules.token_dit import TOKEN_DIT_MODELS


def from_shared_pair(
    norm_weight: torch.Tensor,
    projection: torch.Tensor | Sequence[torch.Tensor],
    *,
    n_blocks: int,
    n_head: int,
    norm_bias: torch.Tensor | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Per-block ``{"attention.ln_pair.weight", "attention.to_bias.weight"}`` from a released stack's shared pair LayerNorm weight
    ``[d_pair]`` and projection: one ``[n_blocks * n_head, d_pair]`` Linear weight (rows ``n_head b .. n_head b + n_head - 1`` are block
    ``b``'s heads), or a sequence of super-block projections whose row-concatenation is that (AF3's token DiT: one per four blocks).
    ``norm_bias`` (the LayerNorm offset) is accepted and dropped: a per-head constant over all keys, cancelled by the softmax."""
    del norm_bias  # cancelled by the softmax (module docstring)
    proj = projection if isinstance(projection, torch.Tensor) else torch.cat(list(projection), 0)
    if norm_weight.ndim != 1:
        raise ValueError(f"norm_weight must be [d_pair], got {tuple(norm_weight.shape)}")
    if proj.ndim != 2 or proj.shape != (n_blocks * n_head, norm_weight.shape[0]):
        raise ValueError(
            f"projection must be [n_blocks * n_head, d_pair] = [{n_blocks * n_head}, {norm_weight.shape[0]}], got {tuple(proj.shape)}"
        )
    return [
        {
            "attention.ln_pair.weight": norm_weight.detach().clone(),
            "attention.to_bias.weight": proj[b * n_head : (b + 1) * n_head].detach().clone(),
        }
        for b in range(n_blocks)
    ]


class _DiTStack(nn.Module):
    """``blocks``: the blocks, run in order; ``set_backend`` sets every block's backend."""

    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("a stack needs at least one block")
        self.blocks = nn.ModuleList(blocks)

    def set_backend(self, backend: str = "pytorch"):
        for block in self.blocks:
            block.set_backend(backend)  # pyright: ignore[reportCallIssue]  -- every block is a BackendModule
        return self

    @property
    def backend(self) -> str:
        return self.blocks[0].backend  # pyright: ignore[reportReturnType]

    def _n_head(self) -> int:
        return int(self.blocks[0].attention.n_head)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

    @torch.no_grad()
    def load_shared_pair(
        self, norm_weight: torch.Tensor, projection: torch.Tensor | Sequence[torch.Tensor], norm_bias: torch.Tensor | None = None
    ):
        """Copy a released stack's shared pair LayerNorm weight / projection into the blocks (:func:`from_shared_pair`); returns ``self``."""
        per_block = from_shared_pair(norm_weight, projection, n_blocks=len(self.blocks), n_head=self._n_head(), norm_bias=norm_bias)
        for block, state in zip(self.blocks, per_block, strict=True):
            for name, value in state.items():
                block.get_parameter(name).copy_(value)
        return self


class TokenDiTStack(_DiTStack):
    """``n_blocks`` ``TokenDiT(model, ...)`` blocks run in order.

    ``forward(single, cond, pair, mask=None, *, extra_pair_bias=None, compute_dtype=None)``: ``single`` / ``cond`` ``[A, B, L, d]``, ``pair``
    ``[B, L, L, d_pair]``, ``mask`` ``[B, L]`` (or ``[A, B, L]``); the options are passed to every block (a block refuses one it has
    not). Every block reads the same ``pair`` / ``cond`` / ``mask`` objects (its pair bias is hoisted once per pair)."""

    def __init__(
        self,
        model: str,
        n_blocks: int,
        d_single: int = 768,
        d_cond: int = 384,
        d_pair: int = 128,
        n_head: int = 16,
        n: int = 2,
        *,
        backend: str = "pytorch",
        **block_kwargs: Any,
    ) -> None:
        if model not in TOKEN_DIT_MODELS:
            raise ValueError(f"TokenDiTStack: unknown model {model!r}; known: {', '.join(sorted(TOKEN_DIT_MODELS))}")
        klass = TOKEN_DIT_MODELS[model]
        super().__init__([klass(d_single, d_cond, d_pair, n_head, n, backend=backend, **block_kwargs) for _ in range(n_blocks)])
        self.model = model

    def forward(
        self,
        single: torch.Tensor,
        cond: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        extra_pair_bias: torch.Tensor | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        kw: dict[str, Any] = {}
        if extra_pair_bias is not None:
            kw["extra_pair_bias"] = extra_pair_bias
        if compute_dtype is not None:
            kw["compute_dtype"] = compute_dtype
        for block in self.blocks:
            single = block(single, cond, pair, mask, **kw)
        return single


class AtomDiTStack(_DiTStack):
    """``n_blocks`` ``AtomDiT(model, ...)`` blocks with the trunked atom pair, run in order (the pair-free ESMFold2 / MiniWorld atom blocks
    have their own stack, :class:`~miniworld_engine.modules.atom_dit.SWAAtomTransformer`).

    ``forward(single, cond, pair, mask=None, uid=None)``: ``single`` / ``cond`` ``[A, B, N, 128]``, ``pair`` the trunked
    ``[B, nwin, 32, 128, d_pair]``, ``mask`` the bool key mask ``[B, N]`` (real atoms first), ``uid`` ``[B, N]`` the conformer ids of
    Chai-1's blocks (passed only when given). Every block reads the same ``pair`` / ``cond`` / ``mask`` / ``uid`` objects (each block
    hoists its windowed pair bias once per pair). ``key_window`` and the model options (``adaln_eps``) go to every block."""

    def __init__(
        self,
        model: str,
        n_blocks: int,
        d_single: int = 128,
        d_cond: int = 128,
        d_pair: int = 16,
        n_head: int = 4,
        n: int = 2,
        *,
        backend: str = "pytorch",
        **block_kwargs: Any,
    ) -> None:
        klass = ATOM_DIT_MODELS.get(model)
        if klass is None:
            raise ValueError(f"AtomDiTStack: unknown model {model!r}; known: {', '.join(sorted(ATOM_DIT_MODELS))}")
        if not issubclass(klass, _LocalAtomDiT):
            raise ValueError(f"AtomDiTStack: {model!r} has no atom pair ({klass.__name__}); its stack is SWAAtomTransformer")
        super().__init__([klass(d_single, d_cond, d_pair, n_head, n, backend=backend, **block_kwargs) for _ in range(n_blocks)])
        self.model = model

    def forward(
        self,
        single: torch.Tensor,
        cond: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        uid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            single = block(single, cond, pair, mask) if uid is None else block(single, cond, pair, mask, uid=uid)
        return single


__all__ = ["AtomDiTStack", "TokenDiTStack", "from_shared_pair"]

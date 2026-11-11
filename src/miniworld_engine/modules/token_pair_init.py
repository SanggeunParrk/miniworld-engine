"""The token-pair initialisation of the input feature embedder (AlphaFold 3 SI 2.8): ``TokenPairInit``.

A :class:`~miniworld_engine.modules.primitives.BackendModule` that owns the two bias-free Linears (``w_rel``, ``w_bond``); ``_forward_pytorch`` (the dense
reference below: a 139-wide fp32 one-hot per pair, a Linear over it, the outer sum, the bond one-hot Linear) is the source of truth. The kernels
(``kernels/token_pair_init``) never build the one-hot: ``one-hot @ W`` is a gather of up to four table rows, and the backward is one pass over ``dz``
for the gradients of ``left``, ``right`` and both weights.
"""

import math
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniworld_engine.modules.pair_embed import _cuda_arch, _first
from miniworld_engine.modules.primitives import BackendModule, random_input, wide

D_PAIR = 128


def n_rel(r_max: int, s_max: int) -> int:
    """Width of the relative-position one-hot: 2 (2 r_max + 2) + (2 s_max + 2) + 1 (139 for r_max 32, s_max 2)."""
    return 2 * (2 * r_max + 2) + (2 * s_max + 2) + 1


def relative_position_features(
    asym_id, residue_idx, token_idx, entity_id, sym_id, r_max: int, s_max: int, dtype: torch.dtype
) -> torch.Tensor:
    """``[B, L, L, n_rel]`` one-hot features of AlphaFold 3 SI 2.8 (same chain / residue / entity, the clamped residue, token and chain offsets)."""
    same_chain = asym_id[:, :, None] == asym_id[:, None, :]
    same_residue = residue_idx[:, :, None] == residue_idx[:, None, :]
    same_entity = entity_id[:, :, None] == entity_id[:, None, :]
    d_residue = torch.clamp(residue_idx[:, :, None] - residue_idx[:, None, :] + r_max, 0, 2 * r_max) * same_chain
    d_residue = d_residue + ~same_chain * (2 * r_max + 1)
    d_token = torch.clamp(token_idx[:, :, None] - token_idx[:, None, :] + r_max, 0, 2 * r_max) * (same_chain & same_residue)
    d_token = d_token + ~(same_chain * same_residue) * (2 * r_max + 1)
    d_chain = torch.clamp(sym_id[:, :, None] - sym_id[:, None, :] + s_max, 0, 2 * s_max) * same_entity
    d_chain = d_chain + ~same_entity * (2 * s_max + 1)
    feats = torch.cat(
        [
            F.one_hot(d_residue.long(), 2 * r_max + 2),
            F.one_hot(d_token.long(), 2 * r_max + 2),
            F.one_hot(d_chain.long(), 2 * s_max + 2),
            same_entity.unsqueeze(-1),
        ],
        dim=-1,
    )
    return feats.to(dtype)


class TokenPairInit(BackendModule):
    """The pair stream before the trunk::

        z[b, i, j] = left[b, i] + right[b, j] + W_rel one_hot_rel(i, j) + W_bond one_hot(bond[b, i, j], 2)

    ``one_hot_rel`` = AlphaFold 3's relative-position encoding (``r_max`` residue / token offsets, ``s_max`` chain offsets, same-entity flag; 139 channels for
    r_max 32, s_max 2).

    ``forward(left, right, asym_id, residue_idx, token_idx, entity_id, sym_id, bond)``:

    * ``left`` / ``right`` ``[B, L, 128]`` -- the two single-representation projections of the tokens;
    * the five id tensors ``[B, L]`` (any integer dtype) -- chain, residue, token, entity and symmetry indices (data: no gradient);
    * ``bond`` ``[B, L, L]`` -- the token bond adjacency, 0 / 1 (bool, integer or float; any non-zero is a bond).

    Returns ``z`` ``[B, L, L, 128]`` in at least fp32 (fp32 for fp32 / bf16 inputs, fp64 for fp64), differentiable in ``left``, ``right``, ``w_rel``
    ``[128, n_rel]`` and ``w_bond`` ``[128, 2]``. The kernels compute in exact fp32 arithmetic (the reference's TF32 Linears round the weights; they do not).

    Parameters
    ----------
    backend : str
        See :class:`BackendModule`. ``"cuda"`` (H100 sm_90a; A100 sm_80: the same source's sm_80 build): the engine's hand-written CUDA kernels (``kernels/token_pair_init/cuda``; a bitwise
        deterministic backward, a one-launch glue for the ids and the table on forward-only calls). ``"triton"``: the portable Triton kernels
        (``kernels/token_pair_init/triton``), any CUDA card. Both read ``left`` / ``right`` / the weights in fp32 or bf16 and pack each relative-position bin
        into 8 bits (``n_rel - 2 <= 255``).
    """

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results
    backend_versions: ClassVar[dict[str, int]] = {"cuda:a100_sm80": 2}  # 2026-11-10: the sm_80 build of the H100 kernels

    backends = ("pytorch", "cuda", "triton")

    def __init__(self, *, r_max: int = 32, s_max: int = 2, backend: str = "pytorch") -> None:
        super().__init__()
        self.r_max, self.s_max = r_max, s_max
        self.w_rel = nn.Parameter(torch.randn(D_PAIR, n_rel(r_max, s_max)) / math.sqrt(n_rel(r_max, s_max)))
        self.w_bond = nn.Parameter(torch.randn(D_PAIR, 2) / math.sqrt(2))
        self.set_backend(backend)

    def _forward_pytorch(self, left, right, asym_id, residue_idx, token_idx, entity_id, sym_id, bond):
        dt = wide(left, right, self.w_rel, self.w_bond)
        z = left.to(dt)[:, :, None, :] + right.to(dt)[:, None, :, :]
        z = (
            z
            + relative_position_features(asym_id, residue_idx, token_idx, entity_id, sym_id, self.r_max, self.s_max, dt)
            @ self.w_rel.to(dt).t()
        )
        return z + F.one_hot((bond != 0).long(), 2).to(dt) @ self.w_bond.to(dt).t()

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``samples`` token sets of ``length`` tokens: Gaussian ``left`` / ``right`` in ``dtype``, random chain (3) / residue (90) / token (60) / entity (2) /
        symmetry (6) indices, 5 % of the token pairs bonded."""

        def ids(hi, id_dtype=torch.int64):
            return torch.randint(0, hi, (samples, length), device=device, generator=generator).to(id_dtype)

        left = random_input((samples, length, D_PAIR), dtype=dtype, device=device, generator=generator)
        right = random_input((samples, length, D_PAIR), dtype=dtype, device=device, generator=generator)
        bond = torch.rand((samples, length, length), device=device, generator=generator) < 0.05
        return {
            "left": left,
            "right": right,
            "asym_id": ids(3),
            "residue_idx": ids(90),
            "token_idx": ids(60),
            "entity_id": ids(2),
            "sym_id": ids(6, torch.int32),
            "bond": bond,
        }

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_kernels(self, left, right, asym_id, residue_idx, token_idx, entity_id, sym_id, bond) -> str | None:
        if left.device.type != "cuda":
            return "the kernels run on CUDA tensors only"
        if any(t.dtype not in (torch.float32, torch.bfloat16) for t in (left, right, self.w_rel, self.w_bond)):
            return "left, right and both weights must be fp32 or bf16 (the kernels compute in fp32)"
        if left.dim() != 3 or left.shape != right.shape or left.shape[-1] != D_PAIR:
            return f"left / right must be [B, L, {D_PAIR}], got {tuple(left.shape)} / {tuple(right.shape)}"
        if n_rel(self.r_max, self.s_max) - 2 > 255:
            return f"the kernels pack each relative-position bin into 8 bits; r_max={self.r_max}, s_max={self.s_max} needs more"
        ids = (asym_id, residue_idx, token_idx, entity_id, sym_id)
        if any(t.shape != left.shape[:2] or t.dtype.is_floating_point or t.dtype == torch.bool for t in ids):
            return f"the five id tensors must be integer [B, L] = {tuple(left.shape[:2])}"
        if tuple(bond.shape) != (left.shape[0], left.shape[1], left.shape[1]):
            return f"bond must be [B, L, L], got {tuple(bond.shape)}"
        if left.numel() == 0:
            return "empty input"
        return None

    def _supports_cuda(self, left, right, asym_id, residue_idx, token_idx, entity_id, sym_id, bond) -> str | None:
        return _first(
            self._supports_kernels(left, right, asym_id, residue_idx, token_idx, entity_id, sym_id, bond),
            _cuda_arch(left.device) if left.device.type == "cuda" else None,
        )

    def _supports_triton(self, left, right, asym_id, residue_idx, token_idx, entity_id, sym_id, bond) -> str | None:
        return self._supports_kernels(left, right, asym_id, residue_idx, token_idx, entity_id, sym_id, bond)

    @staticmethod
    def _bond8(bond: torch.Tensor) -> torch.Tensor:
        """The bond adjacency as 0 / 1 bytes, flat: a bool tensor already is (no pass over it); any other dtype: a non-zero is a bond."""
        return bond.contiguous().view(torch.uint8).reshape(-1) if bond.dtype is torch.bool else (bond != 0).to(torch.uint8).reshape(-1)

    def _ids_table(self, asym_id, residue_idx, token_idx, entity_id, sym_id, *, one_launch: bool):
        """``(ids [5, B * L] int32, table [n_rel + 2, 128] = [W_rel^T ; W_bond^T])``. ``one_launch``: both in one kernel (nothing differentiates the
        weights, they are contiguous fp32 and the ids contiguous int32 / int64); else torch ops, differentiable in the weights."""
        b, length = asym_id.shape
        id5 = [t.reshape(-1) for t in (asym_id, residue_idx, token_idx, entity_id, sym_id)]
        if one_launch and (
            self.w_rel.dtype == torch.float32
            and self.w_bond.dtype == torch.float32
            and self.w_rel.is_contiguous()
            and self.w_bond.is_contiguous()
            and not (torch.is_grad_enabled() and (self.w_rel.requires_grad or self.w_bond.requires_grad))
            and all(t.dtype in (torch.int32, torch.int64) and t.is_contiguous() for t in id5)
        ):
            from miniworld_engine.kernels.token_pair_init.cuda.function import token_pair_init_prep

            ids, tbl = token_pair_init_prep(*id5, self.w_rel, self.w_bond)  # pyright: ignore[reportCallIssue]  -- a torch.library custom op (CustomOpDef is callable)
            return ids, tbl
        ids = torch.stack([asym_id, residue_idx, token_idx, entity_id, sym_id]).to(torch.int32).reshape(5, b * length)
        return ids, torch.cat([self.w_rel.t(), self.w_bond.t()], 0).contiguous()

    def _forward_cuda(self, left, right, asym_id, residue_idx, token_idx, entity_id, sym_id, bond):
        from miniworld_engine.kernels.token_pair_init.cuda.function import token_pair_init_cuda

        ids, tbl = self._ids_table(asym_id, residue_idx, token_idx, entity_id, sym_id, one_launch=True)
        return token_pair_init_cuda(left.contiguous(), right.contiguous(), tbl, ids, self._bond8(bond), self.r_max, self.s_max)

    def _forward_triton(self, left, right, asym_id, residue_idx, token_idx, entity_id, sym_id, bond):
        from miniworld_engine.kernels.token_pair_init.triton.function import token_pair_init_triton

        ids, tbl = self._ids_table(asym_id, residue_idx, token_idx, entity_id, sym_id, one_launch=False)
        return token_pair_init_triton(left.contiguous(), right.contiguous(), tbl, ids, self._bond8(bond), self.r_max, self.s_max)

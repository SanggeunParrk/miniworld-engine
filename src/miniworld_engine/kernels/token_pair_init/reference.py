"""PyTorch reference of the token-pair initialisation: what "correct" means for kernel family ``token_pair_init``.

The input feature embedder's pair stream before the trunk,

    z[b,i,j] = left[b,i] + right[b,j] + Linear_rel(relative_position_one_hot(i, j)) + Linear_bond(one_hot(bond[b,i,j], 2))

with the relative-position features exactly as AlphaFold 3 SI 2.8 / ``team_gm.modules.layers.RelativePositionEmbedding``
builds them (a 139-wide fp32 one-hot per pair for r_max 32, s_max 2), written as the dense tensor ops the model runs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_position_features(asym_id, residue_idx, token_idx, entity_id, sym_id, r_max: int = 32, s_max: int = 2) -> torch.Tensor:
    """[B, L, L, 2 (2 r_max + 2) + (2 s_max + 2) + 1] fp32 one-hot features."""
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
        [F.one_hot(d_residue.long(), 2 * r_max + 2), F.one_hot(d_token.long(), 2 * r_max + 2),
         F.one_hot(d_chain.long(), 2 * s_max + 2), same_entity.unsqueeze(-1)], dim=-1)
    return feats.float()


def token_pair_init_reference(left, right, w_rel, w_bond, asym_id, residue_idx, token_idx, entity_id, sym_id, bond,
                              r_max: int = 32, s_max: int = 2) -> torch.Tensor:
    """left / right [B, L, P]; w_rel [P, n_rel]; w_bond [P, 2]; the five id tensors [B, L]; bond [B, L, L] (0/1)."""
    z = left[:, :, None, :] + right[:, None, :, :]
    feats = relative_position_features(asym_id, residue_idx, token_idx, entity_id, sym_id, r_max, s_max)
    z = z + feats @ w_rel.t()
    return z + F.one_hot(bond.long(), 2).to(z.dtype) @ w_bond.t()

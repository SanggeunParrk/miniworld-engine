"""The torch reference the RoPE kernel is checked against -- the eager `apply_rotary_emb_3d`."""
from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_3d_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``apply_rotary_emb_3d`` written out: rotate the leading 2*half channels, pass the tail.

    Matches ``modules/swa_atom_attention/apply_rotary_emb_3d`` term for term, in fp32, so a
    disagreement points at the kernel and not at a differently-factored reference.
    """
    ro_dim = cos.shape[-1] * 2
    cos_r = cos.unsqueeze(2).repeat(1, 1, 1, 2).float()
    sin_r = sin.unsqueeze(2).repeat(1, 1, 1, 2).float()
    xr = x[..., :ro_dim].float()
    xr = xr * cos_r + _rotate_half(xr) * sin_r
    return torch.cat([xr.to(x.dtype), x[..., ro_dim:]], dim=-1)

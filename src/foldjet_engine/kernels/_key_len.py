"""The key mask as the attention cores read it to skip the dead key tiles of a padded tail.

A FoldForge call pads its token / atom count to a bucket: mask True for the first n live positions, False after. The keys past the
last attended one are masked in every query row; once a row has seen a live key its running max is a live logit, so their
exponentials are exact zeros and they add nothing to the sum or the output. A core that stops its key loop at the tile holding the
last attended key gives the same bits with fewer tiles. The cores read the mask bytes themselves (``augmented_attention/cuda/
key_len.cuh``: each warp takes the bound from the row): no host-side count, no extra launch, and a captured graph follows the mask
as it is at replay (a static input refreshed in place). A row with no attended key keeps every tile (the references fill masked logits
with a finite minimum: such a row is a uniform average over all keys)."""

from __future__ import annotations

import torch


def kernel_mask(mask: torch.Tensor | None, length: int) -> torch.Tensor | None:
    """``mask`` ([..., length], bool) as the cores read it: a bool [rows, length] with contiguous rows (a view of the caller's tensor when
    it already is one, else a copy), or None without a mask."""
    if mask is None:
        return None
    m = mask.reshape(-1, length)
    if m.dtype is not torch.bool:
        m = m.to(torch.bool)
    if not m.is_contiguous():
        m = m.contiguous()
    return m

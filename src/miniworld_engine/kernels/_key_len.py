"""Per-sample live key counts of a key mask, for attention cores that bound their key loops.

A FoldForge call pads its token / atom count to a bucket: mask True for the first n live positions, False after. The keys past
the last attended one are masked in every query row; once a row has seen a live key its running max is a live logit, so their
exponentials are exact zeros and they add nothing to the sum or the output. A core that stops its key loop at the block holding
``key_len`` gives the same bits with fewer blocks.

A sample with no attended key keeps every block (``key_len`` = L): the references fill masked logits with a finite minimum, so such
a row is a uniform average over all keys, which a shortened loop would change.

Computed on the device (no host sync: CUDA-graph capture safe); the result is a small tensor a hoisted table may hold and a static
input refresh rewrites in place."""

from __future__ import annotations

import torch


def key_len(mask: torch.Tensor | None, length: int) -> torch.Tensor | None:
    """int32 [G]: 1 + the last attended key of each row of ``mask`` ([..., length] bool, flattened to [G, length]), ``length`` when a
    row attends no key; None without a mask."""
    if mask is None:
        return None
    m = mask.reshape(-1, length)
    pos = torch.arange(1, length + 1, device=m.device, dtype=torch.int32)
    last = (m.to(torch.int32) * pos).amax(-1)
    return torch.where(last > 0, last, length).to(torch.int32).contiguous()

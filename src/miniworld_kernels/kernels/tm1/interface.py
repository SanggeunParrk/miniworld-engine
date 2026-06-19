"""CuTeDSL entry point for tm1 (placeholder).

To be implemented. The signature mirrors `tm1_pytorch` so the bench script can
drop the cute kernel in without rewiring.
"""

from __future__ import annotations

import torch


def tm1_cute(
    x: torch.Tensor,
    WL: torch.Tensor,
    WLg: torch.Tensor,
    WR: torch.Tensor,
    WRg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("tm1 CuTeDSL kernel not implemented yet")

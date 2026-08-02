"""Public entry point for the fused trimul input-projection kernel.

Single import surface so callers don't need to know the backend folder layout::

    from miniworld_engine.kernels.trimul_inproj.interface import trimul_inproj_cute
    left_bdll, right_bdll, gate_blld = trimul_inproj_cute(x_normed, WL, WLg, WR, WRg, Wg)

The cute backend is imported lazily so importing this module doesn't require the
cute toolchain (cutlass-dsl + quack).
"""

from __future__ import annotations

import torch


def trimul_inproj_cute(
    x: torch.Tensor,  # normalized pair; (B,L,L,D) if hidden_dim=-1/3, (B,D,L,L) if hidden_dim=1
    WL: torch.Tensor,  # (D, D)  — to_left.weight.T
    WLg: torch.Tensor,  # (D, D)  — to_left_gate.weight.T
    WR: torch.Tensor,  # (D, D)  — to_right.weight.T
    WRg: torch.Tensor,  # (D, D)  — to_right_gate.weight.T
    Wg: torch.Tensor,  # (D, D)  — to_gate.weight.T
    *,
    hidden_dim: int = -1,  # channel (D) axis position: -1/3 -> BLLD, 1 -> BDLL
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(left_bdll, right_bdll, gate_blld)``. See ``reference.py``.

    ``hidden_dim`` lets the pair come in channel-last (BLLD, default) or channel-first
    (BDLL, ``hidden_dim=1``); the BDLL input is read via an M-major GEMM operand with no
    pre-permute. Output layout is unchanged.
    """
    from .cute.launch import trimul_inproj_cute_forward

    return trimul_inproj_cute_forward(x, WL, WLg, WR, WRg, Wg, hidden_dim=hidden_dim)

"""Shape constants and two helpers of the fused SWA atom-DiT block (from the deleted ``swa_dit.interface``)."""

from __future__ import annotations

import torch

#: The block the fused kernels are built for: d_atom 128, 4 heads, SwiGLU hidden 256, window 2 x 64 + 1.
D_ATOM = 128
N_HEAD = 4
N_HIDDEN = 256
HALF_WINDOW = 64
FP32_EPS = float(torch.finfo(torch.float32).eps)
#: A half window at or above this (or negative) means global attention (no window).
GLOBAL_HALF_WINDOW = 1 << 16


def is_global(half_window: int) -> bool:
    """Whether ``half_window`` selects global attention (no window) -- negative, or at least ``GLOBAL_HALF_WINDOW``."""
    return half_window < 0 or half_window >= GLOBAL_HALF_WINDOW


def _matmul_precision() -> str:
    """torch's float32 matmul precision, read as a constant under torch.compile (Dynamo cannot trace the getter)."""
    return torch.get_float32_matmul_precision()


_matmul_precision = torch.compiler.assume_constant_result(_matmul_precision)

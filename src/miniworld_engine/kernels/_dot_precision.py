"""The ``input_precision`` of a Triton ``tl.dot`` whose operands may be fp32, decided once per launch from torch's own switch.

bf16 operands ignore it. For fp32 operands the engine follows the switch cuBLAS follows (``torch.backends.cuda.matmul.allow_tf32`` or
``torch.set_float32_matmul_precision`` not ``"highest"``), the same one the H100 CUDA Transition kernels
(``kernels/transition/cuda/fused_generic_sm90a``) obey: TF32 tensor cores when torch allows them, IEEE fp32 otherwise. A launcher passes
the result to its kernel as a ``tl.constexpr``.
"""
from __future__ import annotations

import torch


def tf32_allowed() -> bool:
    """Does torch allow TF32 tensor-core matmuls for fp32 right now?"""
    return bool(torch.backends.cuda.matmul.allow_tf32) or torch.get_float32_matmul_precision() != "highest"


def dot_precision(dtype: torch.dtype) -> str:
    """``"tf32"`` for fp32 operands when torch allows TF32, else ``"ieee"`` (bf16 and everything else: unchanged)."""
    return "tf32" if dtype is torch.float32 and tf32_allowed() else "ieee"

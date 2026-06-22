"""Fused LayerNorm + Linear (`te.LayerNormLinear` analogue).

LayerNorm over the last dim immediately followed by a Linear (GEMM + bias).
`reference.py` holds the PyTorch math (and the `torch.compile` baseline module);
`interface.py` is the Triton kernel entry point (WIP). See README.md.
"""

from __future__ import annotations

from .cute import layernorm_linear
from .interface import layernorm_linear_triton
from .reference import LayerNormLinearRef, layernorm_linear_pytorch

__all__ = [
    "LayerNormLinearRef",
    "layernorm_linear",          # SM90 cute forward; auto-dispatches fused M2 (N<=256) / M1
    "layernorm_linear_pytorch",
    "layernorm_linear_triton",
]

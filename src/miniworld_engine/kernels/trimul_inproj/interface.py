"""Public entry points for the trimul_inproj family (the fused triangle multiplicative update).

Triton, every arch: :func:`trimul_triton` (single direction) and
:func:`bidirectional_trimul_triton` (outgoing + incoming in one block), both weights-as-args
autograd Functions with a forward-only inference path. The hand-CUDA H100 kernels under
``cuda/`` are reached through ``integrations.trimul_h100``, which states their contract.
"""

from __future__ import annotations

from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
    bidirectional_trimul_triton,
)
from miniworld_engine.kernels.trimul_inproj.triton.unidirectional import trimul_triton

__all__ = ["bidirectional_trimul_triton", "trimul_triton"]

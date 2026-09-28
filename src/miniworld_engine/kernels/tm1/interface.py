"""Public entry point for the tm1 (dual sigmoid-gate projection) kernel family.

tm1 is the first half of triangle multiplication: from one input it produces the two
gated projections ``(sigmoid(x@WLg)·(x@WL), sigmoid(x@WRg)·(x@WR))`` in a single kernel,
so the four GEMMs share one read of ``x`` and the gates never round-trip through HBM.
:mod:`.reference` holds the PyTorch definition the kernel is checked against.

Only the Triton path exists; import is side-effect free with a plain module-level import.
"""

from __future__ import annotations

from miniworld_engine.kernels.tm1.triton.main import triton_tm1

__all__ = [
    "triton_tm1",
]

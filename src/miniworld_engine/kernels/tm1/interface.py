"""Public entry point for the tm1 (dual sigmoid-gate projection) kernel family.

tm1 is the first half of triangle multiplication: from one input it produces the two
gated projections ``(sigmoid(x@WLg)·(x@WL), sigmoid(x@WRg)·(x@WR))`` in a single kernel,
so the four GEMMs share one read of ``x`` and the gates never round-trip through HBM.
:mod:`.reference` holds the PyTorch definition the kernel is checked against.

``triton_tm1`` is the Triton path; on an A100 (sm_80) ``cuda_tm1`` is the hand-CUDA one (bf16, forward + backward; ``cuda_tm1_serves`` is its gate: bf16, widths the
64-column tiles divide, capability 8.0, ``MINIWORLD_GATED_SM80`` not 0). Nothing dispatches between them: no module calls tm1 (the triangle-multiplication modules run
the fused front), so the caller picks. Import is side-effect free: the extension builds on the first ``cuda_tm1`` call.
"""

from __future__ import annotations

from miniworld_engine.kernels.tm1.cuda.sm80 import cuda_tm1
from miniworld_engine.kernels.tm1.cuda.sm80 import serves as cuda_tm1_serves
from miniworld_engine.kernels.tm1.triton.main import triton_tm1

__all__ = [
    "cuda_tm1",
    "cuda_tm1_serves",
    "triton_tm1",
]

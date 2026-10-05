"""Public entry point for tm2 (the trimul output gate + projection + mul).

``triton_tm2`` is the Triton path: ``sigmoid(x_gate @ W_gate) * (x_out @ W_out)`` with the weights in ``(D, D)`` matmul form (``x @ W``), matching ``reference.py``.
On an A100 (sm_80) ``cuda_tm2`` is the hand-CUDA one (bf16, forward + backward; ``cuda_tm2_serves`` is its gate: bf16, widths the 64-column tiles divide, capability
8.0, ``MINIWORLD_GATED_SM80`` not 0). Nothing dispatches between them: no module calls tm2 (the triangle-multiplication modules run the fused back half), so the caller
picks.
"""

from __future__ import annotations

from miniworld_engine.kernels.tm2.cuda.sm80 import cuda_tm2
from miniworld_engine.kernels.tm2.cuda.sm80 import serves as cuda_tm2_serves
from miniworld_engine.kernels.tm2.triton.main import triton_tm2

__all__ = ["cuda_tm2", "cuda_tm2_serves", "triton_tm2"]

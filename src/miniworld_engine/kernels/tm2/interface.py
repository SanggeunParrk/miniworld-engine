"""Public entry point for tm2 (the trimul output gate + projection + mul).

Only the Triton path is exported: ``sigmoid(x_gate @ W_gate) * (x_out @ W_out)`` with the
weights in ``(D, D)`` matmul form (``x @ W``), matching ``reference.py``.
"""

from __future__ import annotations

from miniworld_engine.kernels.tm2.triton.main import triton_tm2

__all__ = ["triton_tm2"]

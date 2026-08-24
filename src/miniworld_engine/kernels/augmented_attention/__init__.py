"""Augmented-attention (pair-bias) fused triton kernels.

Two backends with an identical ``(q, k, v, bias, mask)`` signature, selected per call via the
``compute_efficient`` kwarg; they live in ``triton/main.py`` and ``triton/memory_efficient.py``.
See ``interface.py`` -- the family's public entry point -- for the choice between them.
"""

from __future__ import annotations

from .interface import triton_augmented_attention_pair_bias

__all__ = ["triton_augmented_attention_pair_bias"]

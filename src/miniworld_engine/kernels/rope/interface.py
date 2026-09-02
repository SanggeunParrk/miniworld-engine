"""Public entry point for the rope family.

3D RoPE rotates q/k by a batch-dependent cos/sin before SWA attention. The eager
``apply_rotary_emb_3d`` is four HBM passes with two temporaries; this is one pass.
"""
from __future__ import annotations

from miniworld_engine.kernels.rope.triton.main import triton_rope_3d

__all__ = ["triton_rope_3d"]

"""Engine-backed sliding-window atom attention and its geometry helpers.

Keep the historical team-gm import path and checkpoint parameter names while
using the engine's fused Q/K preprocessing and output gate. CUDA execution
requires the engine's device-appropriate FlashAttention backend; CPU tests use
its dense reference.
"""

from miniworld_engine.modules.swa_atom_attention import (
    _FLASH_AVAILABLE,
    SWA3DRoPEAttention,
    apply_rotary_emb_3d,
    build_3d_rope,
    build_attention_params,
    flash_window_seqused,
    sparse_neighbor_attention,
)

__all__ = [
    "_FLASH_AVAILABLE",
    "SWA3DRoPEAttention",
    "apply_rotary_emb_3d",
    "build_3d_rope",
    "build_attention_params",
    "flash_window_seqused",
    "sparse_neighbor_attention",
]

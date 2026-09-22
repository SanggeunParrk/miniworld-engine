"""SWA3DRoPEAttention (model-level op: sliding-window 3D-RoPE atom attention)."""

from miniworld_engine.modules.swa_atom_attention.module import (
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

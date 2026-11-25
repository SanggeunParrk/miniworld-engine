"""Triton-free static policies shared by MPNN message implementations."""

from __future__ import annotations


_INT32_MAX = 2**31 - 1
_DX_TILE_ELEMENTS = 64 * 128
_INFERENCE_PADDED_TAIL_ELEMENTS = 80 * 128


def _requires_i64_indexing(elements: int) -> bool:
    """Account for the masked padding in the final 64-row dX tile."""
    return elements > _INT32_MAX - (_DX_TILE_ELEMENTS - 1)


def _inference_int32_elements_supported(elements: int) -> bool:
    """Check valid and masked two-group inference tile offsets."""
    return elements <= _INT32_MAX - (_INFERENCE_PADDED_TAIL_ELEMENTS - 1)


__all__ = [
    "_DX_TILE_ELEMENTS",
    "_INFERENCE_PADDED_TAIL_ELEMENTS",
    "_INT32_MAX",
    "_inference_int32_elements_supported",
    "_requires_i64_indexing",
]

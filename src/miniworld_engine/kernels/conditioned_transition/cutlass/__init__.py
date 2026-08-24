"""CUTLASS C++ (Hopper TF32 WGMMA) fused-epilogue ConditionedTransition-tail training path.

Forward fuses SwiGLU into the expand GEMM epilogue and the sigmoid-gate into the squeeze
GEMM epilogue (CUTLASS EVT). Backward uses tuned plain CUTLASS GEMMs + fused elementwise.

The compiled extension (``ct_train_ext``) is built out-of-tree under ``_ct_cutlass``; this
module is import-safe even when the extension is absent (returns None loader).
"""
from __future__ import annotations

__all__ = ["cond_transition_train_cutlass", "load_ext"]


def load_ext():
    try:
        import ct_train_ext  # ty: ignore[unresolved-import]  # built by setup.py, not vendored

        return ct_train_ext
    except Exception:
        return None

"""Public entry point for the Transition (LayerNorm + SwiGLU expand/squeeze) kernel family.

The op is ``y = (silu(LN(x)@Wa^T) * (LN(x)@Wb^T)) @ Ws^T`` over the last dimension; see
:mod:`.reference` for the exact definition. :func:`triton_transition` takes an already
normalised ``x`` and fuses only the expand pair, gate and squeeze;
:func:`triton_transition_fused` folds the input LayerNorm in as well and is the path the
``Transition`` module uses.

:func:`cuda_transition_b2b` and :func:`cute_transition_fused` are deliberately lazy. The
hand-CUDA path compiles its ``.so`` through nvcc on first call and the cute path pulls in
cutlass, so importing either at module scope would make merely importing this family do
compiler and toolkit work -- which ``tests/compile/test_public_api.py::test_import_is_side_effect_free``
forbids. Wrapping them keeps the name available on the public surface while the cost stays
on the first call. The Triton entries need no such treatment: Triton is a hard dependency
and its modules do no work at import.

Note that ``cuda_transition`` is NOT exported here -- it is a frozen name in
:mod:`miniworld_engine.kernels` that raises ``NotImplementedError`` and has no
implementation in this family to point at.
"""

from __future__ import annotations

from miniworld_engine.kernels.transition.triton.fused import (
    triton_swiglu_ffn,
    triton_transition_fused,
)
from miniworld_engine.kernels.transition.triton.main import triton_transition


def cuda_transition_b2b(*args, **kwargs):
    """Lazy hand-CUDA fused b2b Transition forward (builds the .so on first call).

    Fixed AF3 shapes only (d_hidden=128, n=4 -> K=128, ND=512, D=128). Beats the Triton
    b2b forward ~1.29x at this config. Inference-only (no backward saved)."""
    from miniworld_engine.kernels.transition.cuda import (
        cuda_transition_b2b as _impl,
    )

    return _impl(*args, **kwargs)


def cute_transition_fused(*args, **kwargs):
    """Lazy cute (quack SM90 WGMMA) Transition fwd+bwd entry (imports cutlass on first call)."""
    from miniworld_engine.kernels.transition.cute.fused import (
        cute_transition_fused as _impl,
    )

    return _impl(*args, **kwargs)


__all__ = [
    "cuda_transition_b2b",
    "cute_transition_fused",
    "triton_swiglu_ffn",
    "triton_transition",
    "triton_transition_fused",
]

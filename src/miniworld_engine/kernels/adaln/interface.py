"""Public entry point for the adaptive-LayerNorm (adaLN) kernel family.

The family computes the DiT conditioning op ``y = sigmoid(scale)·LN(x) + bias``, where
``scale`` and ``bias`` are two Linear projections of a weighted LayerNorm of ``cond``
(see :mod:`.reference` for the exact formula). Three entry points are exported because
the fwd-only and fwd+bwd cases want different fusions: :func:`adaln_inference` saves
nothing for backward and is free to fold the whole thing into as few kernels as the
shape allows, while :func:`adaln_train` keeps the stats and the gate and pairs its
forward with a symmetric GEMM backward. original single-autograd-Function kernel both were derived from, kept as the reference
implementation the two specialised paths are benchmarked and checked against.

Everything reachable from here is Triton, so the imports are eager: Triton is a hard
dependency and none of these modules touch the GPU at import time. The family's cutlass
sources (``adaln/cutlass``) are build inputs of the inference path, not importable
entry points, so nothing here needs a deferred import.
"""

from __future__ import annotations

from miniworld_engine.kernels.adaln.triton.inference import adaln_inference
from miniworld_engine.kernels.adaln.triton.training import adaln_train

__all__ = [
    "adaln_inference",
    "adaln_train",
]

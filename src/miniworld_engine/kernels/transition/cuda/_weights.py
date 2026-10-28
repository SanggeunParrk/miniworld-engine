"""The squeeze weight K-major (``Ws^T`` contiguous) the fused sm_90a Transition forwards read, reused across calls.

Making that copy on every call is an extra 1.7 µs kernel (+ launch gap) in a ~7 µs inference call; an inference loop calls with the same
weights, so the copy is one of the weight forms of ``kernels._capture.param_forms``: made once and reused while the weight is the same
tensor at the same version, scoped to a CUDA-graph capture like every weight pack, kept while the weight is alive. Weights that require
grad outside no_grad are never cached.

``_CACHE`` is that store (``_capture._FORMS``); FoldForge clears it before it loads a model.
"""

import torch

from miniworld_engine.kernels import _capture

_CACHE = _capture._FORMS


def transposed_contiguous(w: torch.Tensor) -> torch.Tensor:
    """``w.t().contiguous()``, reused while ``w`` is the same tensor and unchanged."""
    if torch.is_grad_enabled() and w.requires_grad:
        return w.t().contiguous()
    return _capture.param_forms("transition_ws_t", (w,), lambda: w.t().contiguous())

"""Per-call weight re-layouts the fused sm_90a Transition builds need, cached across calls.

The forward kernels read the squeeze weight K-major, i.e. ``Ws^T`` contiguous.  Making that copy on every call is an extra 1.7 µs kernel
(+ launch gap) in a ~7 µs inference call; an inference loop calls with the same weights, so the copy is made once and reused until the
weight's storage or version counter changes.  A call made while a CUDA graph is being captured reads the cache but never fills it (the copy would live in the graph's
private pool), so warm the loop up once outside the capture; weights that require grad outside no_grad are never cached.
"""

import torch

_CACHE: dict = {}
_MAX = 64


def transposed_contiguous(w: torch.Tensor) -> torch.Tensor:
    """``w.t().contiguous()``, reused while ``w`` is unchanged."""
    if torch.is_grad_enabled() and w.requires_grad:
        return w.t().contiguous()
    key = (w.data_ptr(), w._version, tuple(w.shape), w.dtype, w.device)
    hit = _CACHE.get(key)
    if hit is not None:                       # a hit is a plain tensor with a stable address: fine to read inside a graph capture
        return hit
    out = w.t().contiguous()
    if w.is_cuda and torch.cuda.is_current_stream_capturing():
        return out                            # never cache a tensor that lives in a graph's private pool
    if len(_CACHE) >= _MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


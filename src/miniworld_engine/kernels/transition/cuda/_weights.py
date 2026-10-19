"""Per-call weight re-layouts the fused sm_90a Transition builds need, cached across calls.

The forward kernels read the squeeze weight K-major, i.e. ``Ws^T`` contiguous.  Making that copy on every call is an extra 1.7 µs kernel
(+ launch gap) in a ~7 µs inference call; an inference loop calls with the same weights, so the copy is made once and reused until the
weight's storage or version counter changes.  A call made while a CUDA graph is being captured reads the cache but never fills it (the copy would live in the graph's
private pool), so warm the loop up once outside the capture; weights that require grad outside no_grad are never cached.

The key is the weight's address and version, and an address is reused as soon as a tensor is freed: a new weight (another model, the next
test) can land on a dead one's address with the same version. An entry therefore also holds a weak reference to its weight and only serves
that very tensor object.
"""

import weakref

import torch

_CACHE: dict = {}
_MAX = 64


def transposed_contiguous(w: torch.Tensor) -> torch.Tensor:
    """``w.t().contiguous()``, reused while ``w`` is the same tensor and unchanged."""
    if torch.is_grad_enabled() and w.requires_grad:
        return w.t().contiguous()
    key = (w.data_ptr(), w._version, tuple(w.shape), w.dtype, w.device)
    hit = _CACHE.get(key)
    if hit is not None and hit[0]() is w:    # a hit is a plain tensor with a stable address: fine to read inside a graph capture
        return hit[1]
    out = w.t().contiguous()
    if w.is_cuda and torch.cuda.is_current_stream_capturing():
        return out                            # never cache a tensor that lives in a graph's private pool
    if len(_CACHE) >= _MAX:
        for dead in [k for k, (ref, _) in _CACHE.items() if ref() is None]:
            del _CACHE[dead]
        if len(_CACHE) >= _MAX:
            _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (weakref.ref(w), out)
    return out

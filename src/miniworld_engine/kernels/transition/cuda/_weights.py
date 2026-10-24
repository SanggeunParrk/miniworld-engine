"""Per-call weight re-layouts the fused sm_90a Transition builds need, cached across calls.

The forward kernels read the squeeze weight K-major, i.e. ``Ws^T`` contiguous.  Making that copy on every call is an extra 1.7 µs kernel
(+ launch gap) in a ~7 µs inference call; an inference loop calls with the same weights, so the copy is made once and reused until the
weight's storage or version counter changes.  A call made while a CUDA graph is being captured reads the cache but never fills it (the copy would live in the graph's
private pool), so warm the loop up once outside the capture; weights that require grad outside no_grad are never cached.

The key is the weight's address and version, and an address is reused as soon as a tensor is freed: a new weight (another model, the next
test) can land on a dead one's address with the same version. An entry therefore also holds a weak reference to its weight and only serves
that very tensor object.

A captured graph reads the entry it hit, so an entry is never evicted while its weight is alive: it goes when the weight dies or when the
same weight is cached at a newer version. A size cap evicted live entries -- a model with more Transitions than the cap (Boltz-2's 64
pairformer blocks plus its template, MSA and confidence stacks) freed transposes that captured trunk graphs still read, and the next
replay faulted with an illegal memory access.
"""

import weakref

import torch

_CACHE: dict = {}


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
    for stale in [k for k, (ref, _) in _CACHE.items() if ref() is None or ref() is w]:
        del _CACHE[stale]
    _CACHE[key] = (weakref.ref(w), out)
    return out

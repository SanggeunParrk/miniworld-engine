"""Weight-pack caches under CUDA-graph capture.

Several integrations keep a derived form of a module's weights (packed / folded / cast copies, a pair bias) and reuse it
while the source tensors are unchanged -- keyed on their pointer and ``_version``. Inside a CUDA-graph capture such a hit is
wrong: the capture then records no packing kernel, and every replay reads the pack as it was when the graph was captured,
never the weights an optimizer step has since written. Not caching at all inside a capture is right but repacks at every call
of the captured step (a sampling loop calls the same block tens of times).

So the cache is *scoped*: an entry made outside a capture serves only eager calls, an entry made during a capture serves only
that capture (its packing kernels are recorded once, and each replay repacks the current weights), and entries of finished
captures are dropped. The scope of a call is the id of the capture sequence its stream belongs to (a side stream forked into
the capture shares it).

One case no key can catch: an update that writes a tensor without bumping its ``_version``. torch's fused optimizers
(``Adam(..., fused=True)`` and the like) do that; after their step call ``torch.autograd.graph.increment_version(params)``
(no kernel) so eager calls repack.

**Static weights.** A graph whose weights never change between replays (inference: a sampling run, a served model) gains nothing
from repacking and pays for it: every replay then re-runs the packing kernels (tens of small copies and casts per block -- a token
DiT inference block goes from 84 us to 200 us). ``static_weights()`` (or ``MINIWORLD_STATIC_WEIGHTS=1``) declares that contract: the
scoping is off and a capture serves from, and fills, the same caches as an eager call, as the integrations did before the scoping.
Under it an in-place weight update after the capture is *not* seen by replays -- do not set it around a training graph, nor around a
sampling run inside one. The benchmark harness sets it for ``mode=inference``.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os

import torch

_STATIC = [os.environ.get("MINIWORLD_STATIC_WEIGHTS", "0") == "1"]
_STATIC_INPUTS = [os.environ.get("MINIWORLD_STATIC_CONDITIONING", "0") == "1"]


def set_static_weights(on: bool) -> bool:
    """Declare (or withdraw) that weights stay fixed between graph replays; returns the previous setting."""
    prev, _STATIC[0] = _STATIC[0], bool(on)
    return prev


def set_static_inputs(on: bool) -> bool:
    """Declare (or withdraw) that the *conditioning inputs* of hoisted per-item tables (a block's conditioning, a pair's bias) stay fixed between
    graph replays; returns the previous setting. Stronger than ``static_weights``: see ``lookup_inputs``."""
    prev, _STATIC_INPUTS[0] = _STATIC_INPUTS[0], bool(on)
    return prev


@contextlib.contextmanager
def static_inputs(on: bool = True):
    """Within the block, hoisted per-item tables (``lookup_inputs``) serve a capture from the eager entries: a replay then runs none of the
    kernels that make them. The caller promises that the weights AND the conditioning / pair tensors keep their contents between replays (an
    inference run over one item); a sampler that refills them in place between replays must not set it."""
    prev = set_static_inputs(on)
    try:
        yield
    finally:
        set_static_inputs(prev)


@contextlib.contextmanager
def static_weights(on: bool = True):
    """Within the block, weight-pack caches ignore the capture scope (see the module docstring). Inference graphs only."""
    prev = set_static_weights(on)
    try:
        yield
    finally:
        set_static_weights(prev)


@functools.lru_cache(maxsize=1)
def _capture_info():
    lib = ctypes.CDLL("libcuda.so.1")
    fn = lib.cuStreamGetCaptureInfo          # (stream, status *, id *): the driver's original three-argument entry point
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64)]
    fn.restype = ctypes.c_int
    return fn


def capture_id() -> int | None:
    """None outside a capture; inside one, the id of the capture sequence (0 when it cannot be read). Always None under
    ``static_weights``: captures then share the eager entries."""
    if _STATIC[0] or not torch.cuda.is_current_stream_capturing():
        return None
    try:
        fn = _capture_info()
    except (OSError, AttributeError):
        return 0
    status, cid = ctypes.c_int(), ctypes.c_uint64()
    if fn(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), ctypes.byref(status), ctypes.byref(cid)) != 0 or status.value != 1:
        return 0
    return int(cid.value) or 0


def scoped(key):
    """``key`` extended by the current scope -- or None when the call must not touch the cache (a capture whose id is unknown)."""
    cid = capture_id()
    if cid == 0:
        return None
    return (cid, key)


def prune(store: dict) -> None:
    """Drop the entries of captures other than the current one (eager entries stay)."""
    cid = capture_id()
    stale = [k for k in store if isinstance(k, tuple) and len(k) == 2 and k[0] is not None and k[0] != cid]
    for k in stale:
        del store[k]


def lookup(store: dict, key, build, limit: int = 64):
    """``build()``, cached in ``store`` under ``key`` within the current scope (see the module docstring)."""
    sk = scoped(key)
    if sk is None:
        return build()
    hit = store.get(sk)
    if hit is not None:
        return hit
    prune(store)
    while len(store) >= limit:
        store.pop(next(iter(store)))
    out = store[sk] = build()
    return out


def lookup_inputs(store: dict, key, build, limit: int = 64, valid=None, alive=None):
    """``lookup`` for a table made from a per-item INPUT (a conditioning tensor, a pair): the table is the same for every call that reads that
    tensor unchanged, so a sampler's steps (eager) and a captured step's block calls share it. Scoped to the capture like every pack, except
    under ``static_inputs()``: then a capture serves from, and fills, the eager entries and a replay re-runs none of the producing kernels.

    An input is an activation: its storage address is reused as soon as it is freed, so (pointer, version) alone can name a different tensor
    with the same version. ``valid(entry)`` is asked of a hit (False: the entry belongs to another tensor, it is remade and replaced) and
    ``alive(entry)`` of every entry when the store is pruned (False: the tensor is gone, the entry is dropped)."""
    sk = (None, key) if _STATIC_INPUTS[0] else scoped(key)
    if sk is None:
        return build()
    hit = store.get(sk)
    if hit is not None and (valid is None or valid(hit)):
        return hit
    if not _STATIC_INPUTS[0]:
        prune(store)
    if alive is not None:
        for k in [k for k, e in store.items() if not alive(e)]:
            del store[k]
    while len(store) >= limit and sk not in store:
        store.pop(next(iter(store)))
    out = store[sk] = build()
    return out

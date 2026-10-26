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
DiT inference block goes from 84 us to 200 us). ``static_weights()`` (or ``settings.static_weights=1``) declares that contract: the
scoping is off and a capture serves from, and fills, the same caches as an eager call, as the integrations did before the scoping.
Under it an in-place weight update after the capture is *not* seen by replays -- do not set it around a training graph, nor around a
sampling run inside one. The benchmark harness sets it for ``mode=inference``.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import weakref

import torch

from miniworld_engine import settings

#: Entries a per-block cache must hold: its key names the block's own weights, so one model run touches one entry per block
#: (24 token-DiT blocks, 48 pairformer blocks). A limit below the block count evicts every entry before its block comes
#: round again, and the cache then rebuilds at every call -- and, in a CUDA-graph capture, records the rebuild kernels so
#: every replay runs them (a token DiT's pair bias alone was 15 % of a 1,024-token sampler).
BLOCK_CACHE_LIMIT = 64

_STATIC = [settings.current().static_weights]
_STATIC_INPUTS = [settings.current().static_conditioning]


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
    fn = (
        lib.cuStreamGetCaptureInfo
    )  # (stream, status *, id *): the driver's original three-argument entry point
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    fn.restype = ctypes.c_int
    return fn


def _graph_capture_id() -> int | None:
    """None outside a capture; inside one, the id of the capture sequence (0 when it cannot be read) -- whatever is declared static."""
    if not torch.cuda.is_current_stream_capturing():
        return None
    try:
        fn = _capture_info()
    except (OSError, AttributeError):
        return 0
    status, cid = ctypes.c_int(), ctypes.c_uint64()
    if (
        fn(
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            ctypes.byref(status),
            ctypes.byref(cid),
        )
        != 0
        or status.value != 1
    ):
        return 0
    return int(cid.value) or 0


def capture_id() -> int | None:
    """The scope of a WEIGHT pack: None outside a capture; inside one, the id of the capture sequence (0 when it cannot be read).
    Always None under ``static_weights``: captures then share the eager entries."""
    return None if _STATIC[0] else _graph_capture_id()


def scoped(key):
    """``key`` extended by the current scope -- or None when the call must not touch the cache (a capture whose id is unknown)."""
    cid = capture_id()
    if cid == 0:
        return None
    return (cid, key)


def prune(store: dict, cid: int | bool | None = False) -> None:
    """Drop the entries of captures other than the current one (eager entries stay). ``cid``: the current scope, when the caller
    scopes by something other than ``capture_id()``."""
    if cid is False:
        cid = capture_id()
    stale = [
        k
        for k in store
        if isinstance(k, tuple) and len(k) == 2 and k[0] is not None and k[0] != cid
    ]
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
    ``alive(entry)`` of every entry when the store is pruned (False: the tensor is gone, the entry is dropped).

    The scope is the capture itself, not ``capture_id()``: ``static_weights()`` says nothing about the inputs, so a capture under it alone
    still makes its own tables and a replay remakes them."""
    if _STATIC_INPUTS[0]:
        cid = None
    else:
        cid = _graph_capture_id()
        if cid == 0:
            return build()
    sk = (cid, key)
    hit = store.get(sk)
    if hit is not None and (valid is None or valid(hit)):
        return hit
    if not _STATIC_INPUTS[0]:
        prune(store, cid)
    if alive is not None:
        for k in [k for k, e in store.items() if not alive(e)]:
            del store[k]
    while len(store) >= limit and sk not in store:
        store.pop(next(iter(store)))
    out = store[sk] = build()
    return out


_FORMS: dict = {}


def param_forms(tag: str, params, build):
    """``build()`` -- the kernel-dtype forms of some parameters (casts, contiguous copies, folds) -- cached while the very same
    parameter storages are unmodified, scoped to the capture like every weight pack (``lookup``): an eager call and a captured step
    under ``static_weights()`` share the entry, so a replay re-runs none of the cast kernels; a capture without it records them once
    and every replay casts the weights as they are then.

    The reason it exists: with fp32 master parameters over bf16 activations every call casts every weight, ~10 small kernels per
    block per call (about 6-12 us), which made inference of the small blocks 10-50 % slower than with bf16 parameters. The key is
    (pointer, version, dtype, shape) of each parameter and the entry keeps the parameters alive, so a freed storage's address cannot
    alias a different tensor. A fused optimizer step does not bump ``_version`` -- as for every other pack, call
    ``torch.autograd.graph.increment_version(params)`` after it. Inference tensors carry no version counter: not cached."""
    try:
        key = (
            tag,
            tuple((p.data_ptr(), p._version, p.dtype, tuple(p.shape)) for p in params),
        )
    except RuntimeError:
        return build()
    return lookup(_FORMS, key, lambda: (build(), tuple(params)), limit=64)[0]


def hoisted_table(store: dict, tensor: torch.Tensor, key, build, limit: int = 64):
    """The table ``build()`` makes from the per-item input ``tensor``, once per (tensor, version, ``key``) (``lookup_inputs``): the entry keeps a weak
    reference to the tensor, so a new tensor that reuses a freed address is not served the old one's table, and entries of freed tensors are dropped.

    A sample-expanded conditioning (stride 0 over the samples: a sampler step's view of the fold's one conditioning) is a NEW tensor object at
    every call over the same small storage, so no weak reference to it outlives the call and the table would be remade at every call -- in a
    captured step in every replay, for the whole life of the sampler. Such an entry holds the tensor itself (its storage is the fold's one
    conditioning, which keeps its address from being reused while the entry stands); a view is otherwise named by the tensor it views."""
    held = any(st == 0 and n > 1 for st, n in zip(tensor.stride(), tensor.shape, strict=True))
    root = tensor._base if tensor._base is not None else tensor
    tkey = (tensor.data_ptr(), tensor._version, tuple(tensor.shape), tensor.dtype, tensor.device)
    entry = lookup_inputs(
        store,
        (tkey, key),
        lambda: (tensor if held else weakref.ref(root), held, build()),
        limit=limit,
        valid=lambda e: e[1] or e[0]() is root,
        alive=lambda e: e[1] or e[0]() is not None,
    )
    return entry[2]

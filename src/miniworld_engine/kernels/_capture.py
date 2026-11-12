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
DiT inference block goes from 84 us to 200 us). ``static_weights()`` (or ``set_static_weights(True)``) declares that contract: the
scoping is off and a capture serves from, and fills, the same caches as an eager call, as the integrations did before the scoping.
Under it an in-place weight update after the capture is *not* seen by replays -- do not set it around a training graph, nor around a
sampling run inside one. The benchmark harness sets it for ``mode=inference``.

**Refreshing hoisted tables (FoldForge R26).** Under ``static_inputs()`` a capture is served the per-item tables (``lookup_inputs`` /
``hoisted_table``) its warm-up made eagerly, and the graph reads those tensors at every replay. A caller that keeps the graph for the next
item writes that item's inputs into the same input tensors and calls ``refresh_static_inputs``: every such read was recorded
(``note_static_read``: the table, the call's ``build``, its inputs, the module that ran -- ``CALLER``, set by ``BackendModule.forward``) and
the refresh writes a new ``build()`` into the recorded tensors. A ``build`` therefore derives its table from the keyed inputs and the weights
only. Entries made inside a capture are recorded into the graph (every replay remakes them) and are not refreshed. The records are held by
the ``static_inputs()`` handle (``StaticTables``) that was open at the capture: they go with it, so a caller that refreshes keeps the
handle, and one that captures a new graph per item (no refresh) leaves nothing behind that holds an item's inputs or tables.

**What a graph reads stays allocated (``hold``).** A captured graph reads the cache entries it was served (weight packs, the token DiT's
grouped conditioning weights, hoisted tables) at every replay, but the stores keep evicting and replacing entries on later calls -- another
shape's warm-up and capture fill the same stores (an LRU of 4 conditioning chunks, count limits, scope prunes, a newer version). Every
entry a capture is served is therefore held (``hold``) by each ``static_inputs()`` handle open at the capture, and goes with the handle
(``release()`` or dropped); with no handle open, an eager-scope entry is held by the process until ``clear_weight_caches``. So: keep the
handle for as long as the graph is replayed (FoldForge's per-shape sampler graph keeps it with the graph). A cache that serves a capture
from outside these helpers (``lookup`` / ``lookup_inputs`` / ``param_forms`` / ``hoisted_table``) calls ``hold`` itself.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import weakref
from typing import cast

import torch

#: Entries a per-block cache must hold: its key names the block's own weights, so one model run touches one entry per block
#: (24 token-DiT blocks, 48 pairformer blocks). A limit below the block count evicts every entry before its block comes
#: round again, and the cache then rebuilds at every call -- and, in a CUDA-graph capture, records the rebuild kernels so
#: every replay runs them (a token DiT's pair bias alone was 15 % of a 1,024-token sampler).
BLOCK_CACHE_LIMIT = 64

_STATIC = [False]
_STATIC_INPUTS = [False]


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
    inference run over one item) -- or, when it refills them in place between replays (a sampler graph reused over folds), that it calls
    :func:`refresh_static_inputs` (or the yielded handle's ``refresh``) after each refill and before the next replay (the module docstring,
    "Refreshing hoisted tables").

    Yields a :class:`StaticTables` handle: the hoisted tables that the captures inside the block read from the eager cache, and every engine
    cache entry those captures read (``hold``), kept allocated while the handle lives: keep it as long as the graphs are replayed."""
    prev = set_static_inputs(on)
    handle = StaticTables()
    _HANDLES.append(handle)
    try:
        yield handle
    finally:
        _HANDLES.remove(handle)
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
    fn = lib.cuStreamGetCaptureInfo  # (stream, status *, id *): the driver's original three-argument entry point
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    fn.restype = ctypes.c_int
    return fn


def _graph_capture_id() -> int | None:
    """None outside a capture; inside one, the id of the capture sequence (0 when it cannot be read) -- whatever is declared static."""
    # a process that never initialised CUDA (CPU-only, or not yet) captures nothing; ``is_initialized`` is a flag read, ``is_available``
    # asks the driver / the environment at every call (four times per token DiT block call)
    if not torch.cuda.is_initialized() or not torch.cuda.is_current_stream_capturing():
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
    stale = [k for k in store if isinstance(k, tuple) and len(k) == 2 and k[0] is not None and k[0] != cid]
    for k in stale:
        del store[k]


#: id(store) -> the eager keys a capture under ``static_weights()`` / ``static_inputs()`` read or filled. A replay reads those entries, so the
#: count limit never evicts them (evicting one freed memory a captured graph still read); they go with the store's other eviction rules.
_PINNED: dict[int, set] = {}


def _pin_if_static_capture(store: dict, sk) -> set:
    pinned = _PINNED.setdefault(id(store), set())
    if sk[0] is None and _graph_capture_id() is not None:
        pinned.add(sk)
    return pinned


def _evict(store: dict, pinned: set, limit: int, keep) -> None:
    """Drop the oldest entries that no static capture reads until fewer than ``limit`` of those are left. The pinned entries do not
    count: they are what live graphs read, and while they filled the limit every new entry was evicted as it was made -- a
    sampler that captures each trajectory then recorded the table's rebuild into every replay from its second trajectory on."""
    free = [k for k in store if k not in pinned and k != keep]
    for k in free[: max(0, len(free) + 1 - limit)]:
        del store[k]


def lookup(store: dict, key, build, limit: int = 64):
    """``build()``, cached in ``store`` under ``key`` within the current scope (see the module docstring)."""
    sk = scoped(key)
    if sk is None:
        return build()
    pinned = _pin_if_static_capture(store, sk)
    hit = store.get(sk)
    if hit is not None:
        return hold(hit, sk[0] is None)
    prune(store)
    _evict(store, pinned, limit, sk)
    out = store[sk] = build()
    return hold(out, False)  # made now: inside a capture its memory is the graph's pool (held by a handle only)


def lookup_inputs(store: dict, key, build, limit: int = 64, valid=None, alive=None, inputs=(), lineage=None):
    """``lookup`` for a table made from a per-item INPUT (a conditioning tensor, a pair): the table is the same for every call that reads that
    tensor unchanged, so a sampler's steps (eager) and a captured step's block calls share it. Scoped to the capture like every pack, except
    under ``static_inputs()``: then a capture serves from, and fills, the eager entries and a replay re-runs none of the producing kernels.

    An input is an activation: its storage address is reused as soon as it is freed, so (pointer, version) alone can name a different tensor
    with the same version. ``valid(entry)`` is asked of a hit (False: the entry belongs to another tensor, it is remade and replaced) and
    ``alive(entry)`` of every entry when the store is pruned (False: the tensor is gone, the entry is dropped).

    The scope is the capture itself, not ``capture_id()``: ``static_weights()`` says nothing about the inputs, so a capture under it alone
    still makes its own tables and a replay remakes them.

    Refresh contract (``refresh_static_inputs``): a capture under ``static_inputs()`` served an eager entry records ``build`` with the entry, and
    a refresh writes a new ``build()`` into the entry's tensors. So ``build`` must derive the table from ``inputs`` (the per-item tensors the
    key names: their CURRENT contents) and the weights only -- never from an intermediate made by the caller before the lookup (that one
    belongs to the capture and is stale until the graph replays).

    ``lineage``: the key without the inputs' versions (their addresses, shapes, the weights...). A new entry of a lineage replaces the
    entries of its earlier versions: an input written in place has a new version, so the old key can never be hit again, yet its entry
    stayed while the input lived (FoldForge 5I28 OpenDDE n = 16: a fold's pair conditioning refilled in place kept both versions' 24 token-DiT
    pair biases, 2 x 11.3 GiB). A graph that read the old entry reads it from its refresh record (which holds the table) after the refill,
    or is stale by the ``static_inputs()`` contract."""
    if _STATIC_INPUTS[0]:
        cid = None
    else:
        cid = _graph_capture_id()
        if cid == 0:
            return build()
    if alive is not None:
        _INPUT_STORES[id(store)] = (store, alive)
    sk = (cid, key)
    # an eager-scope call inside a capture (static_inputs): the entry is pinned, and a hit read from the eager cache is recorded
    eager_in_capture = cid is None and _graph_capture_id() is not None
    pinned = _PINNED.setdefault(id(store), set())
    if eager_in_capture:
        pinned.add(sk)
    hit = store.get(sk)
    if hit is not None and (valid is None or valid(hit)):
        if eager_in_capture and (id(store), sk) not in _CAPTURE_BUILT:
            note_static_read((id(store), sk), hit, build, inputs)
        return hold(hit, cid is None)
    if not _STATIC_INPUTS[0]:
        prune(store, cid)
    if alive is not None:
        for k in [k for k, e in store.items() if not alive(e)]:
            del store[k]
            pinned.discard(k)  # its tensor is gone, so is every graph that read the entry
    if sk not in store:
        _evict(store, pinned, limit, sk)
    if lineage is not None:
        lines = _LINEAGES.setdefault(id(store), {})
        old = lines.get((cid, lineage))
        # (an entry a capture under static_inputs() read stays: that graph may still replay it -- e.g. a refill that only bumped the version)
        if old is not None and old != sk and old in store and old not in pinned:
            del store[old]
            _CAPTURE_BUILT.discard((id(store), old))
        lines[(cid, lineage)] = sk
        if len(lines) > 4 * limit:  # lineages whose entries are gone (their inputs freed)
            for k in [k for k, v in lines.items() if v not in store and v != sk]:
                del lines[k]
    out = store[sk] = build()
    if eager_in_capture:  # an eager-scope entry made inside a capture: its kernels are recorded, every replay remakes it (no refresh)
        _CAPTURE_BUILT.add((id(store), sk))
    elif _CAPTURE_BUILT:
        _CAPTURE_BUILT.discard((id(store), sk))
    return hold(out, False)  # made now: inside a capture its memory is the graph's pool (held by a handle only)


#: id(store) -> {(scope, lineage): the key of its newest entry} (``lookup_inputs``'s ``lineage``).
_LINEAGES: dict = {}

#: id(store) -> (the store, its ``alive(entry)``): every per-input cache ``lookup_inputs`` filled with an ``alive`` test, for
#: ``release_dead_inputs``.
_INPUT_STORES: dict[int, tuple[dict, object]] = {}


def release_dead_inputs() -> int:
    """Drop every per-input cache entry (``lookup_inputs`` / ``hoisted_table``: a hoisted pair bias, conditioning tables) whose input
    tensor is gone; returns how many were dropped. ``lookup_inputs`` drops them only at its next call on the same store, so a table of a
    finished item (a token DiT's hoisted pair bias: [heads x blocks, S, S], 11.3 GiB for OpenDDE's 3968 structural tokens) otherwise
    outlives its item until the next item reaches the same module -- through the whole next trunk. A graph that read such an entry read
    its input too, so it is no longer replayable either (as at ``lookup_inputs``'s own drop). Cheap (a weak-reference test per entry);
    called by :class:`~miniworld_engine.modules.PairformerBlock` on every eager call, and by callers between items."""
    dropped = 0
    for store, alive in list(_INPUT_STORES.values()):
        dead = [k for k, e in store.items() if not alive(e)]  # type: ignore[operator]
        pinned = _PINNED.get(id(store))
        for k in dead:
            del store[k]
            if pinned is not None:
                pinned.discard(k)
        dropped += len(dead)
    return dropped


#: (scope, (tag, ((pointer, version, dtype, shape), ...))) -> (weak references to the parameters, the forms). Also the Transition's ``Ws^T``
#: (``transition/cuda/_weights.py``) and what FoldForge clears before it loads a model.
_FORMS: dict = {}


#: id -> form: the forms a capture read from the eager scope (``capture_reads_eager``): held for the life of the process, so a newer version
#: of the parameters replacing the entry never frees memory a captured graph still reads. (A dict: a form read by many captures is held once.)
_HELD_BY_CAPTURES: dict[int, object] = {}


# ------------------------------------------------------------------------------------------- what a captured graph reads stays allocated
#: id -> entry: the cache entries captures read while no ``static_inputs()`` handle was open (plain ``static_weights()``,
#: ``set_static_weights``): held until ``clear_weight_caches``.
_HELD_UNOWNED: dict[int, object] = {}


def hold(entry, eager: bool = True):
    """The capture running now reads ``entry`` (an engine cache entry: a tensor, or a tuple / object holding tensors): keep it allocated for
    as long as the graph can be replayed. Returns ``entry``; a no-op outside a capture.

    The rule (the memory side of the R26 contract "capture once per shape, refresh + replay per item"): a cache may evict, prune or replace
    an entry at any later call -- another shape's capture fills the same stores (an LRU of conditioning chunks, a per-scope pack store, a
    lineage replacement) -- and an entry's memory then went back to the allocator, and after ``torch.cuda.empty_cache()`` to the driver, while
    a graph that read it was still replayed (an illegal address, NaN, or another tensor's values). So every entry a capture is served is
    held by:

    - each ``static_inputs()`` handle open at the capture (:class:`StaticTables`): released with the handle (``release()`` or dropped) --
      keep the handle for as long as its graph is replayed, as for its refresh;
    - with no handle open, the process (``eager``: an entry of the eager scope, which a later eager call or capture may evict), until
      ``clear_weight_caches``. An entry made inside a capture without a handle (its own scope) is not held: it lives in the graph's
      private memory pool, which the graph keeps.

    A hold never extends an entry past its handle: a store that drops the entry while the handle lives frees nothing the graph reads, and
    once the handle goes the entry lives exactly as long as the store keeps it."""
    if _graph_capture_id() is None:
        return entry
    if _HANDLES:
        for handle in _HANDLES:
            handle._held[id(entry)] = entry
    elif eager:
        _HELD_UNOWNED[id(entry)] = entry
    return entry


# ------------------------------------------------------------------------------------------------------------- the weight-cache registry
#: name -> store: every module-level cache whose entries are derived from weights or per-item inputs (keyed on tensor addresses / versions /
#: identities). A cache registers itself where it is defined (``_PACKS = _capture.weight_cache("family.packs")``), so
#: ``clear_weight_caches`` finds it without a hand list; tests/cpu/test_capture_api.py fails on a module-level dict in the kernel sources
#: that is neither registered nor listed there as not weight-derived.
_WEIGHT_CACHES: dict[str, dict] = {}

#: Per-instance cache attributes (``module.__dict__[name]``) holding weight-derived forms: dropped by ``clear_weight_caches(*modules)``.
_INSTANCE_CACHE_ATTRS: set[str] = set()


def weight_cache(name: str, store: dict | None = None) -> dict:
    """Register ``store`` (a new empty dict by default) as the weight-derived cache ``name`` and return it. Registering a name again (a module
    reload) replaces the earlier store."""
    store = {} if store is None else store
    _WEIGHT_CACHES[name] = store
    return store


def instance_cache_attr(name: str) -> str:
    """Register ``name`` as a per-instance cache attribute (a weight-derived form kept in a module's ``__dict__``); returns ``name``."""
    _INSTANCE_CACHE_ATTRS.add(name)
    return name


def weight_caches() -> dict[str, dict]:
    """The registered weight-derived caches by name (of the kernel modules imported so far: a module not imported has filled none)."""
    return dict(_WEIGHT_CACHES)


def clear_weight_caches(*modules: torch.nn.Module) -> None:
    """Drop every weight- or input-derived cache of the engine: each registered store (``weight_cache``: packs, hoisted tables, the parameter
    forms, the TF32 operand cache) and, for the given modules and their submodules, the per-instance cache attributes. Call it before weights
    are replaced by new tensors (a second model, a test that builds many): a new tensor at the address of a freed one must not be served the
    old one's form. Graphs captured before the call may read the dropped entries: drop them first (replaying one afterwards is undefined):
    the entries captures read (``hold``: in the ``static_inputs()`` handles and process-wide) are let go too. Forms held for live captures
    by ``capture_reads_eager`` stay held."""
    for store in _WEIGHT_CACHES.values():
        store.clear()
    _PINNED.clear()
    _CAPTURE_BUILT.clear()
    _HELD_UNOWNED.clear()
    for handle in list(_LIVE_HANDLES):
        handle._held.clear()
    for module in modules:
        for sub in module.modules():
            for attr in _INSTANCE_CACHE_ATTRS:
                sub.__dict__.pop(attr, None)


weight_cache("capture.forms", _FORMS)
weight_cache("capture.input_lineages", _LINEAGES)


def param_forms(tag, params, build, *, inputs_scope: bool = False, capture_reads_eager: bool = False):
    """``build()`` -- the kernel-dtype forms of some parameters (casts, contiguous copies, transposes, folds) -- cached while the very same
    parameter tensors are alive and unmodified, scoped to the capture like every weight pack (``lookup``): an eager call and a captured
    step under ``static_weights()`` share the entry, so a replay re-runs none of the cast kernels; a capture without it records them once
    and every replay casts the weights as they are then.

    The reason it exists: with fp32 master parameters over bf16 activations every call casts every weight, ~10 small kernels per
    block per call (about 6-12 us), which made inference of the small blocks 10-50 % slower than with bf16 parameters. The key is
    (pointer, version, dtype, shape) of each parameter and an entry serves only the very tensors it was made from (weak references:
    a freed storage's address reused by another tensor at the same version misses). An entry lives as long as its parameters and is
    replaced when they are cached at a newer version -- never evicted by count: a captured graph reads the entry it hit, and a size cap
    that freed a live entry freed memory a replay still read (the Transition's 64-entry ``Ws^T`` cache faulted Boltz-2's trunk graph so).
    A fused optimizer step does not bump ``_version`` -- as for every other pack, call ``torch.autograd.graph.increment_version(params)``
    after it. Inference tensors carry no version counter: not cached.

    ``inputs_scope``: ``static_inputs()`` (whose contract covers the weights too) also shares the eager entry with a capture, as
    ``static_weights()`` does for every call. ``capture_reads_eager``: a capture is served the eager entry its warm-up made, without
    ``static_weights()`` -- for the inference-only forms (grad off), the contract ``kernels._tf32.tf32_cached`` has for the portable kernels;
    such an entry is then held for good. ``tag`` names the form (any hashable)."""
    if torch.compiler.is_compiling():
        # A cache keyed on tensor identity cannot be traced (Dynamo died on the weak references: FoldForge's compiled Protenix-v2
        # sampler, AdaLN inference). Traced, the forms are computed in the graph; the opaque ops call this at run time and cache.
        return build()
    params = tuple(params)
    try:
        key = (tag, tuple((p.data_ptr(), p._version, p.dtype, tuple(p.shape)) for p in params))
    except RuntimeError:
        return build()
    if capture_reads_eager and _graph_capture_id() is not None:
        eager = _FORMS.get((None, key))
        if eager is not None and all(r() is p for r, p in zip(eager[0], params, strict=True)):
            _HELD_BY_CAPTURES[id(eager[1])] = eager[1]  # the graph reads it from now on: never freed under it
            return hold(eager[1])
    sk = (None, key) if inputs_scope and _STATIC_INPUTS[0] else scoped(key)
    if sk is None:
        return build()
    hit = _FORMS.get(sk)
    if hit is not None and all(r() is p for r, p in zip(hit[0], params, strict=True)):
        return hold(hit[1], sk[0] is None)
    prune(_FORMS)
    ids = [id(p) for p in params]
    for k in [
        k
        for k, (refs, _) in _FORMS.items()
        if any(r() is None for r in refs) or (k[0] == sk[0] and k[1][0] == tag and [id(r()) for r in refs] == ids)
    ]:
        del _FORMS[k]
    value = build()
    _FORMS[sk] = (tuple(weakref.ref(p) for p in params), value)
    return hold(value, False)  # made now: inside a capture its memory is the graph's pool (held by a handle only)


def hoisted_table(store: dict, tensor: torch.Tensor, key, build, limit: int = 64, also=()):
    """The table ``build()`` makes from the per-item input ``tensor``, once per (tensor, version, ``key``) (``lookup_inputs``): the entry keeps a weak
    reference to the tensor, so a new tensor that reuses a freed address is not served the old one's table, and entries of freed tensors are dropped.
    ``also``: the other caller tensors ``build`` reads (a key mask, conformer ids; ``key`` names them): a refresh checks their layout too.

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
        inputs=(tensor, *also),
    )
    return entry[2]


# ------------------------------------------------------------------------------------------- refreshing hoisted tables (FoldForge R26)
class StaleStaticInputs(RuntimeError):
    """A refresh cannot rewrite a recorded hoisted table: the weights it was made from changed (version / object), its input's layout
    changed (shape, strides, dtype, device, address), the capture that read it was not under ``static_weights()``, or a rebuilt table no
    longer has the recorded table's shape / dtype. The graph that read the table must be captured again."""


#: The module whose ``forward`` runs now (``BackendModule.forward`` sets it on every eager call): the owner of the hoisted tables a capture
#: records in it (``refresh_static_inputs(root)`` refreshes the tables of the modules under ``root``).
CALLER: list = [None]

#: (id(store), scoped key) of the eager-scope entries made INSIDE a capture (``static_inputs()``): their kernels are recorded, every replay
#: remakes them, so a capture that reads one records nothing to refresh.
_CAPTURE_BUILT: set = set()

#: The ``static_inputs()`` handles open now: each collects the tables the captures inside its block read.
_HANDLES: list = []


class _Read:
    """One hoisted table a capture read from the eager cache: the tensors the graph reads, how to make them again, and what they were made from."""

    __slots__ = ("__weakref__", "build", "holders", "inputs", "owner", "params", "table", "unsafe")


def _layout(t: torch.Tensor) -> tuple:
    return (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype, t.device)


def _version_of(p: torch.Tensor) -> int | None:
    try:
        return p._version
    except RuntimeError:  # an inference tensor has no version counter: identity only
        return None


def _params_of(owner) -> tuple:
    return tuple((weakref.ref(p), _version_of(p)) for p in owner.parameters())


#: (id(store), key) -> a weak reference to the ``_Read`` of every hoisted table a capture under ``static_inputs()`` read from the eager cache
#: (so a replay reads it as it is then, without its producing kernels). The record itself is held by the ``static_inputs()`` handles that were
#: open when it was made (``StaticTables``), or by ``_UNOWNED`` when none was (``set_static_inputs`` alone). A record holds its table, its
#: inputs and the build closure: held globally, it kept every item's tables alive for good -- FoldForge, which does not refresh, captures one
#: sampler graph per fold, and the records kept each finished fold's pair conditioning (3.75 GiB) and 48 token-DiT pair biases (22.6 GiB)
#: through the next fold's trunk (5I28 OpenDDE n = 16: out of memory at the structural-token refiner, 2026-11-13). Registered:
#: ``clear_weight_caches`` drops the records with the stores.
_READS: dict = weight_cache("capture.static_reads")

#: (id(store), key) -> ``_Read``: the records made with no ``static_inputs()`` handle open (``set_static_inputs(True)`` alone): held until
#: ``release_static_inputs`` / ``clear_weight_caches``, as every record was before the handles owned them.
_UNOWNED: dict = weight_cache("capture.static_reads_unowned")

#: id(module) -> weak reference to a module whose tables a capture read, recorded only in ``static_inputs()`` handles that have all been
#: dropped: a ``refresh_static_inputs`` of it would rewrite nothing (the records went with the handles) and the graph would replay the old
#: item's tables, so it raises instead. Cleared for a module when a capture records its tables again.
_ABANDONED: dict = weight_cache("capture.abandoned_refreshes")

#: Every ``StaticTables`` handle alive (a global ``release_static_inputs`` takes the records out of each).
_LIVE_HANDLES: weakref.WeakSet = weakref.WeakSet()


def _record(rkey) -> _Read | None:
    """The live record of ``rkey`` (None: none was made, or its holders are gone)."""
    ref = _READS.get(rkey)
    return None if ref is None else ref()


def note_static_read(rkey, table, build, inputs=()) -> None:
    """Record that the capture running now reads the hoisted ``table`` from the eager cache (``lookup_inputs`` does it for its stores; a cache of
    its own calls it on a hit served to a capture from the eager scope). ``build()`` makes the table again from the CURRENT contents of
    ``inputs`` (the per-item tensors it is made from: their layout is checked at a refresh) and the weights; ``rkey`` names the entry (one
    record per entry, whichever capture read it first). The record lives as long as one of the ``static_inputs()`` handles open now."""
    r = _record(rkey)
    if r is None or r.table is not table:
        r = _Read()
        r.table, r.build, r.holders = table, build, 0
        r.inputs = tuple((t, _layout(t)) for t in inputs if t is not None)
        owner = CALLER[0]
        r.owner = None if owner is None else weakref.ref(owner)
        r.params = () if owner is None else _params_of(owner)
        # the table's build reads weight packs: without static_weights() a capture packs its own (made by its kernels, valid only after a
        # replay), so an eager refresh before the replay would read garbage
        r.unsafe = None if _STATIC[0] else "the capture was not under static_weights() (its weight packs are the graph's own)"
        _READS[rkey] = weakref.ref(r)
    if r.owner is not None:
        _ABANDONED.pop(id(r.owner()), None)
    if not _HANDLES:
        _UNOWNED[rkey] = r
    for handle in _HANDLES:
        if handle._records.get(rkey) is not r:
            handle._records[rkey] = r
            r.holders += 1


def _handle_dropped(records: dict) -> None:
    """A ``StaticTables`` handle was collected: its records go with it unless another handle (or ``_UNOWNED``) holds them; the modules of the
    records that go are marked (``_ABANDONED``) so a refresh of them raises instead of rewriting nothing."""
    for rkey, r in records.items():
        r.holders -= 1
        if r.holders <= 0 and _record(rkey) is r and _UNOWNED.get(rkey) is not r:
            owner = None if r.owner is None else r.owner()
            if owner is not None:
                _ABANDONED[id(owner)] = weakref.ref(owner)
    records.clear()


def _copy_into(old, new) -> None:
    """Write the rebuilt ``new`` into the recorded ``old`` (same structure): tensors element-wise into the recorded storage (a tensor at the same
    address is an input the entry holds: skipped), Python values must be equal, other objects (weak references) are skipped."""
    if isinstance(old, torch.Tensor):
        if not isinstance(new, torch.Tensor):
            raise StaleStaticInputs(f"refresh: a table slot was a tensor, the rebuild gave {type(new).__name__}")
        if old.data_ptr() == new.data_ptr():
            return
        if old.shape != new.shape or old.dtype != new.dtype:
            raise StaleStaticInputs(
                f"refresh: the rebuilt table is {tuple(new.shape)} {new.dtype}, the captured one {tuple(old.shape)} {old.dtype}"
            )
        old.copy_(new)
    elif isinstance(old, (tuple, list)):
        if not isinstance(new, (tuple, list)) or len(new) != len(old):
            raise StaleStaticInputs("refresh: the rebuilt table has another structure")
        for a, b in zip(old, new, strict=True):
            _copy_into(a, b)
    elif isinstance(old, dict):
        if not isinstance(new, dict) or new.keys() != old.keys():
            raise StaleStaticInputs("refresh: the rebuilt table has another structure")
        for k in old:
            _copy_into(old[k], new[k])
    elif isinstance(old, (bool, int, float, str)) and old != new:
        raise StaleStaticInputs(f"refresh: a table value changed ({old!r} -> {new!r}); the graph baked the old one")


def _stale(r: _Read) -> str | None:
    if r.unsafe is not None:
        return r.unsafe
    for ref, version in r.params:
        p = ref()
        if p is None:
            return "a weight of the module that made it is gone"
        if version is not None and p._version != version:
            return "a weight of the module that made it changed after the capture"
    for t, layout in r.inputs:
        if _layout(t) != layout:
            return f"its input's layout changed after the capture ({layout[1:]} -> {_layout(t)[1:]})"
    return None


def _selected(root, keys) -> list:
    keys = [k for k in keys if _record(k) is not None]
    if root is None:
        return keys
    under = {id(m) for m in root.modules()}
    out = []
    for k in keys:
        owner = cast(_Read, _record(k)).owner
        if owner is not None and id(owner()) in under:
            out.append(k)
    return out


def _abandoned_under(root) -> str | None:
    """The first module (under ``root``; any when None) whose recorded tables went with their dropped ``static_inputs()`` handles."""
    for key, ref in list(_ABANDONED.items()):
        module = ref()
        if module is None or id(module) != key:
            del _ABANDONED[key]
            continue
        if root is None or any(m is module for m in root.modules()):
            return type(module).__name__
    return None


def refresh_static_inputs(root: torch.nn.Module | None = None, *, tables: StaticTables | None = None) -> int:
    """Make every hoisted per-item table that a capture under ``static_inputs()`` reads from the eager cache again, from the CURRENT contents of
    its input tensors, INTO the tensors the graph reads (same storage, no reallocation of the tables; the rebuild's temporaries come and go on
    the current stream, one table at a time). Call it after writing a new item's inputs into the graph's static input tensors (in place) and
    before replaying: the replay then equals a fresh capture -- and eager -- on the new inputs, bitwise.

    ``root``: only the tables made by modules under ``root`` (``root.modules()``); ``tables``: only those a ``static_inputs()`` block's
    captures read (the handle it yielded). Neither: every recorded table. Returns the number of tables rewritten. Every selected table is
    checked first and :class:`StaleStaticInputs` is raised -- nothing written -- when one cannot be refreshed (its module's weights changed
    or were replaced, its input's layout changed, the capture was not under ``static_weights()``, or -- without ``tables`` -- the records of
    a module under ``root`` went with their dropped ``static_inputs()`` handle: keep the handle for as long as its graph is refreshed).

    Runs eagerly on the current stream; it may itself be captured (a graph of the rebuilds, replayed before the step graph): the rebuilds
    allocate, read the inputs and write the tables with no host synchronisation."""
    if tables is None:
        gone = _abandoned_under(root)
        if gone is not None:
            raise StaleStaticInputs(
                f"refresh_static_inputs: the tables a capture read for {gone} were recorded in a static_inputs() handle that has been dropped"
                " (the records go with it); keep the handle (`with static_inputs() as tables`) while its graph is refreshed, or capture again"
            )
    keys = _selected(root, list(_READS) if tables is None else list(tables._records))
    reads = [cast(_Read, _record(k)) for k in keys]
    for r in reads:
        why = _stale(r)
        if why is not None:
            raise StaleStaticInputs(f"refresh_static_inputs: a hoisted table cannot be refreshed: {why}; capture the graph again")
    for r in reads:
        _copy_into(r.table, r.build())
    return len(reads)


def release_static_inputs(root: torch.nn.Module | None = None, *, tables: StaticTables | None = None) -> int:
    """Drop the refresh records (``refresh_static_inputs``'s selection); returns how many. A record holds its table, its inputs and the
    capture's build closure: drop it with the graph that read the table (a record also goes with the ``static_inputs()`` handles that hold
    it; ``clear_weight_caches`` drops them all)."""
    keys = _selected(root, list(_READS) if tables is None else list(tables._records))
    for k in keys:
        r = _record(k)
        del _READS[k]
        _UNOWNED.pop(k, None)
        for handle in list(_LIVE_HANDLES):
            if handle._records.get(k) is r:
                del handle._records[k]
    # the entries the captures read (``hold``) go with the handle's release -- or, for a release of everything, with every handle's
    if tables is not None:
        tables._held.clear()
    elif root is None:
        _HELD_UNOWNED.clear()
        for handle in list(_LIVE_HANDLES):
            handle._held.clear()
    return len(keys)


class StaticTables:
    """The hoisted tables the captures inside one ``static_inputs()`` block read from the eager cache (yielded by ``static_inputs()``). It
    holds their refresh records, and every engine cache entry those captures read (``hold``: weight packs, conditioning chunks, hoisted
    tables), so no later eviction frees memory the graphs read: both live as long as the handle (or until released) -- keep it for as
    long as the graphs captured inside the block are replayed or refreshed."""

    def __init__(self) -> None:
        self._records: dict = {}
        self._held: dict[int, object] = {}
        _LIVE_HANDLES.add(self)
        weakref.finalize(self, _handle_dropped, self._records)

    def __len__(self) -> int:
        return sum(1 for k, r in self._records.items() if _record(k) is r)

    def held(self) -> tuple:
        """The engine cache entries the captures inside the block read and this handle keeps allocated (``hold``): for accounting."""
        return tuple(self._held.values())

    def refresh(self, root: torch.nn.Module | None = None) -> int:
        """``refresh_static_inputs(root, tables=self)``."""
        return refresh_static_inputs(root, tables=self)

    def release(self) -> int:
        """``release_static_inputs(tables=self)``: the refresh records and the held entries (drop the graphs first)."""
        return release_static_inputs(tables=self)


def signature(root: torch.nn.Module, *tensors: torch.Tensor | None) -> tuple:
    """A hashable key of a captured graph's static layout: ``root``'s identity and each input's (shape, strides, dtype, device) -- what must be
    equal for a graph captured on one set of static inputs to serve another item through ``refresh_static_inputs``. Host metadata only (no
    device read, no parameter walk: a changed weight is caught by the refresh itself, which raises :class:`StaleStaticInputs`)."""
    return (id(root), tuple(None if t is None else (tuple(t.shape), t.stride(), t.dtype, t.device) for t in tensors))

"""The public CUDA-graph capture and weight-cache API.

``miniworld_engine`` is a namespace package (no ``__init__.py``: the prebuilt kernel wheels install into it), so nothing is exported from
the package root; this module is the public home of what a caller that captures CUDA graphs or replaces weights needs::

    from miniworld_engine.capture import clear_weight_caches, static_inputs, static_weights

    with static_weights(), static_inputs():      # an inference graph over one item: weights and inputs fixed between replays
        graph = capture(model_step)

    clear_weight_caches(model)                   # before loading a second set of weights (drop old graphs first)

``static_weights()``
    Weights stay fixed between graph replays (inference graphs only): a capture inside it serves from, and fills, the eager weight-pack
    caches, so a replay runs none of the packing / casting kernels. An in-place weight update after the capture is then NOT seen by
    replays: never around a training graph.
``static_inputs()``
    Stronger: the conditioning / pair inputs of the hoisted per-item tables (a block's conditioning table, a pair's bias) also keep their
    contents for the life of the graph, so a replay re-runs none of their producing kernels -- unless the caller refreshes the tables
    (below) after refilling those inputs in place. Yields a :class:`StaticTables` handle (the tables its captures read).
``clear_weight_caches(*modules)``
    Drop every weight- or input-derived cache of the engine (each store registered with ``kernels._capture.weight_cache``: the
    parameter forms, weight packs, hoisted tables, the TF32 operand cache, the refresh records) and, for the given modules and their
    submodules, the per-instance cache attributes. Call it before weights are replaced by new tensors (a second model, a test that builds
    many): a new tensor allocated at the address of a freed one must not be served the old one's form. Graphs captured before the call may
    read the dropped entries: drop them first.
``weight_caches()``
    The registered stores by name (inspection; a kernel module not imported yet has registered nothing and filled nothing).

**One graph per shape, many items (FoldForge R26).** A sampler graph captured under ``static_weights()`` + ``static_inputs()`` reads the
hoisted per-item tables (the token DiT's pair bias, the atom blocks' conditioning tables / windowed pair bias / slid-edge tables, the
bias-only block's attention weights) from the eager cache: they were made in the warm-up and the replay runs none of their kernels. To
serve the next item with the SAME graph::

    with static_weights(), static_inputs() as tables:      # once per shape
        warm_up(); warm_up()                                # eager, on the static inputs of item 1
        with torch.cuda.graph(g): step()
    ...
    for item in items:                                      # every later item of that shape
        copy_inputs_into_static_buffers(item)               # in place (copy_): same tensors, same storage
        refresh_static_inputs(model)                        # or tables.refresh(model): the tables, rebuilt into the graph's buffers
        g.replay()                                          # == a fresh capture == eager on the item, bitwise

``refresh_static_inputs(root=None, *, tables=None)``
    Rebuild every hoisted table such a capture reads, from the CURRENT contents of its inputs, into the very tensors the graph reads (no
    reallocation; temporaries one table at a time on the current stream). ``root`` limits it to the tables made by modules under ``root``,
    ``tables`` to those of one ``static_inputs()`` block. Returns how many tables it rewrote. Raises :class:`StaleStaticInputs` (and writes
    nothing) when a table cannot be refreshed: its module's weights changed or were replaced, an input's layout changed, or the capture was
    not under ``static_weights()``. It runs eagerly (once per item, before the replay) and may itself be captured into a graph replayed in
    front of the step graph.
``release_static_inputs(root=None, *, tables=None)``
    Drop the refresh records (each holds its table and inputs alive): with the graph that read them.
``signature(root, *inputs)``
    A hashable key for a per-shape graph cache: ``root``'s identity and each input's shape / strides / dtype / device (metadata only).

Only the tables the graph reads from the eager cache need the refresh: a table the capture made itself (its input was made inside the
captured step, or did not exist at the warm-up) is recorded in the graph and every replay remakes it. Everything else a captured step
computes from its inputs is recomputed by the replay.

Semantics are those of ``miniworld_engine.kernels._capture`` (its module docstring: capture scopes, the fused-optimizer
``increment_version`` rule); this module is the stable name for them.
"""

from __future__ import annotations

from miniworld_engine.kernels._capture import (
    StaleStaticInputs,
    StaticTables,
    clear_weight_caches,
    refresh_static_inputs,
    release_static_inputs,
    signature,
    static_inputs,
    static_weights,
    weight_caches,
)

__all__ = [
    "StaleStaticInputs",
    "StaticTables",
    "clear_weight_caches",
    "refresh_static_inputs",
    "release_static_inputs",
    "signature",
    "static_inputs",
    "static_weights",
    "weight_caches",
]

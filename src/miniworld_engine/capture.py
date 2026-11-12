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
    contents for the life of the graph, so a replay re-runs none of their producing kernels. A sampler that refills those tensors in
    place between replays must not set it.
``clear_weight_caches(*modules)``
    Drop every weight- or input-derived cache of the engine (each store registered with ``kernels._capture.weight_cache``: the
    parameter forms, weight packs, hoisted tables, the TF32 operand cache) and, for the given modules and their submodules, the
    per-instance cache attributes. Call it before weights are replaced by new tensors (a second model, a test that builds many): a new
    tensor allocated at the address of a freed one must not be served the old one's form. Graphs captured before the call may read the
    dropped entries: drop them first.
``weight_caches()``
    The registered stores by name (inspection; a kernel module not imported yet has registered nothing and filled nothing).

Semantics are those of ``miniworld_engine.kernels._capture`` (its module docstring: capture scopes, the fused-optimizer
``increment_version`` rule); this module is the stable name for them.
"""

from __future__ import annotations

from miniworld_engine.kernels._capture import clear_weight_caches, static_inputs, static_weights, weight_caches

__all__ = ["clear_weight_caches", "static_inputs", "static_weights", "weight_caches"]

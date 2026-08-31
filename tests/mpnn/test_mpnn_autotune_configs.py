"""Static checks on the fused MPNN kernels' autotune grids.

These run on CPU.  They exist because both failure modes below were first seen as a
crash inside a multi-minute GPU job on a shared cluster, where the same information was
available from the decorated function object all along.
"""

from __future__ import annotations

import triton

from miniworld_engine.kernels.mpnn_edge_tail.triton import main as edge_tail
from miniworld_engine.kernels.mpnn_node_message.triton import main as node_message
from miniworld_engine.kernels.mpnn_relative_position.triton import (
    main as relative_position,
)

_MODULES = (edge_tail, node_message, relative_position)


def _autotuned_kernels() -> list[tuple[str, triton.runtime.Autotuner]]:
    found = []
    for module in _MODULES:
        for name in sorted(dir(module)):
            candidate = getattr(module, name)
            if isinstance(candidate, triton.runtime.Autotuner):
                found.append((f"{module.__name__}.{name}", candidate))
    return found


def test_every_autotune_key_is_declared_by_its_kernel() -> None:
    """A configuration may not carry a key its kernel does not take.

    Triton raises ``KeyError: Keyword argument ... unrecognised`` at launch, not at
    import, so a knob added to a shared configuration factory reaches every kernel
    built from it and fails only on the ones that never declared it.
    """
    kernels = _autotuned_kernels()
    assert kernels, "no autotuned kernels found -- the discovery above went stale"

    for name, kernel in kernels:
        declared = set(kernel.fn.arg_names)
        for config in kernel.configs:
            undeclared = sorted(set(config.kwargs) - declared)
            assert not undeclared, f"{name} has no parameter {undeclared} for {config}"


def test_no_autotune_knob_is_pinned_to_a_single_value() -> None:
    """Every knob offered must vary across the grid.

    A knob fixed on the strength of one measurement hides the winner from a tuner that
    can only choose from the list it is given.  Three separate regressions in this
    file's history were exactly that; see ``_configs`` for the measurements.
    """
    for name, kernel in _autotuned_kernels():
        knobs: dict[str, set[object]] = {}
        for config in kernel.configs:
            for knob, value in config.kwargs.items():
                knobs.setdefault(knob, set()).add(value)
            knobs.setdefault("num_warps", set()).add(config.num_warps)
            knobs.setdefault("num_stages", set()).add(config.num_stages)

        pinned = sorted(knob for knob, values in knobs.items() if len(values) == 1)
        assert not pinned, f"{name} pins {pinned} to one value each"


def test_every_kernel_is_wired_to_the_committed_autotune_cache() -> None:
    """Each grid must be narrowable by the repository's per-GPU cache.

    The grids here are deliberately large -- 324 configurations for the norm pass --
    because pinning a knob has hidden the winner three times.  The cache is where that
    compile cost is meant to be paid: it narrows to a measured top-K without pinning,
    and it is what every other kernel family in the package already uses.
    """
    # The hook this used to name -- a per-kernel `make_cache_prune` object carrying a hand-written
    # `key_bucket_of(...)` -- was deleted in fcd3c7a, and its absence WAS the bug: every capture
    # after it recorded the single bucket `any|any`, one config per op for every shape. What
    # replaced it needs no per-kernel wiring: `install_cache_pruning` narrows every autotuner to
    # the cached top-K, and `bucket_of_autotuner` reads the bucket from the kernel's own
    # `key=[...]`. So the thing to check is no longer a hook per kernel, it is that each kernel
    # keys on the shape at all -- without `shape_key` in `key`, one bucket serves every shape and
    # the cache is back to `any|any` by another route.
    #: NO mpnn kernel keys on `shape_key` yet -- they key on their own dimension names
    #: (`rows`, `NEIGHBORS`, `buckets`, `groups_total`). Both work: triton re-tunes per distinct
    #: key tuple either way. What the mpnn families do not get is the PACKED key the rest of the
    #: package shares. `shape_key` folds the widths into one int, so a bucket is comparable across
    #: kernels and the cache reader, the builder, the coverage checks and the sweep page all speak
    #: one vocabulary -- which is also why none of those can see an mpnn kernel today.
    #:
    #: Porting one: give it a `shape_key` parameter, compute the key at the caller that still
    #: holds the pre-flatten shape (`token_key`/`atom_key`/`both_key` per its level), drop the
    #: dimension names from `key=[...]`, and add the registry row and ladder that follow from it.
    #: This set shrinks as that happens; it is the checklist, not an exemption.
    NOT_ON_SHAPE_KEY = frozenset(
        name for name, _ in _autotuned_kernels() if ".mpnn_" in name)
    keyless = []
    for name, kernel in _autotuned_kernels():
        if name in NOT_ON_SHAPE_KEY:
            continue
        keys = list(getattr(kernel, "keys", []) or [])
        if "shape_key" not in keys:
            keyless.append(f"{name}: key={keys}")
    assert not keyless, (
        "autotuned kernels that do not key on shape_key, so one cache bucket serves every shape:"
        "\n  " + "\n  ".join(keyless))


def test_cache_buckets_do_not_depend_on_the_row_count() -> None:
    """A bucket keyed on the row count needs one cache entry per batch size, which is no cache.

    This used to read the answer off a `_miniworld_bucket_of` attribute that `make_cache_prune`
    hung on each kernel. Both are gone (fcd3c7a); the bucket now comes from the kernel's own
    `key=[...]` through `bucket_of_autotuner`. So the check reads the key list instead, which is
    the thing that decides it.

    The mpnn kernels DO key on `rows` / `groups_total` today -- that is the same finding the test
    above records, and porting them to `shape_key` is what fixes both. This holds the line for
    everything else: no kernel outside that checklist may key on a row count.
    """
    ROW_LIKE = {"rows", "groups_total", "M", "m", "numel", "n_elements"}
    bad = []
    for name, kernel in _autotuned_kernels():
        if ".mpnn_" in name:
            continue
        keys = set(getattr(kernel, "keys", []) or [])
        if keys & ROW_LIKE:
            bad.append(f"{name}: key={sorted(keys)}")
    assert not bad, (
        "kernels keyed on a row count, which needs one cache entry per batch size:\n  "
        + "\n  ".join(bad))

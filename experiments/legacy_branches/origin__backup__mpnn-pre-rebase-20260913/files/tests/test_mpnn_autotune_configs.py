"""Static checks on the fused MPNN kernels' autotune grids.

These run on CPU.  They exist because both failure modes below were first seen as a
crash inside a multi-minute GPU job on a shared cluster, where the same information was
available from the decorated function object all along.
"""

from __future__ import annotations

import triton

from miniworld_kernels.kernels.mpnn_edge_tail.triton import main as edge_tail
from miniworld_kernels.kernels.mpnn_node_message.triton import main as node_message
from miniworld_kernels.kernels.mpnn_relative_position.triton import (
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
    seen: dict[str, str] = {}
    for name, kernel in _autotuned_kernels():
        prune = getattr(kernel, "early_config_prune", None)
        op = getattr(prune, "_miniworld_op", None)
        assert op, f"{name} has no make_cache_prune hook, so it re-tunes every process"
        assert op not in seen, f"{name} and {seen[op]} both tune as op {op!r}"
        seen[op] = name


def test_cache_buckets_do_not_depend_on_the_row_count() -> None:
    """A bucket keyed on the row count would need one cache entry per batch size.

    Row count is what the ``BLOCK_M``/``TILES``/``GROUPS`` knobs exist to absorb, so it
    belongs in the autotune ``key`` (it is there) and not in the cache bucket.
    """
    row_like = ("rows", "chunk_rows", "groups_total", "chunk_groups", "row_offset")
    # Every name a bucket could plausibly read, so the call succeeds whatever it asks
    # for and the produced bucket string shows what it actually used.
    arguments = {
        "WIDTH": 128,
        "NEIGHBORS": 48,
        "DROPOUT": 1,
        "EDGE_WEIGHT_STRIDE": 128,
        "BUCKET_BLOCK": 128,
        **dict.fromkeys(row_like, 262144),
    }
    for name, kernel in _autotuned_kernels():
        bucket = kernel.early_config_prune._miniworld_bucket_of  # noqa: SLF001
        produced = bucket({}, arguments)
        leaked = sorted(dim for dim in row_like if f"{dim}=" in produced)
        assert not leaked, f"{name} buckets its cache on {leaked}: {produced!r}"

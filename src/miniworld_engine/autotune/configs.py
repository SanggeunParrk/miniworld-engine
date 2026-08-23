"""Autotune configs come from CSV. Nothing in this package generates them.

One CSV per op, named ``<op>.csv``, under the active config directory. Two formats are accepted
and told apart by the header.

MATERIALISED -- one row IS one config. Every column that is not ``num_warps`` / ``num_stages`` /
``maxnreg`` is a tile axis and must match the kernel's constexpr spelling exactly. This is the
right shape for a tuned result: a handful of specific winning configs.

    BLOCK_M1,BLOCK_N,num_warps,num_stages
    64,128,4,3
    128,128,8,4

GRID SPEC (header ``axis,values``) -- one row is one AXIS, and the configs are the cartesian
product. This is the right shape for a SEARCH SPACE, where the materialised form does not scale:
the generated sweep is 205,266 configs over 91 ops and its largest op alone is 15,552 rows, which
restate the same six value sets 15,552 times. As a spec that op is six rows, and the whole grid is
~550 rows rather than 205,266.

    axis,values
    BLOCK_M1,32 64 128 256
    BLOCK_N,32 64 128 256
    BLOCK_K,16 32 64
    GROUP_M,1 2 4 8 16 32
    num_warps,1 2 4 8 16 32
    num_stages,1 2 3 4 5 6 8 10 12
    slice,0-8000

``slice,<start>-<stop>`` is optional and takes a half-open range of the product, which is what
makes config-set sharding cheap -- a shard states which part of the grid it owns instead of
materialising it. Expansion is ``itertools.product`` over the axes in FILE ORDER and must stay
deterministic, since a slice names positions in that sequence.

A kernel declares only which op it is::

    @triton.autotune(configs=configs_for("layernorm_fwd_saveact_triton"), key=["N"])

``configs_for`` returns a live list, but selecting the set AFTER a kernel module has imported is
too late for that kernel: ``triton.Autotuner.__init__`` keeps the list it is handed only when the
list is non-empty, and substitutes its own ``[Config({})]`` otherwise -- refilling the original
then updates a list nobody reads, and the kernel dies at launch with a ``dynamic_func() missing
required positional arguments`` naming its tile axes. So the set must be chosen BEFORE the import.

Set ``MINIWORLD_CONFIG_DIR`` and the choice is made when this module loads, which is necessarily
before any kernel module. ``use_config_dir`` remains for callers that own the import order; it
raises if any op already registered empty.
"""

from __future__ import annotations

import csv
import itertools
import os
from pathlib import Path

import triton

_META = ("num_warps", "num_stages", "maxnreg")

#: op -> the live list handed to that op's autotuner.
_LISTS: dict[str, list] = {}
#: Ops that registered before a directory was selected. Triton has already dropped their list, so
#: no later refill can reach them -- they are unrecoverable within the process.
_STRANDED: set[str] = set()
_DIR: Path | None = None


def _read_spec(path: Path, rows: list[dict]) -> list:
    """Expand an ``axis,values`` GRID SPEC into the full cartesian product.

    A search grid written out one row per config is unusable at scale: the generated sweep is
    205,266 configs over 91 ops, and the largest single op is 15,552 rows -- 13 MB of CSV that no
    one can read, diff, or review, restating the same six value sets 15,552 times. The spec says
    the value sets once:

        axis,values
        BLOCK_M1,32 64 128 256
        BLOCK_N,32 64 128 256
        BLOCK_K,16 32 64
        GROUP_M,1 2 4 8 16 32
        num_warps,1 2 4 8 16 32
        num_stages,1 2 3 4 5 6 8 10 12

    Six rows for the same 15,552 configs, and the whole grid becomes ~550 rows instead of 205,266.

    ``slice`` is an optional pseudo-axis, ``slice,<start>-<stop>``, taking a half-open range of the
    product. That is what makes CONFIG-SET sharding cheap: a shard directory states which part of
    the grid it owns instead of materialising it, so 26 shard dirs cost kilobytes rather than the
    13 MB the materialised split took.

    Expansion order is ``itertools.product`` over the axes in FILE ORDER, and it must stay
    deterministic: a shard's ``slice`` names positions in this sequence, so reordering the rows of
    a spec silently re-cuts every shard.
    """
    spec, order = {}, []
    for i, row in enumerate(rows, start=2):
        axis = (row.get("axis") or "").strip()
        raw = (row.get("values") or "").strip()
        if not axis:
            continue
        if axis in spec:
            raise ValueError(f"{path}:{i}: axis {axis!r} declared twice")
        # `slice` is written "start-stop", so it needs `-` as a separator; a tile axis never does
        # and must not, or a typo would silently split one value into two.
        text = raw.replace("-", " ") if axis == "slice" else raw.replace("|", " ")
        try:
            spec[axis] = [int(v) for v in text.split()]
        except ValueError as exc:
            raise ValueError(f"{path}:{i}: {axis}: {exc}") from exc
        if not spec[axis]:
            raise ValueError(f"{path}:{i}: axis {axis!r} has no values")
        order.append(axis)

    rng = spec.pop("slice", None)
    if "slice" in order:
        order.remove("slice")
    for meta in ("num_warps", "num_stages"):
        if meta not in spec:
            raise ValueError(f"{path}: grid spec has no {meta} row")
    axes = [a for a in order if a not in _META]
    if not axes:
        raise ValueError(f"{path}: grid spec has no tile-axis row")

    out = []
    for combo in itertools.product(*(spec[a] for a in order)):
        v = dict(zip(order, combo))
        out.append(triton.Config({a: v[a] for a in axes}, num_warps=v["num_warps"],
                                 num_stages=v["num_stages"], maxnreg=v.get("maxnreg")))
    if rng is not None:
        # written as a single "start-stop" token, so it parses as two ints
        if len(rng) != 2:
            raise ValueError(f"{path}: slice must be 'start-stop', got {rng}")
        start, stop = rng
        out = out[start:stop]
        if not out:
            raise ValueError(f"{path}: slice {start}-{stop} selects nothing")
    return out


def _read(path: Path) -> list:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no config rows")
    if set(rows[0]) >= {"axis", "values"}:
        return _read_spec(path, rows)
    axes = [c for c in rows[0] if c and c not in _META]
    if not axes:
        raise ValueError(f"{path}: header has no tile-axis column")
    out = []
    for i, row in enumerate(rows, start=2):
        try:
            kwargs = {a: int(row[a]) for a in axes if row.get(a) not in (None, "")}
            warps = int(row["num_warps"])
            stages = int(row["num_stages"])
            maxnreg = int(row["maxnreg"]) if row.get("maxnreg") not in (None, "") else None
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{i}: {exc}") from exc
        out.append(triton.Config(kwargs, num_warps=warps, num_stages=stages, maxnreg=maxnreg))
    return out


def use_config_dir(directory, *, require_all: bool = True) -> dict[str, int]:
    """Point every op at ``<directory>/<op>.csv`` and refill the lists already handed out.

    Raises when a registered op has no rows and ``require_all``. An op with no configs does not
    fail at selection time -- Triton substitutes a single empty config, and the kernel then dies at
    launch with ``dynamic_func() missing 2 required positional arguments: 'BLOCK_M1', 'BLOCK_N'``,
    which names neither the op nor the missing CSV. Checking here turns that into one message that
    lists exactly which files to write.
    """
    global _DIR
    if _STRANDED:
        raise RuntimeError(
            f"{len(_STRANDED)} op(s) registered before a config directory was selected and have "
            f"already been given triton's substitute config: " + ", ".join(sorted(_STRANDED)[:8])
            + ("..." if len(_STRANDED) > 8 else "")
            + ". Set MINIWORLD_CONFIG_DIR before importing any kernel module.")
    _DIR = Path(directory)
    counts = {}
    for op, live in _LISTS.items():
        live[:] = _load(op)
        counts[op] = len(live)
    empty = sorted(op for op, n in counts.items() if not n)
    if empty and require_all:
        raise FileNotFoundError(
            f"{len(empty)} op(s) have no configs under {_DIR}: "
            + ", ".join(empty[:8]) + ("..." if len(empty) > 8 else ""))
    return counts


def _load(op: str) -> list:
    if _DIR is None:
        return []
    path = _DIR / f"{op}.csv"
    return _read(path) if path.is_file() else []


def configs_for(op: str) -> list:
    """The live config list for ``op``. Empty until a config directory is selected."""
    live = _LISTS.get(op)
    if live is None:
        live = _LISTS[op] = []
        live[:] = _load(op)
        if not live:
            _STRANDED.add(op)
    return live


def registered_ops() -> frozenset[str]:
    """Ops that asked for configs, i.e. every op a CSV set has to cover."""
    return frozenset(_LISTS)


def op_of(configs: list | None) -> str | None:
    """Which op was handed this exact list object, or None.

    ``triton.Autotuner`` stores the list ``configs_for`` returned (it substitutes its own only for
    an empty list), so object identity is a reliable back-reference from a live autotuner to its
    op name -- the only one left now that the prune objects that used to carry the name are gone.
    """
    for op, live in _LISTS.items():
        if live is configs:
            return op
    return None


def missing_ops() -> list[str]:
    """Ops with no rows under the active directory. These cannot launch."""
    return sorted(op for op, live in _LISTS.items() if not live)


_ENV_DIR = os.environ.get("MINIWORLD_CONFIG_DIR", "").strip()
if _ENV_DIR:
    # Import-time selection: this module loads before any kernel module can call configs_for,
    # which is the only ordering that lets every op keep the list it was handed.
    _DIR = Path(_ENV_DIR)

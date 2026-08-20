"""Autotune configs come from CSV. Nothing in this package generates them.

One CSV per op, named ``<op>.csv``, under the active config directory. Its header names the
columns: every column that is not ``num_warps`` / ``num_stages`` / ``maxnreg`` is a tile axis and
must match the kernel's constexpr spelling exactly. One row is one config.

    BLOCK_M1,BLOCK_N,num_warps,num_stages
    64,128,4,3
    128,128,8,4

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


def _read(path: Path) -> list:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no config rows")
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


def op_of(configs: list) -> str | None:
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

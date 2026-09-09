"""``registry_module.csv``: the shapes the model runs, and the only place its columns are named.

Its own module rather than a section of :mod:`builder` or :mod:`derive`, because BOTH read it and
they already depend on each other -- ``derive`` drives the cases ``builder`` declares, and
``builder`` takes every number in those cases from this file. A shared leaf breaks the cycle and
makes the dependency direction obvious: the CSV knows nothing about either.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

REGISTRY_MODULE = Path(__file__).resolve().parents[1] / "kernels" / "registry_module.csv"
REGISTRY_KERNEL = Path(__file__).resolve().parents[1] / "kernels" / "registry_kernel.csv"

#: What each stream name means, as a length ladder. The names are the ones the model uses for its
#: activations (see viz.sweep_page.shape_name and MiniWorld's own token/atom vocabulary); the
#: ladders are MiniWorld's collate buckets -- CropConfig.bucket_token_size 128 and
#: bucket_atom_size 1024, so a production length is always a multiple of one of those.
STREAM_LADDERS: dict[str, tuple[int, ...]] = {
    "token_pair": (128, 256, 384, 512),
    "token_single": (128, 256, 384, 512),
    "atom_single": (1024, 2048, 4096, 8192),
    "msa_token": (128, 256, 384, 512),
}


@dataclasses.dataclass(frozen=True)
class ModuleRow:
    """One line of ``registry_module.csv``: a module and the input shapes it is fed."""

    module: str
    stream: str
    lengths: tuple[int, ...]
    dims: dict[str, int]
    options: tuple[tuple[str, str], ...]
    impls: tuple[str, ...]
    dtypes: tuple[str, ...]
    computes: tuple[str, ...]
    modes: tuple[str, ...]
    source: str


def _ints(raw: str) -> tuple[int, ...]:
    return tuple(int(v) for v in raw.split("|") if v)


def _strs(raw: str) -> tuple[str, ...]:
    return tuple(v for v in raw.split("|") if v)


def _dims(raw: str) -> dict[str, int]:
    out = {}
    for part in raw.split(";"):
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k] = int(v)
    return out


def module_rows(path: Path = REGISTRY_MODULE) -> list[ModuleRow]:
    """Read ``registry_module.csv``. The only place its column names are spelled out."""
    with path.open(newline="") as fh:
        rows = []
        for r in csv.DictReader(fh):
            opts = tuple(tuple(o.split("=", 1)) for o in _strs(r["options"]))  # type: ignore[misc]
            rows.append(ModuleRow(
                module=r["module"], stream=r["stream"], lengths=_ints(r["lengths"]),
                dims=_dims(r["dims"]), options=opts, impls=_strs(r["impls"]),
                dtypes=_strs(r["dtypes"]), computes=_strs(r["computes"]),
                modes=_strs(r["modes"]), source=r["source"]))
    return rows



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

#: Token counts run to 768, not to the training crop. `CropConfig.max_tokens` is 384 in every
#: committed data config, and reading it as the ceiling is what cut this ladder at 512: the crop
#: bounds TRAINING, and inference runs the trunk at whatever length the target is. The step is
#: `bucket_token_size` 128, so the ladder is 128 through 768; atoms step by `bucket_atom_size`
#: 1024 and run to 8192.
#:
#: What each stream name means, as a length ladder. The names are the ones the model uses for its
#: activations (see viz.sweep_page.shape_name and MiniWorld's own token/atom vocabulary); the
#: ladders are MiniWorld's collate buckets -- CropConfig.bucket_token_size 128 and
#: bucket_atom_size 1024, so a production length is always a multiple of one of those.
STREAM_LADDERS: dict[str, tuple[int, ...]] = {
    "token_pair": (128, 256, 384, 512, 640, 768),
    "token_single": (128, 256, 384, 512, 640, 768),
    "atom_single": (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192),
    "msa_token": (128, 256, 384, 512, 640, 768),
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
    eval_augmentation: int = 1
    train_augmentation: int = 1

    def augmentation(self, mode: str) -> int:
        return self.train_augmentation if mode == "train" else self.eval_augmentation


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
            opts = tuple((key, value) for option in _strs(r["options"])
                         for key, value in [option.split("=", 1)])
            rows.append(ModuleRow(
                module=r["module"], stream=r["stream"], lengths=_ints(r["lengths"]),
                dims=_dims(r["dims"]), options=opts, impls=_strs(r["impls"]),
                dtypes=_strs(r["dtypes"]), computes=_strs(r["computes"]),
                modes=_strs(r["modes"]), source=r["source"],
                eval_augmentation=int(r.get("eval_augmentation") or 1),
                train_augmentation=int(r.get("train_augmentation") or 1)))
            if min(rows[-1].eval_augmentation, rows[-1].train_augmentation) < 1:
                raise ValueError(f"{r['module']}: augmentation must be positive")
    return rows



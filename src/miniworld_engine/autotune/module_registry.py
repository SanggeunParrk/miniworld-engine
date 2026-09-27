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

#: Inference keeps its multi-bucket coverage. Training pads to exactly two
#: lengths per token/atom stream; it is filtered separately below. Neither the
#: inference ladder nor runtime cache-key bucketing is narrowed by this policy.
STREAM_LADDERS: dict[str, tuple[int, ...]] = {
    "token_pair": (128, 256, 384, 512, 640, 768),
    "token_single": (128, 256, 384, 512, 640, 768),
    "atom_single": (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192),
    "msa_token": (128, 256, 384, 512, 640, 768),
    "atom_pair": (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192),
    "noise": (1,),
}

TRAIN_TOKEN_LENGTHS = (384, 768)
TRAIN_ATOM_LENGTHS = (4096, 8192)
TRAIN_STREAM_LADDERS: dict[str, tuple[int, ...]] = {
    "token_pair": TRAIN_TOKEN_LENGTHS,
    "token_single": TRAIN_TOKEN_LENGTHS,
    "msa_token": TRAIN_TOKEN_LENGTHS,
    "atom_single": TRAIN_ATOM_LENGTHS,
    "atom_pair": TRAIN_ATOM_LENGTHS,
    # Noise is one conditioning vector, not a token/atom length axis.
    "noise": (1,),
}


def lengths_for_mode(stream: str, lengths: tuple[int, ...], mode: str) -> tuple[int, ...]:
    """Restrict training work while preserving the declared inference shapes."""
    if mode == "eval":
        return lengths
    if mode != "train":
        raise ValueError(f"unknown build mode: {mode!r}")
    allowed = TRAIN_STREAM_LADDERS[stream]
    return tuple(length for length in lengths if length in allowed)


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

    def lengths_for(self, mode: str) -> tuple[int, ...]:
        if mode not in self.modes:
            return ()
        return lengths_for_mode(self.stream, self.lengths, mode)


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


def transition_driver_shapes(kernel: str) -> tuple[tuple[str, int, int, int], ...]:
    """Exact (side, length, hidden width, expansion ratio) tuples for B2B probes.

    Widths from norms/attention projections are not transition dimensions. Keep
    K and n paired as declared by each module, and honor the two dispatch domains.
    The module build remains authoritative for modes/flags and augmentation.
    """
    small = kernel == "transition_fwd_b2b_triton"
    sides = {"token_pair": "pair", "token_single": "token", "atom_single": "atom",
             "msa_token": "msa"}
    shapes = set()
    for row in module_rows():
        if row.module not in {"transition", "swiglu_ffn"}:
            continue
        width = row.dims["d_hidden"]
        if kernel in {"transition_b2b_residual_triton", "transition_segmented_b2b_triton"}:
            if width not in (128, 256) or row.dims.get("n", 4) != 4:
                continue
        elif (width <= 128) != small:
            continue
        expanded = row.dims.get("d_expanded", row.dims.get("n", 4) * width)
        if kernel in {"transition_b2b_residual_triton", "transition_segmented_b2b_triton"} and expanded != 4 * width:
            continue
        if expanded % width:
            raise ValueError("Transition driver requires an integral expansion ratio")
        for length in row.lengths:
            shapes.add((sides[row.stream], length, width, expanded // width))
    return tuple(sorted(shapes))

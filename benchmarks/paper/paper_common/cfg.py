"""Load a measurement config of the harness with command-line overrides (the same keys as foldjet_engine.measure.config)."""

from __future__ import annotations

import argparse
from pathlib import Path

from foldjet_engine.measure import config


def add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--module", nargs="+", help="override the config's modules (smoke runs)")
    p.add_argument("--backend", nargs="+", help="override the config's backends (pytorch is always added)")
    p.add_argument("--rows", nargs="+", type=int, help="override the config's requirements rows (1 = first data row of the CSV)")
    p.add_argument("--lengths", help="override: min | all | 128,256")
    p.add_argument("--modes", nargs="+", choices=["eval", "train"], help="override the config's modes")
    p.add_argument("--dtypes", nargs="+", help="override the config's dtypes")
    p.add_argument("--max-cases", type=int, default=0, help="stop after N cases (smoke runs; 0 = all)")


def load(path: str, args: argparse.Namespace):
    """The config at ``path`` (a relative ``requirements:`` path is taken relative to the config file) with the overrides applied."""
    path = str(Path(path).resolve())
    cfg = config.load(path)
    req = Path(cfg.requirements)
    if not req.is_absolute():
        cfg = config.replace(cfg, requirements=str((Path(path).parent / req).resolve()))
    lengths = getattr(args, "lengths", None)
    if lengths not in (None, "min", "all"):
        lengths = [int(v) for v in lengths.split(",")]
    return config.replace(
        cfg,
        modules=getattr(args, "module", None),
        backends=getattr(args, "backend", None),
        rows=getattr(args, "rows", None),
        lengths=lengths,
        modes=getattr(args, "modes", None),
        dtypes=getattr(args, "dtypes", None),
    )


def limit(cfg, max_cases: int):
    """``config.cases`` capped at ``max_cases`` (0 = no cap): returns the cases list."""
    out = []
    for case in config.cases(cfg):
        out.append(case)
        if max_cases and len(out) >= max_cases:
            break
    return out

"""A fake GPU for the CPU tests of the per-card routing (`modules.dispatch.route_for_card`, `autotune.records`).

`fake_card(monkeypatch, name, capability)` makes `torch.cuda` report that card (no driver needed); `write_manifest` writes a
manifest for it into a temp directory the records module then reads (`records.use_manifest_dir`)."""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from miniworld_engine.autotune import records
from miniworld_engine.modules import dispatch


def fake_card(monkeypatch, name: str, capability: tuple[int, int]) -> torch.device:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    dispatch._card_cache.clear()
    dispatch._unrecorded_warned.clear()
    dispatch._refused_warned.clear()
    return torch.device("cuda", 0)


def write_manifest(directory: Path, name: str, capability: tuple[int, int], ops, *, dtype: str = "bf16", status: str = "ok",
                   skip: frozenset[str] = frozenset(), training: bool = True) -> Path:
    """`run_all`'s manifest for the card with every Triton kernel of ``ops`` marked ``status`` (except ``skip``; the training kernels
    -- backward, saved-activation -- only when ``training``)."""
    path = Path(directory) / f"{records.card_key(name, capability)}.csv"
    rows = [["kernel", "backend", "family", "file", "status", "detail", "dtype"],
            ["#provenance", "0.0.0", "0" * 12, "clean", "2026-10-24", "test", ""]]
    seen = set()
    for op in ops:
        # a training record covers what both modes launch (an inference-only kernel is not in the training set)
        modes = ("inference", "training") if training else ("inference",)
        for kernel in (k for m in modes for k in records.triton_kernels(op, dtype, m)):
            if kernel not in seen and kernel not in skip:
                seen.add(kernel)
                rows.append([kernel, "triton", "x", "x.py", status, "", records.row_dtype(kernel, dtype)])
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path

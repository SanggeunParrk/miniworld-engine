"""Raw records -> the paper CSV. One row per (case, backend); repeats (one record file per Slurm job) are aggregated here."""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

#: the columns every experiment's CSV starts with (benchmarks/paper/README.md); an experiment appends its own after them
BASE = [
    "id", "card", "gpu", "engine_version", "commit", "module", "mode", "backend", "L", "d", "n_head", "dtype",
    "ms_median", "ms_min", "ms_max", "peak_mem_gib", "status", "reason",
]  # fmt: skip
SPEED_EXTRA = ["stack", "row", "variant", "samples", "depth", "repeats", "timed_as"]

_D_KEYS = ("d_pair", "d_model", "c_z", "d_hidden", "normalized_shape", "d_in", "dim", "c_in")
_H_KEYS = ("n_head", "num_heads", "n_heads", "heads")


def dims_of(kwargs: dict) -> tuple[object, object]:
    """``(d, n_head)`` of a case's constructor kwargs ('' where the class has none): the channel width and the head count."""
    d = next((kwargs[k] for k in _D_KEYS if isinstance(kwargs.get(k), int)), "")
    if d == "":
        d = next((v for v in kwargs.values() if isinstance(v, int) and not isinstance(v, bool)), "")
    return d, next((kwargs[k] for k in _H_KEYS if k in kwargs), "")


def variant_of(kwargs: dict) -> str:
    return ";".join(f"{k}={v}" for k, v in kwargs.items())


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # a record cut by a time limit: the case is simply absent
                    continue
    return out


def meta_of(path: str) -> dict:
    p = Path(f"{path}.meta.json")
    return json.loads(p.read_text()) if p.exists() else {}


def record_files(exp_dir: Path, card: str, pattern: str = "*.jsonl") -> list[str]:
    """Raw record files of a card: ``<exp_dir>/records/<card>/`` or, when RECORDS is set (smoke runs), ``$RECORDS/<card>/``."""
    base = Path(os.environ["RECORDS"]) if os.environ.get("RECORDS") else exp_dir / "records"
    return sorted(glob.glob(str(base / card / pattern)))


def csv_path(exp_dir: Path, name: str) -> Path:
    """Where the CSV goes: ``<exp_dir>/<name>``, or ``$PAPER_CSV_DIR/<name>`` (smoke runs)."""
    return Path(os.environ["PAPER_CSV_DIR"]) / name if os.environ.get("PAPER_CSV_DIR") else exp_dir / name


def stage_of(path: str) -> str:
    """``records/<card>/<stage>.r<k>.jsonl`` -> stage (engine | kit | ...)."""
    return Path(path).name.split(".")[0]


def write(path: Path, rows: list[dict], extra: list[str]) -> None:
    cols = BASE + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[collect] {len(rows)} rows -> {path}")


def speed_rows(exp_id: str, exp_dir: Path, card: str) -> list[dict]:
    """Aggregate the speed records of ``records/<card>/`` (foldjet_engine.measure.speed): the per-repeat medians give ms_median (their
    median), ms_min and ms_max; a case no repeat measured keeps the status and reason of its last record (never re-routed)."""
    groups: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    for f in record_files(exp_dir, card):
        if f.endswith(".meta.json"):
            continue
        meta, stage = meta_of(f), stage_of(f)
        for rec in read_jsonl(f):
            if rec.get("timing", "graph") != "graph":
                continue
            key = (stage, rec["row"], rec["module"], json.dumps(rec["kwargs"], sort_keys=True), rec["length"], rec.get("depth"),
                   rec.get("samples"), rec["dtype"], rec["mode"], rec["backend"])  # fmt: skip
            groups[key].append((rec, meta))
    rows = []
    for key, items in groups.items():
        stage = key[0]
        ok = [(r, m) for r, m in items if r.get("status") == "ok"]
        rec, meta = (ok or items)[-1]
        d, h = dims_of(rec["kwargs"])
        row = {
            "id": exp_id,
            "card": card,
            "gpu": (rec.get("env") or {}).get("gpu") or meta.get("gpu", ""),
            "engine_version": meta.get("engine_version", ""),
            "commit": meta.get("commit", ""),
            "module": rec["module"],
            "mode": rec["mode"],
            "backend": rec["backend"],
            "L": rec["length"],
            "d": d,
            "n_head": h,
            "dtype": rec["dtype"],
            "status": "ok" if ok else rec.get("status", ""),
            "reason": "" if ok else rec.get("reason", ""),
            "stack": stage,
            "row": rec["row"],
            "variant": variant_of(rec["kwargs"]),
            "samples": rec.get("samples", ""),
            "depth": rec.get("depth") or "",
            "repeats": len(ok),
            "timed_as": rec.get("timed_as", "graph") if ok else "",
        }
        if ok:
            ms = [r["median_ms"] for r, _ in ok]
            row.update(ms_median=statistics.median(ms), ms_min=min(ms), ms_max=max(ms))
        rows.append(row)
    return rows


ACCURACY_EXTRA = ["stack", "row", "variant", "samples", "depth", "light", "err_backend", "err_pytorch", "tol", "tensor", "external"]


def accuracy_rows(exp_id: str, exp_dir: Path, card: str) -> list[dict]:
    """The accuracy records (foldjet_engine.measure.accuracy) of ``records/<card>/`` as rows: status ``measured`` with the light
    (GREEN / YELLOW / RED), the errors against the float64 gold, the tolerance and the worst tensor; ``unsupported`` / ``skipped`` / ``error`` with
    their reason. A case measured twice keeps its last record."""
    last: dict[tuple, tuple[dict, dict, str]] = {}
    for f in record_files(exp_dir, card):
        meta, stage = meta_of(f), stage_of(f)
        for rec in read_jsonl(f):
            key = (stage, rec["row"], rec["module"], json.dumps(rec["kwargs"], sort_keys=True), rec["length"], rec.get("depth"),
                   rec.get("samples"), rec["dtype"], rec["mode"], rec["backend"])  # fmt: skip
            last[key] = (rec, meta, stage)
    rows = []
    for rec, meta, stage in last.values():
        d, h = dims_of(rec["kwargs"])
        rows.append(
            {
                "id": exp_id, "card": card, "gpu": meta.get("gpu", ""), "engine_version": meta.get("engine_version", ""),
                "commit": meta.get("commit", ""), "module": rec["module"], "mode": rec["mode"], "backend": rec["backend"],
                "L": rec["length"], "d": d, "n_head": h, "dtype": rec["dtype"], "status": rec["status"], "reason": rec.get("reason", ""),
                "stack": stage, "row": rec["row"], "variant": variant_of(rec["kwargs"]), "samples": rec.get("samples", ""),
                "depth": rec.get("depth") or "", "light": rec.get("light", ""), "err_backend": rec.get("err_backend", ""),
                "err_pytorch": rec.get("err_pytorch", ""), "tol": rec.get("tol", ""), "tensor": rec.get("tensor", ""),
                "external": rec.get("external", ""),
            }
        )  # fmt: skip
    return rows


MEMORY_EXTRA = ["stack", "row", "variant", "samples", "depth", "base_gib", "input_gib", "param_gib", "step_extra_gib"]


def memory_rows(exp_id: str, exp_dir: Path, card: str) -> list[dict]:
    """The peak-memory records (paper_common/memory.py): ``peak_mem_gib`` = ``max_memory_allocated`` of one step (inputs, parameters and
    temporaries); the input tensors, parameters and the memory in use before the step in their own columns."""
    last: dict[tuple, tuple[dict, dict, str]] = {}
    for f in record_files(exp_dir, card):
        meta, stage = meta_of(f), stage_of(f)
        for rec in read_jsonl(f):
            key = (stage, rec["row"], rec["module"], json.dumps(rec["kwargs"], sort_keys=True), rec["length"], rec.get("depth"),
                   rec.get("samples"), rec["dtype"], rec["mode"], rec["backend"])  # fmt: skip
            last[key] = (rec, meta, stage)
    rows = []
    for rec, meta, stage in last.values():
        d, h = dims_of(rec["kwargs"])
        ok = rec.get("status") == "ok"
        rows.append(
            {
                "id": exp_id, "card": card, "gpu": meta.get("gpu", ""), "engine_version": meta.get("engine_version", ""),
                "commit": meta.get("commit", ""), "module": rec["module"], "mode": rec["mode"], "backend": rec["backend"],
                "L": rec["length"], "d": d, "n_head": h, "dtype": rec["dtype"], "peak_mem_gib": rec["peak_gib"] if ok else "",
                "status": rec["status"], "reason": rec.get("reason", ""), "stack": stage, "row": rec["row"],
                "variant": variant_of(rec["kwargs"]), "samples": rec.get("samples", ""), "depth": rec.get("depth") or "",
                "base_gib": rec["base_gib"] if ok else "", "input_gib": rec["input_gib"] if ok else "",
                "param_gib": rec["param_gib"] if ok else "", "step_extra_gib": rec["peak_gib"] - rec["base_gib"] if ok else "",
            }
        )  # fmt: skip
    return rows

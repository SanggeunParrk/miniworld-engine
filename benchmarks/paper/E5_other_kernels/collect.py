"""E5, all operator classes and all candidates: raw records of records/<card>/ -> E5_<card>.csv (columns: benchmarks/paper/README.md).

    python collect.py --card h100            (RECORDS=<dir> reads <dir>/<card>/ instead; PAPER_CSV_DIR=<dir> writes there)

ONE row per (operator class, mode, L, dtype, backend). The record families of a card (the stage name = the file prefix ``<stage>.r<k>.jsonl``) are:

  e5                the unified run (cfg_e5_all.yaml; run_measure.py --ext <any of externals.AVAILABLE>)                     -> every class (preferred when present)
  ext               the first E5 run (TriAttn: pytorch / cuda / cueq context rows + ds4sci + trifast)                         -> owner of TriAttn context and of ds4sci / trifast
  e5ds4sci          ds4sci re-run after the double-scale fix of its adapter (README: the first-run ds4sci ``ok`` records are ignored)  -> owner of ds4sci
  ext_attn          the Pairformer-operator candidates (third-party-attn)  -> owner of AttentionPairBias, OPM, PWA, Transition, and of the other TriAttn candidates
  ext_trimul        the triangle-multiplication candidates (third-party-trimul)                                             -> owner of TriMul
  fill_l2048_train  5090 TriMul train L = 2048 (ran alone after OOM on the shared GPU)                                     -> TriMul

Each family also contains pytorch / cuda context rows of its own cases. When one (class, mode, L, dtype, backend) cell exists in several families the OWNER decides:
TriAttn: e5, ext, ext_attn; TriMul: e5, fill_l2048_train, ext_trimul; every other class: e5, ext_attn (so the engine cuda / pytorch / cueq context rows of TriAttn come
from the first E5 run and those of every other class from the family's own run). Within the owner order a cell takes the first family that has an ``ok`` record measured with
the harness's 20 timed iterations; failing that the first family with an ``ok`` record of another iteration count (the ``iterations`` column says which: the 5090 TriMul
cells of ext_trimul are 100-iteration records); failing that the first family's last non-ok record (status + reason). Repeats (one file per Slurm job) are aggregated:
ms_median = median of the repeats' medians, ms_min / ms_max = their extremes. Records of cases outside ``requirements_e5_all.csv`` (the 5090 TriMul records cover a wider grid) are ignored.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paper_common import csvio

STD_ITERATIONS = 20
REQUIREMENTS = HERE / "requirements_e5_all.csv"
OWNERS = {
    "TriAttn": ["e5ds4sci", "ext", "e5", "ext_attn"],
    "TriMul": ["e5", "fill_l2048_train", "ext_trimul"],
}
DEFAULT_OWNERS = ["e5", "ext_attn"]
EXTRA = [
    "class",
    "stack",
    "row",
    "variant",
    "samples",
    "depth",
    "repeats",
    "iterations",
    "timed_as",
    "graph_valid",
    "ms_eager",
    "ms_graph_invalid",
]
SPEED_FAMILIES = ("e5", "e5ds4sci", "ext", "ext_attn", "ext_trimul", "fill_l2048_train")
#: a re-run family that owns only these backends (its pytorch context rows are not used: the original family owns the context)
ONLY_BACKENDS = {"e5ds4sci": {"ds4sci"}}
#: ``ok`` records that are invalid: the first E5 run's ds4sci adapter scaled q twice (fixed; re-measured in ``e5ds4sci``)
INVALID_OK = {("ext", "ds4sci")}


def class_of(module: str, kwargs: dict) -> str:
    """The operator class label of a case (the paper's classes; a shape that differs in the architecture is its own class)."""
    if module == "TriangleAttention":
        return "TriAttn"
    if module == "TriangleMultiplication":
        return "TriMul"
    if module == "AttentionPairBias":
        return f"AttentionPairBias-{kwargs.get('d_single')}"
    if module == "OuterProductMean":
        return "OPM"
    if module == "MSAPairWeightedAveraging":
        return "PWA"
    if module == "Transition":
        return f"Transition-d{kwargs.get('d_hidden')}"
    return module


def _val(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _kwargs(dims: str, options: str) -> dict:
    out = {}
    for part in [p for p in (dims + ";" + options).split(";") if p]:
        k, v = part.split("=", 1)
        out[k] = _val(v)
    return out


def required_cases(path: Path = REQUIREMENTS) -> dict[tuple, int]:
    """``{(module, kwargs-json, L, depth, samples, dtype, mode): row}`` of the cases the unified config measures (depth / samples = the first listed: ``min``)."""
    out: dict[tuple, int] = {}
    with open(path) as fh:
        for row, r in enumerate(csv.DictReader(fh), start=1):
            kw = _kwargs(r["dims"], r["options"])
            depth = int(r["depths"].split("|")[0]) if r["depths"] else None
            for mode in r["modes"].split("|"):
                samples_field = r["eval_samples" if mode == "eval" else "train_samples"]
                if not samples_field:
                    continue
                samples = int(samples_field.split("|")[0])
                for dtype in r["dtypes"].split("|"):
                    for length in r["lengths"].split("|"):
                        out[(r["module"], json.dumps(kw, sort_keys=True), int(length), depth, samples, dtype, mode)] = row
    return out


def case_key(rec: dict) -> tuple:
    return (
        rec["module"],
        json.dumps(rec["kwargs"], sort_keys=True),
        rec["length"],
        rec.get("depth"),
        rec.get("samples"),
        rec["dtype"],
        rec["mode"],
    )


def iterations_of(meta: dict) -> int | None:
    return ((meta.get("settings") or {}).get("speed") or meta.get("settings") or {}).get("iterations")


def read(card: str, base: Path) -> dict[tuple, dict[str, list[tuple[dict, dict]]]]:
    """``{(case key, backend): {family: [(record, meta), ...]}}`` of the speed records of one card, limited to the required cases."""
    need = required_cases()
    out: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f in sorted((base / card).glob("*.jsonl")):
        stage = f.name.split(".")[0]
        if stage not in SPEED_FAMILIES or stage.endswith("_acc"):
            continue
        meta = csvio.meta_of(str(f))
        for rec in csvio.read_jsonl(str(f)):
            if rec.get("timing", "graph") != "graph":
                continue
            k = case_key(rec)
            if stage in ONLY_BACKENDS and rec["backend"] not in ONLY_BACKENDS[stage]:
                continue
            if (stage, rec["backend"]) in INVALID_OK and rec.get("status") == "ok":
                continue
            if k in need:
                out[(k, rec["backend"])][stage].append((rec, meta))
    return out


def rows_of(card: str, base: Path) -> list[dict]:
    need = required_cases()
    rows = []
    for (k, backend), fams in read(card, base).items():
        cls = class_of(k[0], json.loads(k[1]))
        order = [f for f in OWNERS.get(cls, DEFAULT_OWNERS) if f in fams]
        oks = {f: [(r, m) for r, m in fams[f] if r.get("status") == "ok"] for f in order}

        its = {f: {iterations_of(m) for _r, m in oks[f]} for f in order}
        pick = next((f for f in order if oks[f] and its[f] == {STD_ITERATIONS}), None) or next((f for f in order if oks[f]), None)
        fam = pick or order[0]
        items = oks[fam] if pick else fams[fam]
        rec, meta = items[-1]
        kw = rec["kwargs"]
        d, h = csvio.dims_of(kw)
        row = {
            "id": "E5", "card": card, "gpu": (rec.get("env") or {}).get("gpu") or meta.get("gpu", ""),
            "engine_version": meta.get("engine_version", ""), "commit": meta.get("commit", ""), "module": rec["module"], "mode": rec["mode"],
            "backend": backend, "L": rec["length"], "d": d, "n_head": h, "dtype": rec["dtype"],
            "status": "ok" if pick else rec.get("status", ""), "reason": "" if pick else str(rec.get("reason", ""))[:400],
            "class": cls, "stack": fam, "row": need[k], "variant": csvio.variant_of(kw), "samples": rec.get("samples", ""),
            "depth": rec.get("depth") or "", "repeats": len(items) if pick else 0, "timed_as": rec.get("timed_as", "graph") if pick else "",
            "iterations": iterations_of(meta) if not pick else min(its[fam], key=lambda v: v or 0),
        }  # fmt: skip
        if pick:
            ms = [r["median_ms"] for r, _ in items]
            row.update(ms_median=statistics.median(ms), ms_min=min(ms), ms_max=max(ms))
            if (rec.get("peak_mem_gib") or rec.get("peak_gib")) is not None:
                row["peak_mem_gib"] = rec.get("peak_mem_gib") or rec.get("peak_gib")
        rows.append(row)
    apply_graph_check(rows, card, base)
    rows.sort(key=lambda r: (r["row"], r["dtype"], r["mode"], r["L"], r["backend"]))
    return rows


def apply_graph_check(rows: list[dict], card: str, base: Path) -> None:
    """graph_check.py records (``graphcheck.r*.jsonl``) -> ``graph_valid`` / ``ms_eager`` of every row; a cell that failed the CUDA-graph validity check becomes
    status ``invalid_graph`` (reason = the failed checks, the invalid graph time kept in ``ms_graph_invalid``; ms_median / min / max cleared): its timing is not a
    measurement of the library. ``ms_eager`` (eager CUDA-event timing, same warmup / iterations / L2 flush) is the corrected time of such a cell and is given for every checked cell."""
    chk = {}
    for f in sorted((base / card).glob("graphcheck*.r*.jsonl")):
        for r in csvio.read_jsonl(str(f)):
            k = (case_key(r), r["backend"])
            real = r.get("verdict") in ("valid", "invalid_graph")
            if real or k not in chk or chk[k].get("verdict") == "check_error":
                chk[k] = r  # a real verdict is never replaced by a later placeholder / error record
    for row in rows:
        row.update(graph_valid="not checked" if row["backend"] != "pytorch" else "reference", ms_eager="", ms_graph_invalid="")
        k = (
            (
                row["module"],
                json.dumps(_kw(row), sort_keys=True),
                row["L"],
                row["depth"] or None,
                row["samples"],
                row["dtype"],
                row["mode"],
            ),
            row["backend"],
        )
        r = chk.get(k)
        if r is not None and r.get("verdict") == "check_error":
            row["graph_valid"] = "check error"
            row["ms_graph_invalid"] = ""
            continue
        if r is None or r.get("verdict") not in ("valid", "invalid_graph"):
            continue
        row["ms_eager"] = r["eager"]
        hard = [
            x for x in r.get("reasons", []) if not x.startswith("(d)")
        ]  # (d), the profile check, is advisory (CUPTI omits kernels of replayed graphs)
        if not hard:
            row["graph_valid"] = "valid"
        else:
            row["graph_valid"] = "invalid"
            row["ms_graph_invalid"] = row["ms_median"] if row.get("ms_median") not in (None, "") else ""
            row.update(status="invalid_graph", reason="; ".join(hard)[:400], ms_median="", ms_min="", ms_max="")


def _kw(row: dict) -> dict:
    return {k: _val(v) for k, v in (p.split("=", 1) for p in row["variant"].split(";") if p)}


def main(argv=None) -> None:
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True, choices=["h100", "a100", "rtx5090"])
    a = p.parse_args(argv)
    base = Path(os.environ["RECORDS"]) if os.environ.get("RECORDS") else HERE / "records"
    rows = rows_of(a.card, base)
    csvio.write(csvio.csv_path(HERE, f"E5_{a.card}.csv"), rows, EXTRA)
    c = defaultdict(int)
    for r in rows:
        c[(r["stack"], r["status"], r["iterations"])] += 1
    for k, v in sorted(c.items(), key=str):
        print(f"[collect]   family {k[0]:17s} status {k[1]:12s} iterations {k[2]}: {v} rows")


if __name__ == "__main__":
    main()

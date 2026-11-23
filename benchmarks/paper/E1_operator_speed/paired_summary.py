"""Paired E1 records (paired.py) -> per-case paired ratios, the E1-format rows and the engine-not-fastest list.

    python E1_operator_speed/paired_summary.py RECORDS.jsonl [...] --card rtx5090 --tag triattn_opm [--old ../E1_operator_speed/E1_rtx5090.csv]

Writes ``E1_operator_speed/E1_<card>_paired_<tag>.csv`` (the E1 columns, ``stack = paired``, ms = median of the per-round medians,
ms_min / ms_max = min / max of the rounds) and ``paper_common/out/<card>/engine_not_fastest_cells_paired_<tag>.csv`` (the columns of
``engine_not_fastest_cells.csv`` + ``ratio_min`` / ``ratio_max`` / ``old_ratio``). ratio = baseline / engine cuda, per round, then the median over
the rounds (< 1: the baseline is faster). ``--old``: the cross-process campaign CSV whose ratio (same case) goes into ``old_ratio``.
Prints the per-module / baseline summary and every case whose paired ratio is < 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paper_common import csvio  # noqa: E402

DT = {"master": "bf16", "float32": "fp32"}
BASE = ["pytorch", "triton", "cueq", "anthropic"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("records", nargs="+")
    p.add_argument("--card", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--old", default=None)
    a = p.parse_args()
    meta = csvio.meta_of(a.records[0])
    rows, cases = [], defaultdict(dict)
    for f in a.records:
        for rec in csvio.read_jsonl(f):
            d, h = csvio.dims_of(rec["kwargs"])
            ok = rec.get("status") == "ok"
            var = csvio.variant_of(rec["kwargs"])
            row = {"id": "E1", "card": a.card, "gpu": rec["env"]["gpu"], "engine_version": meta.get("engine_version", ""),
                   "commit": meta.get("commit", ""), "module": rec["module"], "mode": rec["mode"], "backend": rec["backend"],
                   "L": rec["length"], "d": d, "n_head": h, "dtype": rec["dtype"], "status": rec.get("status", ""),
                   "reason": "" if ok else rec.get("reason", ""), "stack": "paired", "row": rec["row"], "variant": var,
                   "samples": rec.get("samples", ""), "depth": rec.get("depth") or "", "repeats": len(rec.get("rounds_ms", [])),
                   "timed_as": rec.get("timed_as", "graph") if ok else ""}  # fmt: skip
            if ok:
                row.update(ms_median=rec["median_ms"], ms_min=min(rec["rounds_ms"]), ms_max=max(rec["rounds_ms"]))
                key = (rec["module"], rec["mode"], str(rec["row"]), var, str(rec["length"]), str(rec.get("depth") or ""),
                       "" if rec.get("samples") is None else str(rec["samples"]), rec["dtype"])  # fmt: skip
                cases[key][rec["backend"]] = rec
            rows.append(row)
    csvio.write(HERE / f"E1_{a.card}_paired_{a.tag}.csv", rows, csvio.SPEED_EXTRA)

    old = {}
    if a.old:
        ms = defaultdict(dict)
        with open(a.old, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["status"] != "ok" or not r["ms_median"] or (r["backend"] == "anthropic") != (r["stack"] == "kit"):
                    continue
                k = (r["module"], r["mode"], r["row"], r["variant"], r["L"], r["depth"], r["samples"], r["dtype"])
                ms[k][r["backend"]] = float(r["ms_median"])
        for k, b in ms.items():
            if "cuda" in b:
                for v in BASE:
                    if v in b:
                        old[(k, v)] = b[v] / b["cuda"]

    slower, agg = [], defaultdict(list)
    for k, b in sorted(cases.items()):
        if "cuda" not in b:
            continue
        c = b["cuda"]
        for v in BASE:
            if v not in b or len(b[v]["rounds_ms"]) != len(c["rounds_ms"]):
                continue
            rr = [x / y for x, y in zip(b[v]["rounds_ms"], c["rounds_ms"], strict=True)]
            ratio = statistics.median(rr)
            mod, mode, row, var, L, depth, samples, dt = k
            agg[(mod, mode, DT.get(dt, dt), v)].append(ratio)
            o = old.get((k, v))
            if ratio < 1 or (o is not None and o < 0.97):
                slower.append(dict(card=a.card, module=mod, mode=mode, dtype=DT.get(dt, dt), baseline=v, L=L, row=row, variant=var, depth=depth,
                                   samples=samples, engine_ms=c["median_ms"], baseline_ms=b[v]["median_ms"], ratio=ratio,
                                   ratio_min=min(rr), ratio_max=max(rr), old_ratio="" if o is None else o))  # fmt: skip
    out = HERE.parent / "paper_common" / "out" / a.card / f"engine_not_fastest_cells_paired_{a.tag}.csv"
    with open(out, "w", newline="") as fh:
        cols = ["card", "module", "mode", "dtype", "baseline", "L", "row", "variant", "depth", "samples", "engine_ms", "baseline_ms", "ratio",
                "ratio_min", "ratio_max", "old_ratio"]  # fmt: skip
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(x for x in slower)
    print(f"[paired] {len(slower)} rows (paired ratio < 1, or old ratio < 0.97) -> {out}")
    print("\nmodule | mode | dtype | baseline | n | geomean | min | max | n < 1 | n < 0.97")
    for (mod, mode, dt, v), xs in sorted(agg.items()):
        g = math.exp(sum(map(math.log, xs)) / len(xs))
        print(f"{mod} | {mode} | {dt} | {v} | {len(xs)} | {g:.3f} | {min(xs):.3f} | {max(xs):.3f} | {sum(x < 1 for x in xs)} | {sum(x < 0.97 for x in xs)}")
    print("\nrows: module | mode | dtype | baseline | L | row | engine ms | baseline ms | paired ratio [min-max] | old ratio")
    for x in sorted(slower, key=lambda x: x["ratio"]):
        o = f"{x['old_ratio']:.3f}" if x["old_ratio"] != "" else "-"
        print(f"{x['module']} | {x['mode']} | {x['dtype']} | {x['baseline']} | {x['L']} | {x['row']} | {x['engine_ms']:.4f} | {x['baseline_ms']:.4f} | "
              f"{x['ratio']:.3f} [{x['ratio_min']:.3f}-{x['ratio_max']:.3f}] | {o}")


if __name__ == "__main__":
    main()

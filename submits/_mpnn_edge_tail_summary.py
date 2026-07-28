"""Print the mpnn_edge_tail run as a table: time, speedup, and every accuracy column.

Usage: python submits/_mpnn_edge_tail_summary.py "<device name>" "<run id>"
"""

import csv
import pathlib
import sys

device, run_id = sys.argv[1], sys.argv[2]
root = pathlib.Path("benchmarks/kernels/mpnn_edge_tail/artifacts") / device
paths = sorted(root.glob(f"*{run_id}*.csv"))
if not paths:
    print(f"no CSVs under {root} matching {run_id}")
    raise SystemExit(0)

for path in paths:
    rows = list(csv.DictReader(path.open()))
    print(f"\n### {path.name}")
    if not rows:
        print("  (empty)")
        continue
    print(
        f"  {'impl':<16}{'L':>6}{'compiled':>10}{'cgraph':>10}"
        f"{'ms':>12}{'vs pytorch':>12}{'grad_rel':>12}{'out_rel':>12}  status"
    )
    baseline = {}
    for row in rows:
        if row["implementation"] == "pytorch" and row["status"] == "ok":
            baseline[row["seq_len"]] = float(row["value"])
    for row in rows:
        ok = row["status"] == "ok"
        value = float(row["value"]) if ok and row["value"] else float("nan")
        base = baseline.get(row["seq_len"])
        speedup = f"{base / value:.2f}x" if ok and base and value else "-"
        fmt = lambda key: (  # noqa: E731
            f"{float(row[key]):.3e}" if row.get(key) not in (None, "") else "-"
        )
        print(
            f"  {row['implementation']:<16}{row['seq_len']:>6}"
            f"{row['compiled']:>10}{row['cudagraph']:>10}"
            f"{value:>12.3f}{speedup:>12}"
            f"{fmt('grad_rel_frob'):>12}{fmt('output_rel_frob'):>12}  {row['status']}"
        )
        if not ok:
            print(f"      error: {(row['error'] or '')[:160]}")

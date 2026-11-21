"""E5 triangle multiplication: records/<card>/ext_trimul.r*.jsonl -> a markdown table (median over the repeats, ms) with the ratio to the engine's cuda.

    python E5_other_kernels/collect_trimul.py records/h100 [records/a100 ...] > table.md

Columns: pytorch, cuda, then each external backend as ``ms (x ratio)`` where ratio = backend / cuda (> 1: slower than the engine's cuda); a case a backend
does not serve shows ``-`` (reasons: the records' ``reason``).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BACKENDS = ["pytorch", "cuda", "fast_trimul", "gpumode", "geonn"]


def load(card_dir: Path):
    cells: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    reasons: dict[tuple, str] = {}
    reps = set()
    for f in sorted(card_dir.glob("ext_trimul.r*.jsonl")):
        reps.add(f.name.split(".")[1])
        for line in f.read_text().splitlines():
            r = json.loads(line)
            key = (r["mode"], r["kwargs"]["d_pair"], r["length"])
            if r.get("status") == "ok" and "median_ms" in r:
                cells[key][r["backend"]].append(r["median_ms"])
            elif r["backend"] not in cells[key]:
                reasons[(*key, r["backend"])] = f"{r.get('status')}: {str(r.get('reason', ''))[:70]}"
    return cells, reasons, sorted(reps)


def main(argv) -> int:
    for d in argv or ["records/h100"]:
        cells, reasons, reps = load(Path(d))
        print(f"\n### {Path(d).name}  (repeats: {', '.join(reps) or 'none'}; median of the repeats' medians, ms; ratio = backend / cuda)\n")
        print("| mode | d | L | " + " | ".join(BACKENDS) + " |")
        print("|---|---|---|" + "---|" * len(BACKENDS))
        for key in sorted(cells, key=lambda k: (k[0], k[1] != 128, k[1], k[2])):
            row = {b: statistics.median(v) for b, v in cells[key].items()}
            cuda = row.get("cuda")
            out = []
            for b in BACKENDS:
                if b not in row:
                    out.append("-")
                elif b in ("pytorch", "cuda") or not cuda:
                    out.append(f"{row[b]:.3f}")
                else:
                    out.append(f"{row[b]:.3f} (x{row[b] / cuda:.2f})")
            print(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(out) + " |")
        seen = {}
        for (m, _d, _l, b), why in reasons.items():
            seen.setdefault((m, b, why), 0)
            seen[(m, b, why)] += 1
        for (m, b, why), n in sorted(seen.items()):
            print(f"\n- {b} ({m}): {n} cell(s) not measured: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

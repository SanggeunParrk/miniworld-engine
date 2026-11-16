"""Final fair-protocol table + graphs for benchmarks/opm_win/README.md.
   python report.py KIT.jsonl BEFORE.jsonl AFTER.jsonl OUTDIR
Per (row, L, S): ms of each backend in its own stack (cool: median of reps; sustained: mean of the rotated reps), PyTorch of the engine
stack as the ms reference, and every backend's speed-up over PyTorch of ITS stack (the drift-free comparison: Anthropic ran in the kit
CUDA-13 stack). lead = (PyTorch_kit / Anthropic) ratio vs (PyTorch_engine / CUDA after) ratio: > 1 means the engine is faster.
Writes OUTDIR/table.md and OUTDIR/speedup_{cool,sus}.png."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    d = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        k = (r["row"], r["length"], r["depth"])
        d[k][r["backend"]] = (statistics.median(r["cool"]), statistics.fmean(r["sus"])) if r.get("status") == "ok" else None
    return d


kit, before, after = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
out = Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
keys = sorted(k for k in after if after[k].get("cuda"))


def sp(d, k, b, m):  # speed-up over the stack's own PyTorch
    e = d.get(k, {})
    return e["pytorch"][m] / e[b][m] if e.get(b) and e.get("pytorch") else None


lines = []
for m, name in ((0, "cool"), (1, "sustained")):
    lines.append(f"\n### {name} (ms; x = speed-up over PyTorch of the same stack; lead = CUDA after vs Anthropic, > 1 faster)\n")
    lines.append("| row | L | S | PyTorch | Anthropic | Triton | CUDA before | CUDA after | lead before | lead after | after vs Triton |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        a, b, e = kit.get(k, {}), before.get(k, {}), after[k]

        def cell(d, bk):
            v = d.get(bk)
            s = sp(d if d is not a else a, k, bk, m) if v else None
            return f"{v[m]:.3f} ({s:.2f}x)" if v and s else (f"{v[m]:.3f}" if v else "-")

        pa, pb, pe = sp(kit, k, "anthropic", m), sp(before, k, "cuda", m), sp(after, k, "cuda", m)
        lb = f"{pb / pa:.3f}" if pa and pb else "-"
        la = f"{pe / pa:.3f}" if pa and pe else "-"
        pt = sp(after, k, "triton", m)
        lt = f"{pe / pt:.3f}" if pt and pe else "-"
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {e['pytorch'][m]:.3f} | {cell(kit, 'anthropic')} | {cell(after, 'triton')} | "
                     f"{cell(before, 'cuda')} | {cell(after, 'cuda')} | {lb} | {la} | {lt} |")
(out / "table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = sorted({k[0] for k in keys})
for m, name in ((0, "cool"), (1, "sus")):
    fig, axes = plt.subplots(len(rows), 1, figsize=(15, 3.2 * len(rows)), squeeze=False)
    for ax, row in zip(axes[:, 0], rows):
        ks = [k for k in keys if k[0] == row]
        xs = range(len(ks))
        series = [("PyTorch", None, "#8A93A3", None, None), ("Anthropic", (kit, "anthropic"), "#D9622B", "//", None),
                  ("Triton", (after, "triton"), "#3B6FB6", None, ":"), ("CUDA before", (before, "cuda"), "#E3C77A", None, None),
                  ("CUDA after", (after, "cuda"), "#B8860B", None, None)]
        w = 0.16
        for si, (lab, src, col, hatch, ls) in enumerate(series):
            vals = [1.0 if src is None else (sp(src[0], k, src[1], m) or 0) for k in ks]
            ax.bar([x + (si - 2) * w for x in xs], vals, w, label=lab, color=col if hatch is None else "white", edgecolor=col,
                   hatch=hatch, linestyle=ls or "-", linewidth=1.4 if ls else 0.8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"L{k[1]}\nS{k[2]}" for k in ks], fontsize=7)
        ax.set_ylabel("speed-up over PyTorch")
        ax.set_title(f"OuterProductMean row {row} ({'cool' if m == 0 else 'sustained'})", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"speedup_{name}.png", dpi=110)
print("graphs ->", out)

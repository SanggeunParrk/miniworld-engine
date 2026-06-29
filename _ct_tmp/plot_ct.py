"""Parse the CT bench .out and emit a markdown report + speedup graphs (team-gm style).

CPU-only (matplotlib Agg). Run via srun on a compute node (libstdc++ on LD_LIBRARY_PATH).
Usage: python plot_ct.py <bench.out> <output_dir>
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = Path(sys.argv[1])
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
lines = src.read_text().splitlines()

# rows: section -> list of dicts
inf_rows, tr_rows = [], []
section = None
for ln in lines:
    if ln.startswith("=== INFERENCE"):
        section = "inf"; continue
    if ln.startswith("=== TRAINING"):
        section = "tr"; continue
    parts = ln.split("|")
    if section == "inf" and len(parts) == 4 and re.match(r"\s*(atom|token)\s+\d+", parts[0]):
        head = parts[0].split()
        corr = parts[1].split()
        tim = parts[2].split()
        sp = parts[3].split()
        inf_rows.append(dict(stream=head[0], M=int(head[1]), d=int(head[2]),
                             cos=float(corr[0]), maxerr=corr[1],
                             ours=float(tim[0]), eager=float(tim[1]), comp=float(tim[2]),
                             vs_eager=float(sp[0].rstrip("x")), vs_comp=float(sp[1].rstrip("x"))))
    if section == "tr" and len(parts) == 4 and re.match(r"\s*(atom|token)\s+\d+", parts[0]):
        head = parts[0].split()
        corr = parts[1].split()  # cos_y dx dcond dWa dWs dWsc dbsc
        tim = parts[2].split()
        sp = parts[3].split()
        tr_rows.append(dict(stream=head[0], M=int(head[1]), d=int(head[2]),
                            cos_y=float(corr[0]), cos_min=min(float(c) for c in corr),
                            ours=float(tim[0]), eager=float(tim[1]), comp=float(tim[2]),
                            vs_eager=float(sp[0].rstrip("x")), vs_comp=float(sp[1].rstrip("x"))))

md = []
md.append("# ConditionedTransition tail — bench (H100, fp32 / TF32)\n")
md.append(f"_Source: `{src}`_\n")
md.append("AdaLN is out of scope; `x` is the post-AdaLN activation. "
          "Inference dispatch: atom (d<=128) -> fused b2b, token (d>=256) -> composed 2-kernel. "
          "Training: autograd Function (cuBLAS GEMMs + fused triton elementwise). "
          "Baselines: torch eager and torch.compile, both TF32.\n")

def tbl(rows, cols, hdr):
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        md.append("| " + " | ".join(str(c(r)) for c in cols) + " |")
    md.append("")

md.append("## Inference: correctness + latency\n")
md.append("_cos vs torch eager TF32 reference; us per call; speedup = baseline/ours (higher better)._\n")
tbl(inf_rows,
    [lambda r: r["stream"], lambda r: r["M"], lambda r: r["d"],
     lambda r: f'{r["cos"]:.6f}', lambda r: r["maxerr"],
     lambda r: f'{r["ours"]:.1f}', lambda r: f'{r["eager"]:.1f}', lambda r: f'{r["comp"]:.1f}',
     lambda r: f'{r["vs_eager"]:.2f}x', lambda r: f'{r["vs_comp"]:.2f}x'],
    ["stream", "M", "d", "cos", "maxerr", "ours_us", "eager_us", "compile_us", "vs_eager", "vs_compile"])

md.append("## Training (fwd+bwd): correctness + latency\n")
md.append("_cos_y = output cosine; cos_min = worst grad cosine (over dx,dcond,dWa,dWb,dWs,dWsc,db_sc); us per fwd+bwd._\n")
tbl(tr_rows,
    [lambda r: r["stream"], lambda r: r["M"], lambda r: r["d"],
     lambda r: f'{r["cos_y"]:.5f}', lambda r: f'{r["cos_min"]:.5f}',
     lambda r: f'{r["ours"]:.1f}', lambda r: f'{r["eager"]:.1f}', lambda r: f'{r["comp"]:.1f}',
     lambda r: f'{r["vs_eager"]:.2f}x', lambda r: f'{r["vs_comp"]:.2f}x'],
    ["stream", "M", "d", "cos_y", "cos_min", "ours_us", "eager_us", "compile_us", "vs_eager", "vs_compile"])

# --- graphs: one fig per (section), grouped bars latency by (stream,M), backends ours/eager/compile
def plot(rows, title, fname):
    if not rows:
        return
    labels = [f'{r["stream"][:1]}{r["d"]}\nM={r["M"]}' for r in rows]
    import numpy as np
    x = np.arange(len(rows)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.1), 4.2))
    ax.bar(x - w, [r["ours"] for r in rows], w, label="ours")
    ax.bar(x, [r["eager"] for r in rows], w, label="torch eager")
    ax.bar(x + w, [r["comp"] for r in rows], w, label="torch.compile")
    ax.set_ylabel("us per call (lower better)")
    ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / fname, dpi=120); plt.close(fig)

plot(inf_rows, "ConditionedTransition inference latency (H100, TF32)", "ct_inference.png")
plot(tr_rows, "ConditionedTransition fwd+bwd latency (H100, TF32)", "ct_fwd_bwd.png")
md.append("![inference](ct_inference.png)\n")
md.append("![fwd+bwd](ct_fwd_bwd.png)\n")

(outdir / "conditioned_transition.md").write_text("\n".join(md))
print("WROTE", outdir / "conditioned_transition.md")
print("WROTE", outdir / "ct_inference.png", outdir / "ct_fwd_bwd.png")

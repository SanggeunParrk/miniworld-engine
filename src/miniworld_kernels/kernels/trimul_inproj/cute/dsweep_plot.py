"""Plot the trimul D-sweep (latency vs D, one panel per L). No GPU needed; render
on a CPU srun (no --gres). Parses the `DATA ...` lines from dsweep.out (one per D
process) and uses the shared viz style (color_for/label_for/apply_theme/save_figure)
so colours match every other figure in the repo. Missing ours_v4 points (IMA at
D=64, PTX-codegen fail at D>=256) are simply absent from its line.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from miniworld_kernels.viz import apply_theme, color_for, label_for, save_figure

COLS = ["pytorch", "nvidia(dtv1)", "cuequivariance", "ours_v4", "ours_v6"]
# v4 = single fused back (red, the hero); v6 = split back (orange ours-variant).
PLOT_KEY = {"ours_v4": "ours", "ours_v6": "v5"}  # v5->ours-alt (orange) in the palette
LABELS = {"ours_v4": "ours v4 (single)", "ours_v6": "ours v6 (split)"}
BENCH = _src_root / "miniworld_kernels/kernels/trimul_inproj/benchmark"


def _color(c):
    return color_for(PLOT_KEY.get(c, c))


def _label(c):
    return LABELS.get(c, label_for(c))


def parse(out_path):
    """Return {(D,L): {col: ms}} from all `DATA ...` lines in the .out."""
    rows = {}
    for line in out_path.read_text().splitlines():
        if not line.startswith("DATA "):
            continue
        for cell in line[len("DATA "):].strip().split(";"):
            if not cell:
                continue
            dl, vals = cell.split("=")
            D, L = (int(x) for x in dl.split(":"))
            xs = [float(v) for v in vals.split(",")]
            rows[(D, L)] = dict(zip(COLS, xs))
    return rows


def main():
    rows = parse(BENCH / "dsweep_split.out")
    ds = sorted({D for (D, _) in rows})
    ls = sorted({L for (_, L) in rows})
    apply_theme()
    fig, axes = plt.subplots(1, len(ls), figsize=(6 * len(ls), 4.6), sharex=True, squeeze=False)
    for ax, L in zip(axes[0], ls):
        for c in COLS:
            xs = [D for D in ds if not math.isnan(rows.get((D, L), {}).get(c, float("nan")))]
            ys = [rows[(D, L)][c] for D in xs]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", markersize=7, color=_color(c), label=_label(c),
                    linewidth=2.5 if c.startswith("ours") else 1.8,
                    linestyle="--" if c == "ours_v6" else "-")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ds)
        ax.set_xticklabels([str(d) for d in ds])
        ax.set_xlabel("D (channel)")
        ax.set_ylabel("ms / layer  (lower is better)")
        ax.set_title(f"trimul forward, L={L}  (B=1, bf16, no mask)")
        ax.legend()
        ax.set_title(ax.get_title() + "\nv4=single fused back, v6=split (cute LNLinear + triton GateElem)",
                     fontsize=9)
    fig.tight_layout()
    paths = save_figure(fig, BENCH / "dsweep_split_latency.png")
    print("wrote " + ", ".join(str(p.name) for p in paths))


if __name__ == "__main__":
    main()

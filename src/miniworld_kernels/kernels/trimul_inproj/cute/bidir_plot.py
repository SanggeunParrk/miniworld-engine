"""Plot the bidirectional trimul bench (latency vs L + speedup). No GPU; CPU srun.
Parses `DATA L=pytorch,ours;...` from bidir.out, uses the shared viz style."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from miniworld_kernels.viz import apply_theme, color_for, label_for, save_figure

COLS = ["pytorch", "ours"]
BENCH = _src_root / "miniworld_kernels/kernels/trimul_inproj/benchmark"


def parse(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.startswith("DATA "):
            continue
        for cell in line[len("DATA "):].strip().split(";"):
            if not cell:
                continue
            L, vals = cell.split("=")
            rows[int(L)] = dict(zip(COLS, (float(v) for v in vals.split(","))))
    return rows


def main():
    rows = parse(BENCH / "bidir.out")
    ls = sorted(rows)
    apply_theme()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: latency
    for c in COLS:
        ys = [rows[L][c] for L in ls]
        ax.plot(ls, ys, marker="o", markersize=7, color=color_for(c), label=label_for(c),
                linewidth=2.5 if c == "ours" else 1.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ls)
    ax.set_xticklabels([str(x) for x in ls])
    ax.set_xlabel("L (sequence length)")
    ax.set_ylabel("ms / layer  (lower is better)")
    ax.set_title("Bidirectional trimul forward\n(B=1, bf16, d_pair=d_hidden=128 → back over 256ch)",
                 fontsize=10)
    ax.legend()

    # right: speedup of ours vs compiled pytorch
    sp = [rows[L]["pytorch"] / rows[L]["ours"] for L in ls]
    ax2.bar([str(L) for L in ls], sp, color=color_for("ours"))
    for i, s in enumerate(sp):
        ax2.text(i, s, f"{s:.1f}×", ha="center", va="bottom", fontsize=10)
    ax2.set_xlabel("L (sequence length)")
    ax2.set_ylabel("speedup vs compiled PyTorch")
    ax2.set_title("ours speedup (cos=0.99998)", fontsize=10)
    ax2.set_ylim(0, max(sp) * 1.2)

    fig.tight_layout()
    paths = save_figure(fig, BENCH / "bidir_latency.png")
    print("wrote " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()

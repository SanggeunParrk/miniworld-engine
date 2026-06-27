"""Plot single-dir v6 fwd+bwd vs baselines (stacked fwd/bwd). Parses v6_fwdbwd.out. CPU srun."""

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
import numpy as np

from miniworld_kernels.viz import apply_theme, color_for, label_for, save_figure

COLS = ["pytorch", "nvidia(dtv1)", "cuequivariance", "ours_v6"]
SHOW = ["nvidia(dtv1)", "cuequivariance", "ours_v6"]   # pytorch ~4x, omit for readability
KEY = {"nvidia(dtv1)": "dtv1", "cuequivariance": "cuequivariance", "ours_v6": "ours"}
BENCH = _src_root / "miniworld_kernels/kernels/trimul_inproj/benchmark"


def parse(path):
    fwd, fb = {}, {}
    for line in path.read_text().splitlines():
        for tag, d in (("DATA_FWD ", fwd), ("DATA_FB ", fb)):
            if line.startswith(tag):
                for cell in line[len(tag):].strip().split(";"):
                    L, vals = cell.split("=")
                    d[int(L)] = dict(zip(COLS, (float(v) for v in vals.split(","))))
    return fwd, fb


def main():
    fwd, fb = parse(BENCH / "v6_fwdbwd.out")
    ls = sorted(fwd)
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(ls))
    w = 0.25
    for i, c in enumerate(SHOW):
        off = (i - 1) * w
        f = [fwd[L][c] for L in ls]
        b = [fb[L][c] - fwd[L][c] for L in ls]
        col = color_for(KEY[c])
        ax.bar(x + off, f, w, color=col, label=f"{label_for(KEY[c])} fwd")
        ax.bar(x + off, b, w, bottom=f, color=col, alpha=0.45, label=f"{label_for(KEY[c])} bwd")
    for i, c in enumerate(SHOW):
        off = (i - 1) * w
        for j, L in enumerate(ls):
            ax.text(j + off, fb[L][c], f"{fb[L][c]:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in ls])
    ax.set_xlabel("L (sequence length)")
    ax.set_ylabel("ms / layer (fwd + bwd stacked)")
    ax.set_title("Single-dir trimul fwd+bwd (EAGER): ours_v6 vs dtv1 / cuEquiv\n"
                 "(pytorch omitted ~4×; backward is where v6 loses)", fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    paths = save_figure(fig, BENCH / "v6_fwdbwd_latency.png")
    print("wrote " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()

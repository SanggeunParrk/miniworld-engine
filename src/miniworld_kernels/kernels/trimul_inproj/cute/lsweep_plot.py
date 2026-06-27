"""Plot the L-sweep (latency vs L) at D=128 and D=256, focused on v6 (split back).
Parses `DATA D:L=...` from lsweep.out; shared viz style. CPU srun (no --gres)."""

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

from miniworld_kernels.viz import apply_theme, color_for, save_figure

COLS = ["pytorch", "nvidia(dtv1)", "cuequivariance", "ours_v4", "ours_v6"]
# show the interesting kernels (drop pytorch — 10-20x slower, compiled-baseline noisy)
SHOW = ["ours_v6", "ours_v4", "nvidia(dtv1)", "cuequivariance"]
KEY = {"ours_v6": "ours", "ours_v4": "v5", "nvidia(dtv1)": "dtv1", "cuequivariance": "cuequivariance"}
LAB = {"ours_v6": "ours v6 (split)", "ours_v4": "ours v4 (single)",
       "nvidia(dtv1)": "NVIDIA dtv1", "cuequivariance": "cuEquivariance"}
BENCH = _src_root / "miniworld_kernels/kernels/trimul_inproj/benchmark"


def parse(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.startswith("DATA "):
            continue
        for cell in line[len("DATA "):].strip().split(";"):
            if not cell:
                continue
            dl, vals = cell.split("=")
            D, L = (int(x) for x in dl.split(":"))
            rows[(D, L)] = dict(zip(COLS, (float(v) for v in vals.split(","))))
    return rows


def main():
    rows = parse(BENCH / "lsweep.out")
    ds = sorted({D for (D, _) in rows})
    ls_all = sorted({L for (_, L) in rows})
    apply_theme()
    fig, axes = plt.subplots(1, len(ds), figsize=(6 * len(ds), 4.6), squeeze=False)
    for ax, D in zip(axes[0], ds):
        for c in SHOW:
            xs = [L for L in ls_all if not math.isnan(rows.get((D, L), {}).get(c, float("nan")))]
            if not xs:
                continue
            ys = [rows[(D, L)][c] for L in xs]
            ax.plot(xs, ys, marker="o", markersize=6, color=color_for(KEY[c]), label=LAB[c],
                    linewidth=2.5 if c == "ours_v6" else 1.7,
                    linestyle="--" if c == "ours_v4" else "-")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ls_all)
        ax.set_xticklabels([str(x) for x in ls_all])
        ax.set_xlabel("L (sequence length)")
        ax.set_ylabel("ms / layer  (lower is better)")
        ax.set_title(f"trimul forward, D={D}" + ("  (v4 single fails — split only)" if D >= 256 else ""),
                     fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("v6 (split back) L-scaling vs baselines  (B=1, bf16, no mask; pytorch omitted)",
                 fontsize=11)
    fig.tight_layout()
    paths = save_figure(fig, BENCH / "lsweep_latency.png")
    print("wrote " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()

"""v4 (single fused back) vs v6 (split back) across L, at D=128 (the only D where both
run — v4 fails at D>=256). Latency + v6/v4 ratio. Parses lsweep.out. CPU srun."""

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

from miniworld_kernels.viz import apply_theme, color_for, save_figure

COLS = ["pytorch", "nvidia(dtv1)", "cuequivariance", "ours_v4", "ours_v6"]
BENCH = _src_root / "miniworld_kernels/kernels/trimul_inproj/benchmark"
D = 128


def parse(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.startswith("DATA "):
            continue
        for cell in line[len("DATA "):].strip().split(";"):
            if not cell:
                continue
            dl, vals = cell.split("=")
            d, L = (int(x) for x in dl.split(":"))
            rows[(d, L)] = dict(zip(COLS, (float(v) for v in vals.split(","))))
    return rows


def main():
    rows = parse(BENCH / "lsweep.out")
    ls = sorted(L for (d, L) in rows if d == D)
    v4 = [rows[(D, L)]["ours_v4"] for L in ls]
    v6 = [rows[(D, L)]["ours_v6"] for L in ls]
    apply_theme()
    c4, c6 = color_for("ours"), color_for("v5")  # v4 red, v6 orange
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax.plot(ls, v4, marker="o", ms=7, color=c4, lw=2.4, label="v4 (single fused back)")
    ax.plot(ls, v6, marker="o", ms=7, color=c6, lw=2.4, ls="--", label="v6 (split back)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ls)
    ax.set_xticklabels([str(x) for x in ls])
    ax.set_xlabel("L (sequence length)")
    ax.set_ylabel("ms / layer  (lower is better)")
    ax.set_title(f"v4 vs v6 back, D={D}  (B=1, bf16)", fontsize=10)
    ax.legend(fontsize=9)

    ratio = [b / a for a, b in zip(v4, v6)]
    ax2.plot(ls, ratio, marker="o", ms=7, color=c6, lw=2.4)
    ax2.axhline(1.0, color="#888", lw=1.2, ls=":")
    for x, r in zip(ls, ratio):
        ax2.annotate(f"{r:.2f}×", (x, r), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8)
    ax2.fill_between(ls, 1.0, ratio, where=[r > 1 for r in ratio], color=c4, alpha=0.10)
    ax2.fill_between(ls, 1.0, ratio, where=[r <= 1 for r in ratio], color=c6, alpha=0.18)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ls)
    ax2.set_xticklabels([str(x) for x in ls])
    ax2.set_xlabel("L (sequence length)")
    ax2.set_ylabel("v6 / v4  (>1 → v4 faster)")
    ax2.set_title("split vs single: v4 wins small L, v6 ties/wins large L", fontsize=10)

    fig.tight_layout()
    paths = save_figure(fig, BENCH / "v4_vs_v6_latency.png")
    print("wrote " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()

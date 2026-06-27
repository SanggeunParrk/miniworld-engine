"""Plot single-dir v6 fwd+bwd (torch.compile) vs dtv1/cuequiv/pytorch. CPU srun.
Parses 'DATA <method> L:val,...' from v6_fwdbwd_compile.out; shared viz style."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src = _Path(__file__).resolve()
while _src.name != "src" and _src.parent != _src:
    _src = _src.parent
if str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from miniworld_kernels.viz import apply_theme, color_for, label_for, save_figure

BENCH = _src / "miniworld_kernels/kernels/trimul_inproj/benchmark"
KEY = {"pytorch": "pytorch", "nvidia(dtv1)": "dtv1", "cuequivariance": "cuequivariance",
       "ours_v6": "ours"}
SHOW = ["ours_v6", "nvidia(dtv1)", "cuequivariance", "pytorch"]


def parse(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.startswith("DATA "):
            continue
        m = re.match(r"^DATA (.*?)\s+(\d+:.*)$", line)   # method (may have parens) then L:v,...
        if not m:
            continue
        rows[m.group(1)] = {int(c.split(":")[0]): float(c.split(":")[1]) for c in m.group(2).split(",")}
    return rows


def main():
    rows = parse(BENCH / "v6_fwdbwd_compile.out")
    ls = sorted(next(iter(rows.values())).keys())
    apply_theme()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for c in SHOW:
        if c not in rows:
            continue
        ys = [rows[c][L] for L in ls]
        ax.plot(ls, ys, marker="o", ms=7, color=color_for(KEY[c]), label=label_for(KEY[c]),
                linewidth=2.6 if c == "ours_v6" else 1.7)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ls)
    ax.set_xticklabels([str(x) for x in ls])
    ax.set_xlabel("L (sequence length)")
    ax.set_ylabel("ms / layer  (lower is better)")
    ax.set_title("Single-dir trimul fwd+bwd (torch.compile, exact training)", fontsize=10)
    ax.legend(fontsize=9)

    # speedup of ours vs dtv1
    sp = [rows["nvidia(dtv1)"][L] / rows["ours_v6"][L] for L in ls]
    x = np.arange(len(ls))
    ax2.bar(x, sp, color=color_for("ours"))
    for i, s in enumerate(sp):
        ax2.text(i, s, f"{s:.2f}×", ha="center", va="bottom", fontsize=10)
    ax2.axhline(1.0, color="#888", lw=1, ls=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(L) for L in ls])
    ax2.set_xlabel("L")
    ax2.set_ylabel("ours speedup vs dtv1")
    ax2.set_title("ours v6 fwd+bwd vs NVIDIA dtv1 (>1 → ours faster)", fontsize=10)
    ax2.set_ylim(0, max(sp) * 1.25)

    fig.tight_layout()
    paths = save_figure(fig, BENCH / "v6_fwdbwd_compile.png")
    print("wrote " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()

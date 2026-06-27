"""Plot fused (bidirectional) vs separate (out+in) — latency + fuse-speedup. CPU srun."""

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

from miniworld_kernels.viz import apply_theme, color_for, save_figure

COLS = ["pytorch_sep", "pytorch_bidir", "ours_sep", "ours_bidir"]
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
    rows = parse(BENCH / "bidir_vs_sep.out")
    ls = sorted(rows)
    apply_theme()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    PY, OURS = color_for("pytorch"), color_for("ours")
    series = [
        ("pytorch_sep", PY, "--", "PyTorch separate (out+in)"),
        ("pytorch_bidir", PY, "-", "PyTorch fused (bidir)"),
        ("ours_sep", OURS, "--", "ours separate (out+in)"),
        ("ours_bidir", OURS, "-", "ours fused (bidir)"),
    ]
    for c, col, ls_, lab in series:
        ax.plot(ls, [rows[L][c] for L in ls], marker="o", markersize=6, color=col,
                linestyle=ls_, linewidth=2.4 if c.endswith("bidir") else 1.6, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ls)
    ax.set_xticklabels([str(x) for x in ls])
    ax.set_xlabel("L (sequence length)")
    ax.set_ylabel("ms / layer  (lower is better)")
    ax.set_title("Fused vs separate trimul directions\n(B=1, bf16, per-direction h=128)", fontsize=10)
    ax.legend(fontsize=8)

    # fuse speedup = separate / fused
    x = np.arange(len(ls))
    w = 0.36
    py_sp = [rows[L]["pytorch_sep"] / rows[L]["pytorch_bidir"] for L in ls]
    ou_sp = [rows[L]["ours_sep"] / rows[L]["ours_bidir"] for L in ls]
    ax2.bar(x - w / 2, py_sp, w, color=PY, label="PyTorch")
    ax2.bar(x + w / 2, ou_sp, w, color=OURS, label="ours")
    for i, (a, b) in enumerate(zip(py_sp, ou_sp)):
        ax2.text(i - w / 2, a, f"{a:.2f}×", ha="center", va="bottom", fontsize=8)
        ax2.text(i + w / 2, b, f"{b:.2f}×", ha="center", va="bottom", fontsize=8)
    ax2.axhline(1.0, color="#888", lw=1, ls=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(L) for L in ls])
    ax2.set_xlabel("L (sequence length)")
    ax2.set_ylabel("fuse speedup  (separate / fused)")
    ax2.set_title("how much fusing the 2 directions wins", fontsize=10)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    paths = save_figure(fig, BENCH / "bidir_vs_sep.png")
    print("wrote " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()

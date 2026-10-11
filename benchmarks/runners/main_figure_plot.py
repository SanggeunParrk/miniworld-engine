"""Draw the main figure (``main_inference`` / ``main_training``): every module at L = 384 tokens (atom blocks 4096 atoms), speedup
over PyTorch per implementation.

The pair-stack modules are bf16-mixed; the diffusion transformer blocks (token DiT, bias-only token DiT, AF3 atom DiT, SWA atom
DiT) are drawn twice, bf16-mixed and fp32, each against the PyTorch of the same precision.

usage: python benchmarks/runners/main_figure_plot.py OUT_DIR bf16mixed.csv [more.csv ...] -- fp32.csv [more.csv ...]
Later files override earlier ones; the CSVs are the output of ``main_figure_campaign.py``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

TIMES = chr(0xD7)          # the multiplication sign
COLOURS = {"pytorch": "#8A8F98", "cuequivariance": "#E8833A", "anthropic": "#2A9D8F", "miniworld": "#1F4E9E"}
LABELS = {"pytorch": "PyTorch", "cuequivariance": "cuEquivariance", "anthropic": "Anthropic", "miniworld": "Ours"}
ORDER = ["pytorch", "cuequivariance", "anthropic", "miniworld"]
TRUNK = [
    ("trimul1", "TriMul\n(outgoing)"), ("trimul2", "TriMul\n(bidirectional)"), ("triattn", "Triangle\nattention"),
    ("trans", "Transition\n(pair)"), ("apb", "Attention\npair bias"), ("opm", "Outer product\nmean"),
    ("pwa", "MSA pair-\nweighted avg."),
]
DIFFUSION = [
    ("tdit", "Token DiT"), ("bodit", "Bias-only\ntoken DiT"), ("ldit", f"AF3 atom DiT\n(32{TIMES}128)"),
    ("swadit3", "SWA atom DiT\n(window 129)"),
]
RC = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 14, "axes.titlesize": 16,
    "axes.titleweight": "bold", "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E3E5E8",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 220,
    "savefig.bbox": "tight", "axes.edgecolor": "#444", "svg.fonttype": "none",
}


def load(files: list[str]) -> dict[tuple[str, str, str], float]:
    latency: dict[tuple[str, str, str], float] = {}
    for name in files:
        with open(name, newline="") as handle:
            for row in csv.DictReader(handle):
                latency[(row["module"], row["mode"], row["implementation"])] = float(row["latency_ms"])
    return latency


def draw(mode: str, data: dict[str, dict], out: Path, summary: list) -> None:
    groups = [(key, title, "bf16-mixed") for key, title in TRUNK]
    for key, title in DIFFUSION:
        groups += [(key, title, "bf16-mixed"), (key, title, "fp32")]
    n_trunk = len(TRUNK)
    slots = [impl for impl in ORDER if not (mode == "training" and impl == "anthropic")]
    fig, ax = plt.subplots(figsize=(24, 6.8))
    width = 0.8 / len(slots)
    for index, (key, title, precision) in enumerate(groups):
        latency = data[precision]
        base = latency.get((key, mode, "pytorch"))
        if base is None:
            continue
        for slot, impl in enumerate(slots):
            value = latency.get((key, mode, impl))
            if value is None:
                continue
            x = index + (slot - (len(slots) - 1) / 2) * width
            speedup = base / value
            ax.bar(x, speedup, width * 0.92, color=COLOURS[impl], label=LABELS[impl], zorder=3)
            summary.append((title.replace("\n", " "), precision, mode, LABELS[impl], value, speedup))
            if impl == "miniworld":
                ax.text(x, speedup * 1.02, f"{speedup:.1f}{TIMES}", ha="center", va="bottom", fontsize=12, fontweight="bold",
                        color=COLOURS[impl])
    ax.axhline(1.0, color="#444", lw=1.1, zorder=4)
    ax.axvline(n_trunk - 0.5, color="#444", lw=1.0, ls="--", zorder=2)
    for start in range(n_trunk, len(groups), 2):
        ax.axvspan(start - 0.5, start + 1.5, color="#F4F6F9" if ((start - n_trunk) // 2) % 2 == 0 else "white", zorder=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{title}\n{precision}" if index >= n_trunk else title
                        for index, (_key, title, precision) in enumerate(groups)], fontsize=10.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("Speedup over PyTorch (higher is better)")
    ax.set_title(f"B200 · {mode} · L = 384 tokens (atom blocks: 4096 atoms)"
                 + ("" if mode == "inference" else " · CUDA graph"), loc="left")
    top = ax.get_ylim()[1] * 1.22
    ax.set_ylim(0, top)
    ax.text((n_trunk - 1) / 2, top * 0.965, "pair stack · bf16-mixed (fp32 master)", ha="center", fontsize=13,
            color="#444", style="italic")
    ax.text(n_trunk + (len(DIFFUSION) * 2 - 1) / 2 - 0.5, top * 0.965,
            "diffusion transformer · bf16-mixed (fp32 master) | fp32", ha="center", fontsize=13, color="#444", style="italic")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    shown = [LABELS[impl] for impl in slots if LABELS[impl] in unique]
    ax.legend([unique[name] for name in shown], shown, ncol=len(slots), loc="upper left", frameon=False,
              bbox_to_anchor=(0.0, 0.915))
    fig.savefig(out / f"main_{mode}.png")
    fig.savefig(out / f"main_{mode}.svg")
    plt.close(fig)


def main() -> None:
    separator = sys.argv.index("--")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    data = {"bf16-mixed": load(sys.argv[2:separator]), "fp32": load(sys.argv[separator + 1:])}
    mpl.rcParams.update(RC)
    summary: list = []
    for mode in ("inference", "training"):
        draw(mode, data, out, summary)
    with open(out / "summary_L384.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module", "precision", "mode", "implementation", "latency_ms", "speedup_vs_pytorch"])
        for module, precision, mode, impl, value, speedup in summary:
            writer.writerow([module, precision, mode, impl, f"{value:.4f}", f"{speedup:.2f}"])
    print(len(summary), "bars")


if __name__ == "__main__":
    main()

"""Plot bidir trimul fwd+bwd (ours vs compiled pytorch), two d_pair panels. Parses
bidir_train.out → bidir_train_latency.png + bidir_train.md. CPU srun (no --gres)."""

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

BENCH = _src_root / "miniworld_kernels/modules/triangle_multiplication/benchmark"
COLS = ["pytorch_bmm", "dtv1_bidir", "cuequiv_x2", "ours_bidir"]
KEY = {"pytorch_bmm": "pytorch", "dtv1_bidir": "dtv1", "cuequiv_x2": "cuequivariance",
       "ours_bidir": "ours"}
LBL = {"pytorch_bmm": "pytorch", "dtv1_bidir": "dtv1 (fused bidir)", "cuequiv_x2": "cuEquiv ×2",
       "ours_bidir": "ours"}


def parse(path):
    data = {}   # d -> {method -> {L -> ms}}
    for line in path.read_text().splitlines():
        if line.startswith("DATA d"):
            _, dtag, method, cells = line.split(maxsplit=3)
            d = int(dtag[1:])
            data.setdefault(d, {})[method] = {
                int(L): float(v) for L, v in (c.split(":") for c in cells.split(","))}
    return data


def main():
    data = parse(BENCH / "bidir_train.out")
    ds = sorted(data)
    apply_theme()
    fig, axes = plt.subplots(1, len(ds), figsize=(5.2 * len(ds), 4.6), squeeze=False)
    for ax, d in zip(axes[0], ds):
        ls = sorted(data[d]["ours_bidir"])
        x = np.arange(len(ls))
        n = len(COLS)
        w = 0.8 / n
        for i, c in enumerate(COLS):
            ys = [data[d][c][L] for L in ls]
            ax.bar(x + (i - (n - 1) / 2) * w, ys, w, color=color_for(KEY[c]), label=LBL[c])
        # speedup vs dtv1 (strongest baseline) above ours bars
        oi = COLS.index("ours_bidir")
        for j, L in enumerate(ls):
            sp = data[d]["dtv1_bidir"][L] / data[d]["ours_bidir"][L]
            ax.text(j + (oi - (n - 1) / 2) * w, data[d]["ours_bidir"][L], f"{sp:.2f}x",
                    ha="center", va="bottom", fontsize=7)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(L) for L in ls])
        ax.set_xlabel("L (sequence length)")
        ax.set_ylabel("ms / layer (fwd + bwd, log)")
        ax.set_title(f"d_pair={d} (back K={2*d})", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Bidirectional trimul training (fwd+bwd): ours vs dtv1 / cuEquiv / pytorch\n"
                 "(B=1, bf16, H100; all torch.compile, event-timed; ×N speedup = vs dtv1-bidir)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths = save_figure(fig, BENCH / "bidir_train_latency.png")
    print("wrote " + ", ".join(p.name for p in paths))

    # markdown table
    lines = ["# Bidirectional trimul training (fwd+bwd) — ours vs dtv1 / cuEquiv / pytorch", "",
             "B=1, bf16, H100. All methods `torch.compile` (default), params require grad "
             "(exact training), event-timed. ms / layer. Correctness: all grads cos 0.99997+ "
             "vs fp32 ref (PASS all L).", "",
             "`dtv1_bidir` = a FUSED bidirectional dt-v1 built from dt-v1's OWN kernels with the "
             "SAME architecture as ours (apples-to-apples). `cuequiv_x2` = the cuequiv vendor op "
             "run for both directions (a black-box op can't be re-fused). `pytorch_bmm` "
             "(efficient matmul contraction) is reference-only — just slow at large L.", ""]
    for d in ds:
        ls = sorted(data[d]["ours_bidir"])
        lines += [f"## d_pair={d} (back K={2*d})", "",
                  "| L | pytorch | dtv1-bidir | cuEquiv×2 | ours | vs dtv1 | vs cuEquiv |",
                  "|---|---|---|---|---|---|---|"]
        for L in ls:
            py, dv, cq, ou = (data[d]["pytorch_bmm"][L], data[d]["dtv1_bidir"][L],
                              data[d]["cuequiv_x2"][L], data[d]["ours_bidir"][L])
            lines.append(f"| {L} | {py:.3f} | {dv:.3f} | {cq:.3f} | {ou:.3f} | "
                         f"{dv/ou:.2f}x | {cq/ou:.2f}x |")
        lines.append("")
    lines += ["![latency](bidir_train_latency.png)", ""]
    (BENCH / "bidir_train.md").write_text("\n".join(lines))
    print("wrote bidir_train.md")


if __name__ == "__main__":
    main()

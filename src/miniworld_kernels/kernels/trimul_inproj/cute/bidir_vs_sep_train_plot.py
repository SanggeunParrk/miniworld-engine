"""Plot dt-v1 SEPARATE (sequential residual+dropout) vs BIDIRECTIONAL (one residual), training
fwd+bwd. Parses bidir_vs_sep_train.out → bidir_vs_sep_train.{png,md}. CPU srun (no --gres)."""

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

BENCH = _src_root / "miniworld_kernels/kernels/trimul_inproj/benchmark"
COLS = ["dtv1_sep", "dtv1_bidir", "ours_bidir"]
KEY = {"dtv1_sep": "pytorch", "dtv1_bidir": "dtv1", "ours_bidir": "ours"}
LBL = {"dtv1_sep": "dtv1 separate (2 residual blocks)", "dtv1_bidir": "dtv1 bidir (1 block)",
       "ours_bidir": "ours bidir (1 block)"}


def parse(path):
    data = {}
    for line in path.read_text().splitlines():
        if line.startswith("DATA d"):
            _, dtag, method, cells = line.split(maxsplit=3)
            d = int(dtag[1:])
            data.setdefault(d, {})[method] = {
                int(L): float(v) for L, v in (c.split(":") for c in cells.split(","))}
    return data


def main():
    data = parse(BENCH / "bidir_vs_sep_train.out")
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
        # fuse speedup (dtv1_sep / ours_bidir) above ours bars
        oi = COLS.index("ours_bidir")
        for j, L in enumerate(ls):
            sp = data[d]["dtv1_sep"][L] / data[d]["ours_bidir"][L]
            ax.text(j + (oi - (n - 1) / 2) * w, data[d]["ours_bidir"][L], f"{sp:.2f}x",
                    ha="center", va="bottom", fontsize=7)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(L) for L in ls])
        ax.set_xlabel("L (sequence length)")
        ax.set_ylabel("ms / layer (fwd + bwd, log)")
        ax.set_title(f"d_pair={d}", fontsize=10)
        ax.legend(fontsize=7.5)
    fig.suptitle("Pairformer separate (2 sequential dt-v1 residual+dropout blocks) vs "
                 "bidirectional (1 block)\ntraining fwd+bwd, B=1 bf16 H100, train-mode dropout "
                 "p=0.25; ×N = dtv1_sep / ours_bidir", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    paths = save_figure(fig, BENCH / "bidir_vs_sep_train.png")
    print("wrote " + ", ".join(p.name for p in paths))

    lines = ["# Separate (dt-v1 out+in, sequential residual+dropout) vs Bidirectional — training",
             "",
             "B=1, bf16, H100. `torch.compile`, params require grad, train-mode rowwise dropout "
             "p=0.25, event-timed. ms / layer (fwd+bwd). Correctness (eval, dropout off) vs fp32 "
             "ref: all cos 0.99999.", "",
             "`dtv1_sep` = the faithful pairformer block — `pair += drop(dtv1_out(pair)); "
             "pair += drop(dtv1_in(pair))` (incoming sees the outgoing-updated pair). "
             "`dtv1_bidir` / `ours_bidir` = one fused bidirectional update in a single residual. "
             "**NOTE: bidirectional is a DIFFERENT model** (both directions from the same input, "
             "cannot see the outgoing update) — this is the speed comparison only.", "",
             "`fuse↑` = dtv1_sep / bidir (how much one fused block beats two separate blocks).",
             ""]
    for d in ds:
        ls = sorted(data[d]["ours_bidir"])
        lines += [f"## d_pair={d}", "",
                  "| L | dtv1_sep | dtv1_bidir | ours_bidir | dtv1 fuse↑ | ours fuse↑ |",
                  "|---|---|---|---|---|---|"]
        for L in ls:
            s, db, ob = data[d]["dtv1_sep"][L], data[d]["dtv1_bidir"][L], data[d]["ours_bidir"][L]
            lines.append(f"| {L} | {s:.3f} | {db:.3f} | {ob:.3f} | {s/db:.2f}x | {s/ob:.2f}x |")
        lines.append("")
    lines += ["![latency](bidir_vs_sep_train.png)", ""]
    (BENCH / "bidir_vs_sep_train.md").write_text("\n".join(lines))
    print("wrote bidir_vs_sep_train.md")


if __name__ == "__main__":
    main()

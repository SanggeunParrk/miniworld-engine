"""Plot bidir trimul INFERENCE (fwd) ours vs dtv1-bidir / cuequiv / pytorch, 3 d_pair panels.
Parses bidir_infer.out → bidir_infer.{png,md}. CPU srun (no --gres)."""

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

BENCH = _src_root / "miniworld_kernels/modules/triangle_multiplication/benchmark"
COLS = ["pytorch", "dtv1_bidir", "cuequiv_x2", "ours"]
KEY = {"pytorch": "pytorch", "dtv1_bidir": "dtv1", "cuequiv_x2": "cuequivariance", "ours": "ours"}
LBL = {"pytorch": "pytorch", "dtv1_bidir": "dtv1 (fused bidir)", "cuequiv_x2": "cuEquiv ×2",
       "ours": "ours"}


def parse(path):
    data = {}
    for line in path.read_text().splitlines():
        if line.startswith("DATA d"):
            _, dtag, method, cells = line.split(maxsplit=3)
            d = int(dtag[1:])
            vals = {}
            for c in cells.split(","):
                L, v = c.split(":")
                fv = float(v)
                if fv == fv:  # drop nan
                    vals[int(L)] = fv
            data.setdefault(d, {})[method] = vals
    return data


def main():
    data = parse(BENCH / "bidir_infer.out")
    ds = sorted(data)
    apply_theme()
    fig, axes = plt.subplots(1, len(ds), figsize=(5.0 * len(ds), 4.6), squeeze=False)
    for ax, d in zip(axes[0], ds):
        ls = sorted(data[d]["ours"])
        x = np.arange(len(ls))
        n = len(COLS)
        w = 0.8 / n
        for i, c in enumerate(COLS):
            ys = [data[d][c].get(L, float("nan")) for L in ls]
            ax.bar(x + (i - (n - 1) / 2) * w, ys, w, color=color_for(KEY[c]), label=LBL[c])
        oi = COLS.index("ours")
        for j, L in enumerate(ls):
            sp = data[d]["dtv1_bidir"][L] / data[d]["ours"][L]
            ax.text(j + (oi - (n - 1) / 2) * w, data[d]["ours"][L], f"{sp:.2f}x",
                    ha="center", va="bottom", fontsize=7)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(L) for L in ls])
        ax.set_xlabel("L (sequence length)")
        ax.set_ylabel("ms / layer (forward, log)")
        ax.set_title(f"d_pair={d} (back K={2*d})", fontsize=10)
        ax.legend(fontsize=7.5)
    fig.suptitle("Bidirectional trimul INFERENCE (forward): ours vs dtv1 / cuEquiv / pytorch\n"
                 "(B=1, bf16, H100; pytorch=compile, others=CUDA-graph; ×N = vs dtv1-bidir)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    paths = save_figure(fig, BENCH / "bidir_infer.png")
    print("wrote " + ", ".join(p.name for p in paths))

    lines = ["# Bidirectional trimul INFERENCE (forward) — ours vs dtv1 / cuEquiv / pytorch", "",
             "B=1, bf16, H100. Forward only, no_grad. pytorch=`torch.compile`; dtv1-bidir / "
             "cuequiv / ours = manual CUDA-graph (deployment regime). ms / layer. ours uses the "
             "dedicated inference path `bidirectional_trimul_ours`. Correctness: ours & "
             "dtv1_bidir cos 0.99998 vs fp32 ref (all L).", ""]
    for d in ds:
        ls = sorted(data[d]["ours"])
        lines += [f"## d_pair={d} (back K={2*d})", "",
                  "| L | pytorch | dtv1-bidir | cuEquiv×2 | ours | vs dtv1 | vs cuEquiv |",
                  "|---|---|---|---|---|---|---|"]
        for L in ls:
            py = data[d]["pytorch"].get(L, float("nan"))
            db, cq, ou = data[d]["dtv1_bidir"][L], data[d]["cuequiv_x2"][L], data[d]["ours"][L]
            pystr = f"{py:.3f}" if py == py else "OOM"
            lines.append(f"| {L} | {pystr} | {db:.3f} | {cq:.3f} | {ou:.3f} | "
                         f"{db/ou:.2f}x | {cq/ou:.2f}x |")
        lines.append("")
    lines += ["![inference](bidir_infer.png)", ""]
    (BENCH / "bidir_infer.md").write_text("\n".join(lines))
    print("wrote bidir_infer.md")


if __name__ == "__main__":
    main()

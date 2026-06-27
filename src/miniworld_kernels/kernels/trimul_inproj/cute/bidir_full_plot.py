"""Render the full 5-method × {infer,train} × D × L matrix → bidir_full.{md,png}.
Parses bidir_full.out (DATA <mode> d<D> <method> L:v,...). CPU srun (no --gres)."""

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
COLS = ["ours_bidir", "dtv1_bidir", "ours_sep", "dtv1_sep", "cuequiv_sep"]
KEY = {"ours_bidir": "ours", "dtv1_bidir": "dtv1", "ours_sep": "ours", "dtv1_sep": "dtv1",
       "cuequiv_sep": "cuequivariance"}
HATCH = {"ours_sep": "//", "dtv1_sep": "//"}   # sep = hatched, bidir = solid
LBL = {"ours_bidir": "ours bidir", "dtv1_bidir": "dtv1 bidir", "ours_sep": "ours sep",
       "dtv1_sep": "dtv1 sep", "cuequiv_sep": "cuEquiv sep"}


def parse(path):
    data = {}   # (mode,d) -> {method -> {L->ms}}
    for line in path.read_text().splitlines():
        if line.startswith("DATA "):
            _, mode, dtag, method, cells = line.split(maxsplit=4)
            key = (mode, int(dtag[1:]))
            data.setdefault(key, {})[method] = {
                int(L): float(v) for L, v in (c.split(":") for c in cells.split(","))}
    return data


def main():
    data = parse(BENCH / "bidir_full.out")
    apply_theme()
    modes = ["infer", "train"]
    dvals = [128, 256, 512]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for r, mode in enumerate(modes):
        for c, d in enumerate(dvals):
            ax = axes[r][c]
            md = data.get((mode, d), {})
            if not md:
                ax.axis("off"); continue
            ls = sorted(md["ours_bidir"])
            x = np.arange(len(ls)); n = len(COLS); w = 0.8 / n
            for i, col in enumerate(COLS):
                ys = [md[col].get(L, float("nan")) for L in ls]
                ax.bar(x + (i - (n - 1) / 2) * w, ys, w, color=color_for(KEY[col]),
                       hatch=HATCH.get(col), edgecolor="white", linewidth=0.3, label=LBL[col])
            ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([str(L) for L in ls])
            ax.set_title(f"{mode}  d_pair={d}", fontsize=10)
            ax.set_xlabel("L"); ax.set_ylabel("ms/layer (log)")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Bidirectional trimul: 5 methods × {inference, fwd+bwd} × d_pair (B=1 bf16 H100)\n"
                 "sep = 2 sequential residual+dropout blocks (hatched); bidir = 1 residual block",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = save_figure(fig, BENCH / "bidir_full.png")
    print("wrote " + ", ".join(p.name for p in paths))

    out = ["# Bidirectional trimul — full matrix (5 methods × {inference, fwd+bwd} × d × L)", "",
           "ms / layer, B=1 bf16 H100. **sep** = faithful pairformer (2 sequential single-dir "
           "residual blocks, incoming sees the outgoing-updated pair, rowwise dropout p=0.25). "
           "**bidir** = one fused bidirectional update in one residual block. inference = forward "
           "no_grad, CUDA-graph (ours uses dedicated inference path). train = fwd+bwd, dropout on, "
           "torch.compile, event-timed. cuequiv = vendor op, sep only (can't fuse). cos 0.99998+ "
           "vs fp32 ref. **bold = fastest in row.**", ""]
    for mode in modes:
        out.append(f"# {mode.upper()}")
        for d in dvals:
            md = data.get((mode, d))
            if not md:
                continue
            ls = sorted(md["ours_bidir"])
            out += [f"## d_pair={d}", "",
                    "| L | " + " | ".join(LBL[c] for c in COLS) + " |",
                    "|---|" + "---|" * len(COLS)]
            for L in ls:
                vals = [md[c].get(L, float("nan")) for c in COLS]
                mn = min(v for v in vals if v == v)
                cells = []
                for v in vals:
                    s = f"{v:.3f}" if v == v else "OOM"
                    cells.append(f"**{s}**" if v == mn else s)
                out.append(f"| {L} | " + " | ".join(cells) + " |")
            out.append("")
    out += ["![matrix](bidir_full.png)", ""]
    (BENCH / "bidir_full.md").write_text("\n".join(out))
    print("wrote bidir_full.md")


if __name__ == "__main__":
    main()

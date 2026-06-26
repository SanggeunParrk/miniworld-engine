"""Render the TE-style LayerNormLinear fwd+bwd bench (table already in te_style.md) to bar charts.
Data = recorded clean run (H100 bf16, CUDA-warmed, after the db=ones@dY fix). Run via srun (no GPU
needed for plotting, but the env's libstdc++ is — set LD_LIBRARY_PATH=$CONDA_PREFIX/lib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# (M, d) -> (ours, TE, pytorch) ms, full fwd+bwd, H100 bf16
SHAPES = [(16384, 128), (65536, 128), (262144, 128),
          (16384, 256), (65536, 256), (262144, 256),
          (16384, 384), (65536, 384), (262144, 384)]
CONTIG = [(0.2085, 0.1976, 0.2156), (0.2089, 0.2220, 0.2609), (0.4069, 0.5568, 0.4188),
          (0.2052, 0.1977, 0.2331), (0.3144, 0.2879, 0.4227), (0.7597, 0.7895, 0.6779),
          (0.2176, 0.2012, 0.2545), (0.3277, 0.3528, 0.3516), (1.0701, 1.1175, 1.1570)]
MMAJOR = [(0.2117, 0.2294, 0.2572), (0.2164, 0.3412, 0.3676), (0.5179, 1.1302, 1.4778),
          (0.2320, 0.2369, 0.2535), (0.2469, 0.5074, 0.9626), (0.7739, 2.0234, 2.8667),
          (0.2190, 0.2420, 0.3476), (0.3528, 0.7384, 1.2462), (1.1714, 3.0730, 4.1710)]
LABELS = [f"d{d}\nM{m//1024}k" for (m, d) in SHAPES]
COL = {"ours": "#2563eb", "TE": "#f59e0b", "pytorch": "#9ca3af"}

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
x = np.arange(len(SHAPES)); w = 0.27

for col, (data, title) in enumerate([(CONTIG, "contiguous (fair — same algorithm)"),
                                     (MMAJOR, "m-major (trimul BDLL view)")]):
    ours = [d[0] for d in data]; te = [d[1] for d in data]; pt = [d[2] for d in data]
    # top: latency
    ax = axes[0][col]
    ax.bar(x - w, ours, w, label="ours (te_fn)", color=COL["ours"])
    ax.bar(x,     te,   w, label="TE",            color=COL["TE"])
    ax.bar(x + w, pt,   w, label="torch.compile", color=COL["pytorch"])
    ax.set_title(f"LayerNormLinear fwd+bwd latency — {title}", fontsize=11)
    ax.set_ylabel("ms (median, lower=better)"); ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=8)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    # bottom: speedup vs TE
    ax2 = axes[1][col]
    spd = [te[i] / ours[i] for i in range(len(x))]
    bars = ax2.bar(x, spd, 0.6, color=["#16a34a" if s >= 1 else "#dc2626" for s in spd])
    ax2.axhline(1.0, color="black", lw=1, ls="--")
    ax2.set_title(f"speedup vs TE — {title} (>1 = ours faster)", fontsize=11)
    ax2.set_ylabel("TE ms / ours ms"); ax2.set_xticks(x); ax2.set_xticklabels(LABELS, fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    for b, s in zip(bars, spd):
        ax2.text(b.get_x() + b.get_width() / 2, s + 0.03, f"{s:.2f}", ha="center", fontsize=8)

fig.suptitle("TE-style trainable LayerNormLinear (layernorm_linear_te_fn) vs TE / torch.compile  —  H100 bf16",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = Path(__file__).parent / "te_style_fwd_bwd.png"
fig.savefig(out, dpi=120)
print("wrote", out)

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
          (16384, 384), (65536, 384), (262144, 384),
          (16384, 512), (65536, 512), (262144, 512)]
CONTIG = [(0.2344, 0.2853, 0.3283), (0.2389, 0.2214, 0.2647), (0.3731, 0.5536, 0.4188),
          (0.2307, 0.2059, 0.2477), (0.2616, 0.2631, 0.2869), (0.6998, 0.7841, 0.6803),
          (0.2382, 0.2072, 0.2740), (0.3262, 0.3501, 0.3518), (1.0692, 1.1178, 1.1530),
          (0.2407, 0.2183, 0.2705), (0.4174, 0.4184, 0.4188), (1.4495, 1.3873, 1.4522)]
MMAJOR = [(0.2387, 0.2308, 0.2579), (0.2385, 0.3399, 0.3677), (0.4626, 1.1309, 1.5171),
          (0.2304, 0.2437, 0.2550), (0.2605, 0.5058, 0.9675), (0.8208, 2.0124, 2.8567),
          (0.2364, 0.2407, 0.3482), (0.3521, 0.7382, 1.2469), (1.2067, 3.0639, 4.1518),
          (0.2400, 0.2831, 0.4106), (0.4386, 0.9030, 1.4647), (1.4945, 3.9554, 5.2908)]
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

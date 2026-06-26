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
CONTIG = [(0.2176, 0.2034, 0.3279), (0.2371, 0.2199, 0.2957), (0.3727, 0.5554, 0.4180),
          (0.2083, 0.1932, 0.2288), (0.3174, 0.2913, 0.4196), (0.7600, 0.7884, 0.6782),
          (0.2156, 0.2061, 0.2641), (0.3272, 0.3521, 0.4383), (1.0688, 1.1180, 1.1579),
          (0.3127, 0.2703, 0.4167), (0.4174, 0.4165, 0.4208), (1.4546, 1.3922, 1.4594)]
MMAJOR = [(0.3158, 0.3223, 0.3739), (0.3120, 0.3389, 0.3894), (0.4950, 1.1258, 1.4780),
          (0.2161, 0.2277, 0.2560), (0.3217, 0.5076, 0.9642), (0.7816, 2.0176, 2.8608),
          (0.2174, 0.2387, 0.3464), (0.3475, 0.7380, 1.2504), (1.1920, 3.0665, 4.1724),
          (0.2227, 0.2826, 0.4134), (0.4485, 0.9003, 1.4375), (1.5767, 3.9320, 5.2750)]
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

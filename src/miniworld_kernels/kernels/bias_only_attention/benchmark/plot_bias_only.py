"""Render the bias-only module bench (.out) to a speedup bar chart."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def parse(out_path):
    rows = []
    for line in Path(out_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("return"):
            continue
        parts = line.split()
        if len(parts) != 10 or parts[1] != "triton":
            continue
        L = int(parts[0])
        rows.append((L, float(parts[8]), float(parts[9])))  # L, sp_infer, sp_fb
    return rows


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else HERE / "bench_bias_only_module.out"
    rows = parse(out_path)
    Ls = [r[0] for r in rows]
    sp_fwd = [r[1] for r in rows]
    sp_fb = [r[2] for r in rows]

    x = range(len(Ls))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - w / 2 for i in x], sp_fwd, w, label="inference")
    ax.bar([i + w / 2 for i in x], sp_fb, w, label="forward+backward")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(Ls)
    ax.set_xlabel("sequence length L")
    ax.set_ylabel("speedup vs pytorch (no_contig) baseline")
    ax.set_title("bias-only TriangleAttention: triton-LN path speedup (H100, bf16)")
    ax.legend()
    for i, (a, b) in enumerate(zip(sp_fwd, sp_fb)):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    png = HERE / "bench_bias_only_module.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()

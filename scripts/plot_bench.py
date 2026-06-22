"""Parse a bench_layernorm_linear.py log and emit a markdown report + graphs.

Benchmarks in this repo always ship a **table and a graph together** (project
convention — see benchmark/BENCHMARKING.md). This turns a captured ``.out`` into
both: per-metric markdown tables and per-metric PNG line plots (latency vs d,
one subplot per M, one line per backend).

Usage::

    python scripts/plot_bench.py <bench.out> <output_dir> [--title "..."]

``<output_dir>`` is the kernel's own benchmark folder
(``src/miniworld_kernels/kernels/<kernel>/benchmark/``) — results live with the
kernel, not in the top-level ``benchmark/``. See ``benchmark/BENCHMARKING.md``.

Runs CPU-only (no GPU); needs matplotlib. On the cluster use the team-gm env
with its libstdc++ on LD_LIBRARY_PATH (system gcc lacks CXXABI_1.3.15), via srun.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADER_RE = re.compile(r"M=(\d+)\s+d_in=(\d+)\s+d_out=(\d+)")
# backends with both fwd and fwd+bwd:
TIME_RE = re.compile(r"(torch\.compile|TE)\s+fwd=([\d.]+)\s*ms\s+fwd\+bwd=([\d.]+)\s*ms")
# forward-only timing lines (our fused inference kernel; also torch.compile/TE when a
# forward-only sweep like tune.py omits fwd+bwd). cute-fused MUST precede cute so the
# alternation doesn't match the "cute" prefix of "cute-fused".
FWD_ONLY_RE = re.compile(r"(torch\.compile|TE|cute-fused|cute|triton)\s+fwd=([\d.]+)\s*ms")

# Display order; only backends actually present in the log are shown.
BACKENDS = ["torch.compile", "TE", "cute", "cute-fused", "triton"]
METRICS = [("fwd", "forward"), ("fwd+bwd", "forward + backward")]


def parse(out_path: Path) -> dict:
    """Return {(M, d): {backend: {'fwd': ms, ['fwd+bwd': ms]}}}."""
    data: dict = {}
    cur = None
    for line in out_path.read_text().splitlines():
        h = HEADER_RE.search(line)
        if h:
            M, d_in, _d_out = int(h[1]), int(h[2]), int(h[3])
            cur = (M, d_in)
            data[cur] = {}
            continue
        if cur is None:
            continue
        t = TIME_RE.search(line)
        if t:
            data[cur][t[1]] = {"fwd": float(t[2]), "fwd+bwd": float(t[3])}
            continue
        f = FWD_ONLY_RE.search(line)
        if f:
            data[cur][f[1]] = {"fwd": float(f[2])}


    return data


def backends_for(data: dict, metric: str) -> list[str]:
    present = {b for rec in data.values() for b, v in rec.items() if metric in v}
    return [b for b in BACKENDS if b in present]


def axes(data: dict) -> tuple[list[int], list[int]]:
    Ms = sorted({M for (M, _d) in data})
    ds = sorted({d for (_M, d) in data})
    return Ms, ds


def make_table(data: dict, metric: str) -> str:
    Ms, ds = axes(data)
    backends = backends_for(data, metric)
    head = "| d (=d_in=d_out) | " + " | ".join(f"M={M}" for M in Ms) + " |"
    sep = "|" + "---|" * (len(Ms) + 1)
    rows = [f"_backends: {' / '.join(backends)} (bold = fastest)_\n", head, sep]
    for d in ds:
        cells = []
        for M in Ms:
            rec = data.get((M, d), {})
            vals = {b: rec.get(b, {}).get(metric) for b in backends}
            vals = {b: v for b, v in vals.items() if v is not None}
            if not vals:
                cells.append("—")
                continue
            best = min(vals.values())
            parts = [(f"**{v:.4f}**" if v == best else f"{v:.4f}") for b, v in vals.items()]
            cells.append(" / ".join(parts))
        rows.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def make_plot(data: dict, metric: str, label: str, out_png: Path, title: str) -> None:
    """Grouped BAR chart: one subplot per M, x = d groups, one bar per backend."""
    Ms, ds = axes(data)
    fig, axs = plt.subplots(1, len(Ms), figsize=(5 * len(Ms), 4.2), squeeze=False)
    backends = backends_for(data, metric)
    nb = max(len(backends), 1)
    width = 0.8 / nb
    xs = list(range(len(ds)))
    for j, M in enumerate(Ms):
        ax = axs[0][j]
        for bi, backend in enumerate(backends):
            ys = [data.get((M, d), {}).get(backend, {}).get(metric) for d in ds]
            ys_plot = [(y if y is not None else 0.0) for y in ys]
            offs = (bi - (nb - 1) / 2) * width
            bars = ax.bar([x + offs for x in xs], ys_plot, width, label=backend)
            ax.bar_label(bars, labels=[("" if y is None else f"{y:.3f}") for y in ys],
                         fontsize=6, rotation=90, padding=2)
        ax.set_title(f"M = {M}")
        ax.set_xlabel("d (= d_in = d_out)")
        ax.set_xticks(xs)
        ax.set_xticklabels(ds)
        ax.grid(True, axis="y", alpha=0.3)
        if j == 0:
            ax.set_ylabel(f"{label} latency (ms)")
        ax.legend(fontsize=8)
    fig.suptitle(f"{title} — {label} (lower is better)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out_path", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--title", default="LayerNormLinear: TE vs torch.compile (H100, bf16)")
    p.add_argument("--name", default=None, help="basename for outputs (default: out_path stem)")
    args = p.parse_args()

    data = parse(args.out_path)
    if not data:
        raise SystemExit(f"no bench records parsed from {args.out_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.out_path.stem

    md = [f"# {args.title}\n", f"_Source: `{args.out_path}`_\n"]
    for metric, label in METRICS:
        if not backends_for(data, metric):
            continue  # e.g. a forward-only sweep has no fwd+bwd data
        png = args.output_dir / f"{name}_{metric.replace('+', '_')}.png"
        make_plot(data, metric, label, png, args.title)
        md.append(f"## {label} latency (ms) — `torch.compile` / **TE** (bold = faster)\n")
        md.append(make_table(data, metric) + "\n")
        md.append(f"![{label}]({png.name})\n")
    md_path = args.output_dir / f"{name}.md"
    md_path.write_text("\n".join(md))
    print(f"wrote {md_path}")
    for metric, _ in METRICS:
        png = args.output_dir / f"{name}_{metric.replace('+', '_')}.png"
        if png.exists():
            print(f"wrote {png}")


if __name__ == "__main__":
    main()

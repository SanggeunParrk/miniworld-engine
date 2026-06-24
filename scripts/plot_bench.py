"""Parse a team-gm-style bench log and emit a markdown report + graphs.

Benchmarks in this repo always ship a **table and a graph together** (project
convention — see benchmark/BENCHMARKING.md). This turns a captured ``.out`` into
both: per-metric markdown tables and per-metric PNG plots. For 2D shape sweeps
(``M`` and ``d``), each metric gets one grouped-bar PNG with:
- x-axis = ``M``
- one subplot per ``d``
- one bar color per backend

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
import math

import matplotlib.pyplot as plt

HEADER_RE = re.compile(r"M=(\d+)\s+d_in=(\d+)\s+d_out=(\d+)")
BACKENDS = [
    "pytorch",
    "torch.compile",
    "TE",
    "cuequivariance",
    "cute-fused",
    "cute-train",
    "cute",
    "triton",
    "layernorm_kernel",
    "v2",
]
BACKEND_RE = "|".join(re.escape(b) for b in BACKENDS)
# backends with both fwd and fwd+bwd:
TIME_RE = re.compile(rf"({BACKEND_RE})\s+fwd=([\d.]+)\s*ms\s+fwd\+bwd=([\d.]+)\s*ms")
# forward-only timing lines (our fused inference kernel; also torch.compile/TE when a
# forward-only sweep like tune.py omits fwd+bwd). cute-fused MUST precede cute so the
# alternation doesn't match the "cute" prefix of "cute-fused".
FWD_ONLY_RE = re.compile(rf"({BACKEND_RE})\s+fwd=([\d.]+)\s*ms")
CORR_RE = re.compile(
    rf"({BACKEND_RE})\s+"
    r"fwd\(abs=([\deE.+-]+),rel=([\deE.+-]+),cos=([\deE.+-]+)\)\s+"
    r"dx\(abs=([\deE.+-]+),rel=([\deE.+-]+),cos=([\deE.+-]+)\)\s+"
    r"dw\(abs=([\deE.+-]+),rel=([\deE.+-]+),cos=([\deE.+-]+)\)\s+"
    r"db\(abs=([\deE.+-]+),rel=([\deE.+-]+),cos=([\deE.+-]+)\)"
)

METRICS = [("fwd", "forward"), ("fwd+bwd", "forward + backward")]


def parse(out_path: Path) -> dict:
    """Return {(M, d): {backend: {'fwd': ms, ['fwd+bwd': ms], ['corr']: ...}}}."""
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
            data[cur].setdefault(t[1], {}).update({"fwd": float(t[2]), "fwd+bwd": float(t[3])})
            continue
        f = FWD_ONLY_RE.search(line)
        if f:
            data[cur].setdefault(f[1], {}).update({"fwd": float(f[2])})
            continue
        c = CORR_RE.search(line)
        if c:
            data[cur].setdefault(c[1], {})["corr"] = {
                "fwd": {"abs": float(c[2]), "rel": float(c[3]), "cos": float(c[4])},
                "dx": {"abs": float(c[5]), "rel": float(c[6]), "cos": float(c[7])},
                "dw": {"abs": float(c[8]), "rel": float(c[9]), "cos": float(c[10])},
                "db": {"abs": float(c[11]), "rel": float(c[12]), "cos": float(c[13])},
            }


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


def color_map(backends: list[str]) -> dict[str, str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {backend: colors[i % len(colors)] for i, backend in enumerate(backends)}


def plot_grouped_bars(ax, data: dict, metric: str, backends: list[str], Ms: list[int], d: int, colors: dict[str, str]) -> None:
    xs = list(range(len(Ms)))
    nb = max(len(backends), 1)
    width = 0.82 / nb
    for bi, backend in enumerate(backends):
        ys = [data.get((M, d), {}).get(backend, {}).get(metric) for M in Ms]
        ys_plot = [y if y is not None else 0.0 for y in ys]
        offs = (bi - (nb - 1) / 2) * width
        ax.bar(
            [x + offs for x in xs],
            ys_plot,
            width,
            label=backend,
            color=colors[backend],
        )
    ax.set_title(f"d = {d}")
    ax.set_xlabel("M")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(M) for M in Ms], rotation=20)
    ax.grid(True, axis="y", alpha=0.3)


def make_plot(data: dict, metric: str, label: str, out_png: Path, title: str) -> None:
    """Per-metric grouped bar chart with x=M and one subplot per d."""
    Ms, ds = axes(data)
    backends = backends_for(data, metric)
    ncols = min(3, max(len(ds), 1))
    nrows = math.ceil(len(ds) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 4.0 * nrows), squeeze=False, sharey=True)
    colors = color_map(backends)

    for ax, d in zip(axs.flat, ds):
        plot_grouped_bars(ax, data, metric, backends, Ms, d, colors)
    for ax in axs.flat[len(ds):]:
        ax.axis("off")

    for r in range(nrows):
        axs[r][0].set_ylabel(f"{label} latency (ms)")

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.suptitle(f"{title} — {label} (lower is better)")
    fig.legend(handles, labels, loc="upper center", ncol=max(len(backends), 1), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def make_correctness_table(data: dict) -> str | None:
    present = []
    for key, rec in sorted(data.items()):
        M, d = key
        for backend in BACKENDS:
            corr = rec.get(backend, {}).get("corr")
            if corr is None:
                continue
            present.append((M, d, backend, corr))
    if not present:
        return None

    lines = [
        "| d (=d_in=d_out) | M | backend | fwd rel/cos | dx rel/cos | dw rel/cos | db rel/cos |",
        "|---|---:|---|---|---|---|---|",
    ]
    for M, d, backend, corr in present:
        lines.append(
            f"| {d} | {M} | {backend} | "
            f"{corr['fwd']['rel']:.3e} / {corr['fwd']['cos']:.6f} | "
            f"{corr['dx']['rel']:.3e} / {corr['dx']['cos']:.6f} | "
            f"{corr['dw']['rel']:.3e} / {corr['dw']['cos']:.6f} | "
            f"{corr['db']['rel']:.3e} / {corr['db']['cos']:.6f} |"
        )
    return "\n".join(lines)


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
        md.append(f"## {label} latency (ms)\n")
        md.append("_table: rows = d, columns = M, cells = backend latencies_\n")
        md.append(make_table(data, metric) + "\n")
        md.append("_plot: grouped bar chart with x = M and one subplot per d_\n")
        md.append(f"![{label}]({png.name})\n")
    corr_table = make_correctness_table(data)
    if corr_table is not None:
        md.append("## Correctness summary\n")
        md.append("_cells show relative Frobenius error / cosine similarity_\n")
        md.append(corr_table + "\n")
    md_path = args.output_dir / f"{name}.md"
    md_path.write_text("\n".join(md))
    print(f"wrote {md_path}")
    for metric, _ in METRICS:
        png = args.output_dir / f"{name}_{metric.replace('+', '_')}.png"
        if png.exists():
            print(f"wrote {png}")


if __name__ == "__main__":
    main()

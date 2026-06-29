"""FlashAttention-style speedup-vs-default figure from a 1-D sweep CSV.

Companion to ``benchmarks/runners/plot_bench.py`` (which handles the (M,d) grid bench
``.out`` format). Many of our kernels are benchmarked as a *single sweep* over
sequence length / L with hand-validated numbers living in markdown tables
(trimul, transition). This renders those into the **same** FA-style figure:
bars = speedup over the default (PyTorch), higher is better, every bar labelled
with its multiplier — using the shared ``miniworld_kernels.viz`` palette so the
figure matches every other figure in the repo.

CSV format::

    L, PyTorch|pytorch, cuEquivariance|cuequivariance, NVIDIA dt-v1|dtv1, ours|ours
    128, 0.285, 0.071, 0.067, 0.055
    ...

- First column = the sweep axis (its header becomes the x-label).
- Every other column = a backend's latency in **ms** at that point.
- A header cell ``"Label|identity"`` pins the palette: ``Label`` is shown, while
  ``identity`` (a key in ``viz.PALETTE`` — pytorch/triton/te/cuequivariance/
  dtv1/cuda/ours/ours-alt/ours-alt2) picks the colour. Without ``|`` the column
  name is canonicalised. The ``|identity`` form is how we give two "ours"
  variants distinct colours (e.g. a fused-triton path vs a cute path).

Outputs ``<name>_speedup.{png,svg,pdf}`` + ``<name>.md`` (speedup + latency
tables). CPU-only; run via srun, never the login node.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from miniworld_kernels.viz import (
    PALETTE,
    apply_theme,
    canonical,
    color_for,
    label_for,
    save_figure,
)

BASELINE = "pytorch"  # the "default" everything is measured against


def _fallback_color(ident: str) -> str:
    # mirror viz.style's deterministic fallback for off-catalogue identities
    return color_for(ident)


def parse_header(cell: str) -> tuple[str, str, str]:
    """Return (display_label, palette_identity, colour) for a backend column."""
    if "|" in cell:
        label, ident = (s.strip() for s in cell.split("|", 1))
        colour = PALETTE.get(ident, _fallback_color(ident))
        return label, ident, colour
    return label_for(cell), canonical(cell), color_for(cell)


def parse_csv(path: Path) -> tuple[str, list[float], list[dict]]:
    """Return (x_label, x_values, [{label, ident, colour, times[]}, ...])."""
    rows = list(csv.reader(path.read_text().splitlines()))
    header = [c.strip() for c in rows[0]]
    x_label = header[0]
    backends = [parse_header(c) for c in header[1:]]
    series = [{"label": lab, "ident": idt, "colour": col, "times": []}
              for (lab, idt, col) in backends]
    xs: list[float] = []
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        xs.append(float(row[0]))
        for i, cell in enumerate(row[1:]):
            cell = cell.strip()
            series[i]["times"].append(float(cell) if cell else None)
    return x_label, xs, series


def baseline_index(series: list[dict], baseline: str) -> int:
    for i, s in enumerate(series):
        if s["ident"] == baseline:
            return i
    raise SystemExit(f"no baseline column with identity '{baseline}' found")


def render(x_label: str, xs: list[float], series: list[dict], title: str, out: Path,
           baseline: str) -> None:
    apply_theme()
    bi = baseline_index(series, baseline)
    base_times = series[bi]["times"]
    speedups = [
        [(base_times[j] / s["times"][j]) if (s["times"][j] and base_times[j]) else None
         for j in range(len(xs))]
        for s in series
    ]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.3 * len(xs) + 2), 5.0))
    n = len(series)
    width = 0.84 / max(n, 1)
    idx = list(range(len(xs)))
    top = 1.0
    for si, s in enumerate(series):
        sp = speedups[si]
        heights = [v if v is not None else 0.0 for v in sp]
        top = max([top, *heights])
        offs = (si - (n - 1) / 2) * width
        rects = ax.bar([x + offs for x in idx], heights, width,
                       label=s["label"], color=s["colour"])
        for rect, v in zip(rects, sp):
            if v is None or v <= 0:
                continue
            ax.annotate(f"{v:.2f}×",
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=8,
                        color="#2A2F3A")
    ax.axhline(1.0, ls=(0, (4, 2)), lw=1.0, color="#5A6473", zorder=0)
    ax.set_ylim(0, top * 1.30)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"speedup vs {series[bi]['label']} (×)")
    ax.set_title(f"{title} — speedup vs {series[bi]['label']} default (higher is better)",
                 fontsize=12)
    ax.legend(ncol=min(n, 4), frameon=True, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def render_latency(x_label: str, xs: list[float], series: list[dict], title: str, out: Path) -> None:
    """Absolute latency (ms) grouped bars on a **log** y-axis (wide value range)."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.3 * len(xs) + 2), 5.0))
    n = len(series)
    width = 0.84 / max(n, 1)
    idx = list(range(len(xs)))
    lo, hi = float("inf"), 0.0
    for si, s in enumerate(series):
        ys = [v if v is not None else 0.0 for v in s["times"]]
        pos = [v for v in ys if v > 0]
        if pos:
            lo, hi = min(lo, *pos), max(hi, *pos)
        offs = (si - (n - 1) / 2) * width
        rects = ax.bar([x + offs for x in idx], ys, width,
                       label=s["label"], color=s["colour"])
        for rect, v in zip(rects, s["times"]):
            if v is None or v <= 0:
                continue
            ax.annotate(f"{v:.3g}",
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=8,
                        color="#2A2F3A")
    ax.set_yscale("log")
    ax.set_ylim((lo if lo != float("inf") else 0.01) / 2, hi * 3.0)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel(x_label)
    ax.set_ylabel("latency (ms, log)")
    ax.set_title(f"{title} — latency (ms, log scale; lower is better)", fontsize=12)
    ax.legend(ncol=min(n, 4), frameon=True, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def speedup_table(x_label: str, xs: list[float], series: list[dict], baseline: str) -> str:
    bi = baseline_index(series, baseline)
    base = series[bi]["times"]
    others = [s for k, s in enumerate(series) if k != bi]
    head = f"| {x_label} | " + " | ".join(s["label"] for s in others) + " |"
    sep = "|" + "---|" * (len(others) + 1)
    lines = [f"_speedup vs {series[bi]['label']} (bold = fastest at that point)_\n", head, sep]
    for j, x in enumerate(xs):
        vals = []
        for s in others:
            t, b = s["times"][j], base[j]
            vals.append(b / t if (t and b) else None)
        best = max([v for v in vals if v is not None], default=None)
        cells = []
        for v in vals:
            if v is None:
                cells.append("—")
            elif v == best:
                cells.append(f"**{v:.2f}×**")
            else:
                cells.append(f"{v:.2f}×")
        lines.append(f"| {x:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latency_table(x_label: str, xs: list[float], series: list[dict]) -> str:
    head = f"| {x_label} | " + " | ".join(s["label"] for s in series) + " |"
    sep = "|" + "---|" * (len(series) + 1)
    lines = ["_absolute latency (ms), lower is better; bold = fastest_\n", head, sep]
    for j, x in enumerate(xs):
        vals = [s["times"][j] for s in series]
        present = [v for v in vals if v is not None]
        best = min(present) if present else None
        cells = []
        for v in vals:
            if v is None:
                cells.append("—")
            elif v == best:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        lines.append(f"| {x:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--title", required=True)
    p.add_argument("--name", default=None, help="basename for outputs (default: csv stem)")
    p.add_argument("--baseline", default=BASELINE, help="palette identity of the default column")
    args = p.parse_args()

    x_label, xs, series = parse_csv(args.csv_path)
    if not xs:
        raise SystemExit(f"no rows parsed from {args.csv_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.csv_path.stem

    sp_png = args.output_dir / f"{name}_speedup.png"
    render(x_label, xs, series, args.title, sp_png, args.baseline)
    lat_png = args.output_dir / f"{name}_latency.png"
    render_latency(x_label, xs, series, args.title, lat_png)

    md = [
        f"# {args.title}\n",
        f"_Source: `{args.csv_path}`_\n",
        "## Speedup vs default (×)\n",
        f"_higher is better. See `{sp_png.name}`._\n",
        speedup_table(x_label, xs, series, args.baseline) + "\n",
        f"![speedup]({sp_png.name})\n",
        "### Absolute latency (ms)\n",
        f"_log scale, lower is better. See `{lat_png.name}`._\n",
        latency_table(x_label, xs, series) + "\n",
        f"![latency]({lat_png.name})\n",
    ]
    md_path = args.output_dir / f"{name}.md"
    md_path.write_text("\n".join(md))
    print(f"wrote {md_path}")
    print(f"wrote {sp_png}")
    print(f"wrote {lat_png}")


if __name__ == "__main__":
    main()

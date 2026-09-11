"""Render benchmark CSV files into grouped bar plots.

The benchmark runner's job is to write a complete long-form CSV. This script is
the separate plotting step: it reads that CSV and writes publication assets.
Plots are grouped bar charts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from miniworld_engine.viz import (
    apply_theme,
    canonical,
    label_for,
    save_figure,
    sort_backends,
    style_for,
)

BASELINE = "pytorch"
ACTUAL_FIELDS = ("compiled", "cudagraph", "compile_scope", "measurement_scope", "parameter_dtype")


def _measured_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("value") and math.isfinite(float(row["value"]))]


def series_label(impl: str, rows: list[dict[str, str]]) -> str:
    measured = _measured_rows(rows)
    selected = [row for row in measured if canonical(row["implementation"]) == impl]
    differing = [field for field in ACTUAL_FIELDS
                 if len({row.get(field, "") for row in measured}) > 1]
    details = [f"{field}={selected[0].get(field) or 'unknown'}" for field in differing]
    if impl != BASELINE and selected and all(
        row.get("execution_path") == "module.reference.torch" for row in selected
    ):
        details.append("execution=pytorch reference")
    return label_for(impl) + (" [" + ", ".join(details) + "]" if details else "")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def filtered_rows(rows: list[dict[str, str]], metric: str | None, mode: str | None) -> list[dict[str, str]]:
    out = rows
    if metric is not None:
        out = [row for row in out if row["metric"] == metric]
    if mode is not None:
        out = [row for row in out if row["mode"] == mode]
    return out


def infer_x_field(rows: list[dict[str, str]], requested: str | None) -> str:
    if requested is not None:
        return requested
    d_pairs = {row["d_pair"] for row in rows}
    seq_lens = {row["seq_len"] for row in rows}
    if len(d_pairs) > 1 and len(seq_lens) == 1:
        return "d_pair"
    return "seq_len"


def x_label(x_field: str) -> str:
    return "L" if x_field == "seq_len" else "d_pair"


def series_by_impl(rows: list[dict[str, str]], x_field: str) -> dict[str, list[tuple[int, float]]]:
    validate_comparison(rows, x_field)
    series: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        if not row["value"]:
            continue
        value = float(row["value"])
        if not math.isfinite(value):
            continue
        impl = canonical(row["implementation"])
        x_value = int(row[x_field])
        if x_value in series[impl]:
            raise ValueError(f"duplicate measurement for {impl!r}, {x_field}={x_value}; "
                             "select one run or explicitly aggregate repetitions first")
        series[impl][x_value] = value
    return {impl: sorted(points.items()) for impl, points in series.items()}


def validate_comparison(rows: list[dict[str, str]], x_field: str) -> None:
    """Never let first-row captions disguise a mixed-condition comparison."""
    schemas = {row.get("measurement_schema", "") for row in rows}
    if schemas != {"2"} and not all(row.get("_allow_legacy_unverified") == "true" for row in rows):
        raise ValueError("CSV lacks measurement_schema=2 execution evidence; remeasure or use "
                         "--allow-legacy-unverified for an explicitly unverified historical plot")
    if len(schemas) > 1:
        raise ValueError("mixed measurement schemas cannot share a plot")
    fields = {
        "target", "level", "mode", "metric", "unit", "device", "precision",
        "compile_requested", "cudagraph_requested", "mode_requested",
        "compile_wrap", "torch_version", "cuda_version",
        "run_name", "run_id", "sweep_axis", "allow_tf32", "n_block", "n_blocks", "batch_size",
        "seq_len", "d_pair", "dropout",
    } - {x_field}
    for field in sorted(fields):
        values = {row.get(field, "") for row in rows}
        if len(values) > 1:
            raise ValueError(f"mixed benchmark conditions for {field}: {sorted(values)}; "
                             "filter to one run and condition before plotting")
    measured = _measured_rows(rows)
    for impl in {canonical(row["implementation"]) for row in measured}:
        selected = [row for row in measured if canonical(row["implementation"]) == impl]
        for field in ACTUAL_FIELDS:
            values = {row.get(field, "") for row in selected}
            if len(values) > 1:
                raise ValueError(f"mixed actual {field} for implementation {impl!r}: {sorted(values)}")


def output_stem(rows: list[dict[str, str]], requested: str | None) -> str:
    if requested is not None:
        return requested
    row = rows[0]
    target = row["target"]
    if target == "triangle_multiplication":
        target = "trimul"
    suffix = row["sweep_axis"]
    if suffix == "seq_len":
        suffix = "L_sweep"
    elif suffix == "d_pair":
        suffix = "d_sweep"
    graph_suffix = ""
    if row.get("cudagraph", "disabled") != "disabled":
        graph_suffix = f"_{row['cudagraph']}"
    # Include the complete input, including timings: separate repetitions and
    # different compile/dtype settings must not overwrite one another's plots.
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=True).encode()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    run = re.sub(r"[^A-Za-z0-9_.-]+", "_", row.get("run_name", ""))[:100]
    stem = run or f"{target}_{row['mode']}{graph_suffix}_{suffix}"
    return f"{stem}_{digest}"


def common_title(rows: list[dict[str, str]], fallback: str) -> str:
    if not rows:
        return fallback
    row = rows[0]
    graph = row.get("cudagraph_requested", row.get("cudagraph", "disabled"))
    graph_text = "" if graph == "disabled" else f" {graph}"
    return (
        f"{row['target']} {row['mode']}{graph_text} {row['metric']}"
        + (" [LEGACY UNVERIFIED]" if row.get("measurement_schema") != "2" else "")
    )


def caption(rows: list[dict[str, str]]) -> str:
    row = rows[0]
    measured = _measured_rows(rows)
    def actual(field: str) -> str:
        values = {item.get(field, "") for item in measured}
        return (next(iter(values)) or "unknown") if len(values) == 1 else "see implementation labels"
    fixed = []
    seq_lens = sorted({int(item["seq_len"]) for item in rows})
    d_pairs = sorted({int(item["d_pair"]) for item in rows})
    if len(seq_lens) == 1:
        fixed.append(f"L={seq_lens[0]}")
    if len(d_pairs) == 1:
        fixed.append(f"d_pair={d_pairs[0]}")
    fixed_text = f" | {' '.join(fixed)}" if fixed else ""
    return (
        f"{row['device']} | {row['precision']} | compiled={actual('compiled')} | "
        f"cudagraph={actual('cudagraph')} | "
        f"dropout={row.get('dropout', 'unrecorded')} | "
        f"wrap={row.get('compile_wrap', 'unspecified')} | "
        f"torch={row['torch_version']} cuda={row['cuda_version']}{fixed_text}"
         f"\ncompile_scope={actual('compile_scope')} | measurement_scope={actual('measurement_scope')}"
         f" | parameter_dtype={actual('parameter_dtype')}"
        + (f"\nrun={row.get('run_id') or row.get('run_name')}" if row.get("run_id") or row.get("run_name") else "")
        + (" | LEGACY UNVERIFIED: labels lack execution evidence"
           if row.get("measurement_schema") != "2" else "")
    )


def _annotate_bars(ax, rects, values: list[float | None], fmt: str) -> None:
    for rect, value in zip(rects, values, strict=True):
        if value is None or value <= 0:
            continue
        ax.annotate(
            fmt.format(value),
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
            color="#2A2F3A",
        )


def plot_latency(rows: list[dict[str, str]], out: Path, title: str, x_field: str) -> None:
    apply_theme()
    grouped = series_by_impl(rows, x_field)
    unit = rows[0]["unit"]
    x_values = sorted({x_value for points in grouped.values() for x_value, _ in points})
    impls = sort_backends(list(grouped))
    fig, ax = plt.subplots(figsize=(max(7.2, 1.25 * len(x_values) + 2.0), 4.8))
    xs = list(range(len(x_values)))
    width = 0.82 / max(len(impls), 1)
    hi = 0.0
    for index, impl in enumerate(impls):
        values_by_seq = dict(grouped[impl])
        values = [values_by_seq.get(x_value) for x_value in x_values]
        heights = [value if value is not None else 0.0 for value in values]
        hi = max([hi, *heights])
        offset = (index - (len(impls) - 1) / 2) * width
        color, _linestyle = style_for(impl)
        rects = ax.bar(
            [x + offset for x in xs],
            heights,
            width,
            label=series_label(impl, rows),
            color=color,
        )
        _annotate_bars(ax, rects, values, "{:.3g}")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x_value) for x_value in x_values])
    ax.set_xlabel(x_label(x_field))
    ax.set_ylabel(f"{rows[0]['metric']} ({unit})")
    ax.set_title(f"{title}: lower is better")
    if hi > 0:
        ax.set_ylim(0, hi * 1.35)
    ax.legend(ncol=min(len(impls), 4), loc="upper left")
    fig.text(0.01, 0.01, caption(rows), ha="left", va="bottom", fontsize=8, color="#5A6473")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    save_figure(fig, out)
    plt.close(fig)


def plot_speedup(rows: list[dict[str, str]], out: Path, title: str, baseline: str, x_field: str) -> None:
    apply_theme()
    grouped = series_by_impl(rows, x_field)
    if baseline not in grouped:
        raise SystemExit(f"baseline implementation {baseline!r} not found in CSV")
    base = dict(grouped[baseline])
    x_values = sorted(base)
    impls = sort_backends([name for name in grouped if name != baseline])
    fig, ax = plt.subplots(figsize=(max(7.2, 1.25 * len(x_values) + 2.0), 4.8))
    xs = list(range(len(x_values)))
    width = 0.82 / max(len(impls), 1)
    top = 1.0
    for index, impl in enumerate(impls):
        values_by_seq = dict(grouped[impl])
        speedups: list[float | None] = []
        for x_value in x_values:
            value = values_by_seq.get(x_value)
            base_value = base.get(x_value)
            speedups.append((base_value / value) if base_value is not None and value and value > 0 else None)
        heights = [value if value is not None else 0.0 for value in speedups]
        top = max([top, *heights])
        offset = (index - (len(impls) - 1) / 2) * width
        color, _linestyle = style_for(impl)
        rects = ax.bar(
            [x + offset for x in xs],
            heights,
            width,
            label=series_label(impl, rows),
            color=color,
        )
        _annotate_bars(ax, rects, speedups, "{:.2f}x")
    ax.axhline(1.0, color="#5A6473", linewidth=1.0, linestyle=(0, (4, 2)))
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x_value) for x_value in x_values])
    ax.set_xlabel(x_label(x_field))
    ax.set_ylabel(f"speedup vs {series_label(baseline, rows)} (x)")
    ax.set_title(f"{title}: higher is better")
    ax.set_ylim(0, top * 1.35)
    ax.legend(ncol=min(len(impls), 4), loc="upper left")
    fig.text(0.01, 0.01, caption(rows), ha="left", va="bottom", fontsize=8, color="#5A6473")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    save_figure(fig, out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Where to write the SVGs. Default: the CSV's sibling plots/<gpu>/ dir "
        "(raw CSVs live under artifacts/<gpu>/; figures go to plots/<gpu>/).",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--x-field", choices=("seq_len", "d_pair"), default=None)
    parser.add_argument("--allow-legacy-unverified", action="store_true",
                        help="Render historical CSVs with an explicit unverified label; "
                             "never mix them with schema 2 measurements.")
    args = parser.parse_args()

    if args.output_dir is None:
        # Default the figures next to (but separate from) the raw data: CSVs live under
        # <bench>/artifacts/<gpu>/, figures go to <bench>/plots/<gpu>/. Fall back to the
        # CSV's own directory when it is not under an artifacts/ tree.
        csv_dir = args.csv_path.resolve().parent
        args.output_dir = (
            Path(str(csv_dir).replace("/artifacts/", "/plots/"))
            if "/artifacts/" in str(csv_dir)
            else csv_dir
        )

    rows = filtered_rows(read_rows(args.csv_path), args.metric, args.mode)
    if not rows:
        raise SystemExit(f"no matching rows in {args.csv_path}")
    if args.allow_legacy_unverified:
        rows = [row | {"_allow_legacy_unverified": "true"} for row in rows]

    x_field = infer_x_field(rows, args.x_field)
    # Validate both plots before writing either, including duplicate aliases.
    grouped = series_by_impl(rows, x_field)
    args.baseline = canonical(args.baseline)
    if args.baseline not in grouped:
        raise SystemExit(f"baseline implementation {args.baseline!r} not found in CSV")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = output_stem(rows, args.name)
    title = args.title or common_title(rows, name)
    latency_path = args.output_dir / f"{name}_latency"
    speedup_path = args.output_dir / f"{name}_speedup"
    plot_latency(rows, latency_path, title, x_field)
    plot_speedup(rows, speedup_path, title, args.baseline, x_field)
    print(f"wrote {latency_path.with_suffix('.svg')}")
    print(f"wrote {speedup_path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()

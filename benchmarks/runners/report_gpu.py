"""Render one GPU's curated module results into a docs/reports page.

Input is the tracked layout from ``benchmarks/RESULTS.md``:
``benchmarks/modules/<target>/results/<gpu>/tables/<mode>_<axis>[_<variant>].csv``. Output is
one markdown page plus SVG figures: a sweep figure per module (latency and speedup for every
mode/axis table), a summary table at the anchor shape, and one summary figure across modules.

Only ``measurement_schema=2`` tables are rendered; older tables are listed as excluded rather
than plotted, so a stale number can never appear in a figure. Tables written by a repetition
aggregator may carry ``value_min``/``value_max``/``n_repetitions``/``runtime_cache_misses``
columns; they are used for error bars and disclosure when present.

    python benchmarks/runners/report_gpu.py a6000 --out docs/reports/a6000-module-sweeps.md
"""

from __future__ import annotations

import argparse
import csv
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
)

REPO = Path(__file__).resolve().parents[2]
MODULES = REPO / "benchmarks" / "modules"
BASELINE = "pytorch"
VENDOR = "cuequivariance"
#: implementations that are not part of the report even when a table carries them.
EXCLUDED_IMPLS = frozenset({"old-triton"})
ANCHOR_L = 384
MODES = ("inference", "training")
AXES = ("seq_len", "d_pair")

# Display order and names for module units: (target, variant) -> label.
UNITS: list[tuple[str, str, str]] = [
    ("adaptive_layernorm", "", "AdaLN"),
    ("triangle_multiplication", "", "Trimul outgoing"),
    ("triangle_multiplication", "incoming", "Trimul incoming"),
    ("triangle_multiplication_bidirectional", "", "Bidirectional trimul"),
    ("triangle_attention", "", "Triangle attention"),
    ("transition", "", "Transition"),
    ("conditioned_transition", "", "ConditionedTransition"),
    ("augmented_attention_token", "", "Token attention"),
    ("augmented_attention_atom", "", "Atom attention"),
    ("swa_atom_attention", "", "SWA attention"),
    ("dit", "", "DiT"),
    ("dit_atom", "", "Atom DiT"),
    ("swa_dit", "", "SWA DiT"),
]
TABLE_RE = re.compile(r"^(inference|training)_(seq_len|d_pair)(?:_(.+))?\.csv$")

Row = dict[str, str]


def read_rows(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle)
                if canonical(row["implementation"]) not in EXCLUDED_IMPLS]


def discover(gpu: str) -> dict[tuple[str, str], dict[tuple[str, str], list[Row]]]:
    """(target, variant) -> (mode, axis) -> rows, for every table of ``gpu``."""
    found: dict[tuple[str, str], dict[tuple[str, str], list[Row]]] = defaultdict(dict)
    for table in sorted(MODULES.glob(f"*/results/{gpu}/tables/*.csv")):
        match = TABLE_RE.match(table.name)
        if not match:
            continue
        mode, axis, variant = match.group(1), match.group(2), match.group(3) or ""
        found[(table.parents[3].name, variant)][(mode, axis)] = read_rows(table)
    return found


def is_current(rows: list[Row]) -> bool:
    return bool(rows) and all(row.get("measurement_schema") == "2" for row in rows)


def value_of(row: Row) -> float | None:
    if row.get("status") != "ok" or not row.get("value"):
        return None
    value = float(row["value"])
    return value if math.isfinite(value) and value > 0 else None


def series(rows: list[Row], axis: str) -> dict[str, dict[int, Row]]:
    out: dict[str, dict[int, Row]] = defaultdict(dict)
    for row in rows:
        out[canonical(row["implementation"])][int(row[axis])] = row
    return out


def fixed_dims(rows: list[Row], axis: str) -> str:
    other = "d_pair" if axis == "seq_len" else "seq_len"
    values = sorted({int(row[other]) for row in rows})
    name = "d_pair" if other == "d_pair" else "L"
    return f"{name}={values[0]}" if len(values) == 1 else f"{name} in {values}"


def x_name(axis: str) -> str:
    return "L" if axis == "seq_len" else "d_pair"


def _bars(ax, rows: list[Row], axis: str, speedup: bool) -> None:
    grouped = series(rows, axis)
    xs = sorted({x for points in grouped.values() for x in points})
    impls = sort_backends([name for name in grouped if not speedup or name != BASELINE])
    width = 0.82 / max(len(impls), 1)
    base = grouped.get(BASELINE, {})
    top = 1.0 if speedup else 0.0
    for index, impl in enumerate(impls):
        heights, err_lo, err_hi, labels = [], [], [], []
        for x in xs:
            row = grouped[impl].get(x)
            value = value_of(row) if row else None
            if speedup:
                base_value = value_of(base[x]) if x in base else None
                ratio = base_value / value if base_value and value else None
                heights.append(ratio or 0.0)
                labels.append(f"{ratio:.2f}x" if ratio else ("n/a" if row else ""))
                err_lo.append(0.0)
                err_hi.append(0.0)
            else:
                heights.append(value or 0.0)
                labels.append(f"{value:.3g}" if value else (row["status"] if row else ""))
                lo = float(row["value_min"]) if value and row and row.get("value_min") else None
                hi = float(row["value_max"]) if value and row and row.get("value_max") else None
                err_lo.append(value - lo if value and lo is not None else 0.0)
                err_hi.append(hi - value if value and hi is not None else 0.0)
        top = max([top, *heights])
        offset = (index - (len(impls) - 1) / 2) * width
        positions = [i + offset for i in range(len(xs))]
        rects = ax.bar(positions, heights, width, label=label_for(impl), color=_color(impl),
                       yerr=None if speedup else [err_lo, err_hi],
                       error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#2A2F3A"})
        for rect, text in zip(rects, labels, strict=True):
            if text:
                ax.annotate(text, xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", rotation=90, fontsize=7.5, color="#2A2F3A")
    if speedup:
        ax.axhline(1.0, color="#5A6473", linewidth=1.0, linestyle=(0, (4, 2)))
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(x_name(axis))
    ax.set_ylim(0, top * 1.45 if top > 0 else 1.0)


def _color(impl: str) -> str:
    from miniworld_engine.viz import color_for

    return color_for(impl)


def module_figure(label: str, tables: dict[tuple[str, str], list[Row]], out: Path) -> Path:
    apply_theme()
    keys = [(mode, axis) for mode in MODES for axis in AXES if (mode, axis) in tables]
    fig, axes = plt.subplots(2, len(keys), figsize=(4.6 * len(keys), 7.6), squeeze=False)
    for column, key in enumerate(keys):
        rows = tables[key]
        mode, axis = key
        _bars(axes[0][column], rows, axis, speedup=False)
        axes[0][column].set_title(f"{mode} latency ({fixed_dims(rows, axis)})")
        axes[0][column].set_ylabel(f"{rows[0]['metric']} ({rows[0]['unit']}), lower is better")
        _bars(axes[1][column], rows, axis, speedup=True)
        axes[1][column].set_title(f"{mode} speedup vs {label_for(BASELINE)}")
        axes[1][column].set_ylabel("speedup (x), higher is better")
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            for handle, text in zip(*ax.get_legend_handles_labels(), strict=True):
                if text not in labels:
                    handles.append(handle)
                    labels.append(text)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5),
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{label} on {tables[keys[0]][0]['device']}", y=1.06)
    fig.text(0.01, 0.005, caption(tables[keys[0]]), ha="left", va="bottom", fontsize=7.5,
             color="#5A6473")
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    written = save_figure(fig, out)
    plt.close(fig)
    return written[0]


def caption(rows: list[Row]) -> str:
    row = rows[0]
    reps = {r.get("n_repetitions", "") for r in rows if r.get("status") == "ok"}
    rep_text = (f" | median of {'/'.join(sorted(reps))} process repetitions, "
                "error bars min-max") if reps - {""} else ""
    card = f" [{row['gpu_uuid'][:8]}@{row.get('host', '?')}]" if row.get("gpu_uuid") else ""
    return (f"{row['device']}{card} | {row['precision']} | compiled={row['compiled']} | "
            f"cudagraph={row['cudagraph']} | dropout={row.get('dropout', '?')} | "
            f"mask_prob={row.get('mask_prob', '?')} | tf32={row.get('allow_tf32', '?')} | "
            f"A={row.get('n_augment', '?')} | torch={row['torch_version']} "
            f"cuda={row['cuda_version']}{rep_text}")


def anchor(rows: list[Row]) -> dict[str, Row]:
    grouped = series(rows, "seq_len")
    lengths = sorted({x for points in grouped.values() for x in points})
    length = ANCHOR_L if ANCHOR_L in lengths else lengths[0]
    return {impl: points[length] for impl, points in grouped.items() if length in points}


def speedup_range(rows: list[Row], impl: str) -> tuple[float, float] | None:
    grouped = series(rows, "seq_len")
    ratios = []
    for x, row in grouped.get(impl, {}).items():
        base = grouped.get(BASELINE, {}).get(x)
        value, base_value = value_of(row), value_of(base) if base else None
        if value and base_value:
            ratios.append(base_value / value)
    return (min(ratios), max(ratios)) if ratios else None


OOM_ALLOC_RE = re.compile(r"Tried to allocate ([0-9.]+ [GM]iB)")
CAPACITY_RE = re.compile(r"total capacity of ([0-9.]+) GiB")


def reason(row: Row) -> str:
    error = " ".join(row.get("error", "").split())
    if "OutOfMemoryError" in error:
        alloc = OOM_ALLOC_RE.search(error)
        cap = CAPACITY_RE.search(error)
        detail = ", ".join(x for x in (f"tried to allocate {alloc.group(1)}" if alloc else "",
                                       f"{cap.group(1)} GiB card" if cap else "") if x)
        return f"out of memory{': ' + detail if detail else ''}"
    return f"{row['status']}: {error[:90]}" if error else row["status"]


def card_capacities(found: dict) -> dict[str, str]:
    """Total memory each card reported, read from the out-of-memory errors it produced."""
    out: dict[str, str] = {}
    for tables in found.values():
        for rows in tables.values():
            for row in rows:
                cap = CAPACITY_RE.search(row.get("error", ""))
                if cap and row.get("gpu_uuid"):
                    out[row["gpu_uuid"][:8]] = cap.group(1)
    return out


def fmt(value: float | None, digits: int = 3, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def summary_figure(entries: list[dict], device: str, out: Path) -> Path:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(entries) + 2.2), sharey=True)
    ys = list(range(len(entries)))[::-1]
    for ax, mode in zip(axes, MODES, strict=True):
        vendor = [e[mode].get("vendor_speedup") for e in entries]
        ours = [e[mode].get("miniworld_speedup") for e in entries]
        height = 0.38
        ax.barh([y + height / 2 for y in ys], [v or 0.0 for v in ours], height,
                color=_color("miniworld"), label=label_for("miniworld"))
        if any(vendor):
            ax.barh([y - height / 2 for y in ys], [v or 0.0 for v in vendor], height,
                    color=_color(VENDOR), label=label_for(VENDOR))
        for y, value in zip(ys, ours, strict=True):
            ax.annotate(f"{value:.2f}x" if value else "n/a", xy=(value or 0.0, y + height / 2),
                        xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
        for y, value in zip(ys, vendor, strict=True):
            if value:
                ax.annotate(f"{value:.2f}x", xy=(value, y - height / 2), xytext=(3, 0),
                            textcoords="offset points", va="center", fontsize=8)
        ax.axvline(1.0, color="#5A6473", linewidth=1.0, linestyle=(0, (4, 2)))
        ax.set_yticks(ys)
        ax.set_yticklabels([e["label"] for e in entries])
        ax.set_xlabel(f"speedup vs compiled {label_for(BASELINE)} (x), higher is better")
        ax.set_title(f"{mode} at L={ANCHOR_L}, d_pair=128")
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(len(labels), 1),
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Module speedup summary on {device}")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    written = save_figure(fig, out)
    plt.close(fig)
    return written[0]


def build(gpu: str, out_md: Path, figure_dir: Path, title: str | None) -> None:
    found = discover(gpu)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rel = lambda p: p.relative_to(out_md.parent).as_posix()
    known = {(t, v): label for t, v, label in UNITS}
    units = list(UNITS) + [(t, v, f"{t}{' ' + v if v else ''}") for t, v in found
                           if (t, v) not in known]
    entries, excluded, figures, notes = [], [], [], []
    device = ""
    source_hashes: set[str] = set()
    cards: dict[str, list[str]] = defaultdict(list)
    sample: list[Row] | None = None
    for target, variant, label in units:
        tables = found.get((target, variant), {})
        current = {key: rows for key, rows in tables.items() if is_current(rows)}
        stale = sorted(f"{m}_{a}" for (m, a), rows in tables.items() if not is_current(rows))
        if stale:
            excluded.append(f"{label}: {', '.join(stale)}")
        if not current:
            continue
        device = device or next(iter(current.values()))[0]["device"]
        sample = sample or next(iter(current.values()))
        source_hashes |= {r.get("source_hash", "") for rows in current.values() for r in rows}
        for (mode, axis), rows in sorted(current.items()):
            card = rows[0].get("gpu_uuid", "")
            if card:
                cards[f"{card[:8]}@{rows[0].get('host', '?')}"].append(f"{label} {mode} {x_name(axis)}")
        slug = f"{target}{'_' + variant if variant else ''}"
        figures.append((label, rel(module_figure(label, current, figure_dir / slug))))
        entry: dict = {"label": label}
        for mode in MODES:
            cell: dict = {}
            rows = current.get((mode, "seq_len"))
            if rows:
                at = anchor(rows)
                pt, mw = at.get(BASELINE), at.get("miniworld")
                vendor = VENDOR if VENDOR in at else None
                cell = {
                    "L": next(iter(at.values()))["seq_len"] if at else "",
                    "pytorch": value_of(pt) if pt else None,
                    "miniworld": value_of(mw) if mw else None,
                    "vendor": vendor,
                    "vendor_ms": value_of(at[vendor]) if vendor else None,
                    "range": speedup_range(rows, "miniworld"),
                }
                if cell["pytorch"] and cell["miniworld"]:
                    cell["miniworld_speedup"] = cell["pytorch"] / cell["miniworld"]
                if cell["pytorch"] and cell["vendor_ms"]:
                    cell["vendor_speedup"] = cell["pytorch"] / cell["vendor_ms"]
            entry[mode] = cell
            for (m, axis), table_rows in current.items():
                if m != mode:
                    continue
                failed = [r for r in table_rows if r.get("status") != "ok"]
                missed = [r for r in table_rows if r.get("runtime_cache_misses", "0") not in ("", "0")]
                if failed:
                    points = ", ".join(f"{label_for(r['implementation'])} {x_name(axis)}="
                                       f"{r[axis]} ({reason(r)})" for r in failed)
                    notes.append(f"{label} {mode} {x_name(axis)} sweep: not measured at {points}.")
                for r in missed:
                    ops = r.get("runtime_cache_miss_ops", "").replace("|", ", ")
                    notes.append(f"{label} {mode} {x_name(axis)} sweep, "
                                 f"{label_for(r['implementation'])} at {x_name(axis)}={r[axis]}: "
                                 f"{r['runtime_cache_misses']} tuned-cache miss(es) fell back to "
                                 f"the bounded heuristic search{' (' + ops + ')' if ops else ''}.")
        entries.append(entry)
    if not entries:
        raise SystemExit(f"no measurement_schema=2 tables found for gpu={gpu!r}")
    summary_svg = rel(summary_figure(entries, device, figure_dir / "summary_speedup"))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render(gpu, device, title, entries, figures, summary_svg, excluded, notes,
                             sample, source_hashes - {""}, dict(cards), card_capacities(found)))
    print(f"wrote {out_md} ({len(figures)} module figures)")


def card_notes(cards: dict[str, list[str]], capacities: dict[str, str]) -> list[str]:
    """One bullet per physical card when the tables span more than one.

    Nominally identical cards of one model differ in absolute latency (clocks, ECC, thermal
    state), so absolute milliseconds are comparable within a table, and speedups across tables;
    the mapping lets a reader check which tables share a card.
    """
    if len(cards) < 2:
        return []
    lines = [f"- Tables were measured on {len(cards)} physical cards, so absolute latencies are "
             "comparable within a table and speedup ratios across tables. Card by table (each "
             "figure caption repeats its card as `[uuid@host]`):"]
    for card, tables in sorted(cards.items()):
        cap = capacities.get(card.split("@")[0])
        lines.append(f"  - `{card}`{f' (reports {cap} GiB total)' if cap else ''}: "
                     + ", ".join(tables))
    if len(set(capacities.values())) > 1:
        lines.append("- Cards of this model reported different total capacities in their "
                     "out-of-memory errors (" + ", ".join(f"`{c}` {v} GiB" for c, v in
                                                          sorted(capacities.items()))
                     + "); a smaller reported capacity is consistent with ECC being enabled on "
                     "that card, which also costs bandwidth, so its absolute latencies are not "
                     "interchangeable with the other cards'.")
    return lines


def render(gpu: str, device: str, title: str | None, entries: list[dict],
           figures: list[tuple[str, str]], summary_svg: str, excluded: list[str],
           notes: list[str], sample: list[Row] | None, source_hashes: set[str],
           cards: dict[str, list[str]], capacities: dict[str, str]) -> str:
    row = sample[0] if sample else {}
    hashes = ", ".join(f"`{h[:12]}`" for h in sorted(source_hashes)) or "?"
    lines = [f"# {title or f'{device} module benchmark report'}", ""]
    lines += [
        f"Generated by `benchmarks/runners/report_gpu.py {gpu}` from the tracked tables under "
        f"`benchmarks/modules/<target>/results/{gpu}/tables/`. Every number here is a "
        "`measurement_schema=2` measurement of the `torch.compile`d path; tables that predate "
        "that schema are excluded below rather than plotted.", "",
        "## Workload", "",
        f"- Device: {device}; torch {row.get('torch_version', '?')}, "
        f"CUDA {row.get('cuda_version', '?')}; benchmark source hash(es) {hashes}"
        + (" (a table measured after a later benchmark-harness commit carries its own hash; "
           "the tables name theirs in the `source_hash` column)." if len(source_hashes) > 1
           else "."),
        f"- Precision {row.get('precision', '?')}, TF32 {'off' if row.get('allow_tf32') == 'False' else 'on'}, "
        f"mask probability {row.get('mask_prob', '?')}, depth {row.get('n_layers', '?')}, "
        f"compile wrap `{row.get('compile_wrap', '?')}`.",
        "- Inference: augmentation A5, manual CUDA Graph capture of the measured call, "
        "dropout 0. Training: A48, no graph, forward+backward without optimizer, dropout "
        "0.25 where the module has dropout (triangle modules).",
        "- Pair modules sweep L at d_pair=128 and d_pair at L=384; token modules use "
        "d_single=768; atom modules use 8xL atoms with d_single_atom=128. Sweep ranges are "
        "each target's `configs/bench.yaml`.",
        "- Each cell is the median of three independent benchmark processes; error bars are "
        "the min-max across those processes. Speedup is compiled PyTorch time divided by the "
        "implementation's time at the same point. cuEquivariance appears only for the modules "
        "it implements (triangle multiplication and triangle attention).",
        "- Rows from different modules may come from different physical cards; all "
        "implementations within one table share a card and a process. Do not sum rows into a "
        "whole-model claim.",
        *card_notes(cards, capacities),
        "## Summary", "",
        f"![Module speedup summary]({summary_svg})", "",
        f"Latency in ms at L={ANCHOR_L}, d_pair=128 (atom modules: {ANCHOR_L * 8} atoms). "
        "The last column is the MiniWorld speedup range over the whole L sweep.", "",
        "| Module | Mode | PyTorch | cuEquivariance | MiniWorld | MiniWorld speedup | "
        "L-sweep range |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        for mode in MODES:
            cell = entry.get(mode) or {}
            vendor = cell.get("vendor")
            vendor_text = fmt(cell.get("vendor_ms")) if vendor else "—"
            rng = cell.get("range")
            lines.append(
                f"| {entry['label']} | {mode} | {fmt(cell.get('pytorch'))} | {vendor_text} | "
                f"{fmt(cell.get('miniworld'))} | {fmt(cell.get('miniworld_speedup'), 2, 'x')} | "
                f"{fmt(rng[0], 2) + '-' + fmt(rng[1], 2) + 'x' if rng else '—'} |")
    lines += ["", "## Per-module sweeps", "",
              "Top row: latency (lower is better). Bottom row: speedup versus compiled PyTorch "
              "(higher is better). Missing bars are points the implementation could not run; "
              "the reason is recorded in the table's `status`/`error` columns.", ""]
    for label, svg in figures:
        lines += [f"### {label}", "", f"![{label} sweeps]({svg})", ""]
    if notes:
        lines += ["## Disclosures", ""] + [f"- {note}" for note in notes] + [""]
    if excluded:
        lines += ["## Excluded legacy tables", "",
                  "These tables lack `measurement_schema=2` execution evidence and are not "
                  "plotted:", ""] + [f"- {item}" for item in excluded] + [""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gpu", help="results folder name, e.g. a6000")
    parser.add_argument("--out", type=Path, default=None,
                        help="markdown path (default docs/reports/<gpu>-module-sweeps.md)")
    parser.add_argument("--figures", type=Path, default=None,
                        help="figure directory (default <out stem>/ next to the markdown)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    out_md = args.out or REPO / "docs" / "reports" / f"{args.gpu}-module-sweeps.md"
    figure_dir = args.figures or out_md.with_suffix("")
    build(args.gpu, out_md, figure_dir, args.title)


if __name__ == "__main__":
    main()

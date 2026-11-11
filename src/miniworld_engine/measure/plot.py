"""Benchmark figures and tables from speed results that already exist. It never measures.

Usage::

    python -m miniworld_engine.measure.plot [--gpu h100_sm90] [--module Transition ...] [--source tokens|RECORDS...] [--out benchmarks]

Source: by default the card's speed tokens (``judge/tokens/<gpu>/*.jsonl``, the judged current result of every requirement case); or
speed record files / directories written by :mod:`miniworld_engine.measure.speed` (``--source DIR``), e.g. a run not issued yet. Only
``status: ok`` CUDA-graph timings are plotted; a capture fallback (``timed_as: eager``) is listed in the tables, marked, and left
out of the figures; an ``unsupported`` / ``skipped`` / ``failed`` backend leaves a gap.

Output (``<out>/<gpu>/``):

- ``<Module>/L__<slice>.svg``: the L sweep -- median latency (log scale) and speedup vs ``pytorch`` over the case length, one
  column per (dtype, mode), the engine (cuda) against anthropic, cueq and pytorch (the tables list every backend), for one
  slice (model, stream, constructor kwargs, depth, samples);
- ``<Module>/D-<key>__<slice>.svg``: the D sweep -- the same over one channel width of the constructor (``d_pair``, ``d_hidden``,
  ``normalized_shape`` ...) at one length (the length most of the widths share), the other kwargs fixed;
- ``<Module>/README.md``: every figure with its table (median ms per backend, speedup vs ``pytorch``);
- ``README.md``: the index of the card (classes, figure counts, the source and its versions).

The figures are regenerated, not edited: run this after every measurement cycle that changed speed results.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

from miniworld_engine.measure import config as cfgmod

HERE = Path(__file__).resolve()
TOKENS = HERE.parents[1] / "judge" / "tokens"
REPO = HERE.parents[3]
#: legend / line order: the reference first, the engine rungs, then the external baselines
BACKENDS = ("pytorch", "triton", "cuda", "anthropic", "cueq")
LABEL = {"pytorch": "PyTorch", "triton": "Engine (Triton)", "cuda": "Engine (CUDA)", "anthropic": "Anthropic", "cueq": "cuEquivariance"}
COLOR = {"pytorch": "#8A93A3", "triton": "#2E6FDB", "cuda": "#B8860B", "anthropic": "#D9622B", "cueq": "#11A6A0"}
MARKER = {"pytorch": "o", "triton": "s", "cuda": "D", "anthropic": "^", "cueq": "v"}
#: drawn in the figures: the engine (cuda) against the external baselines (anthropic, cueq) and the reference (pytorch); the
#: tables list every backend (triton included). --backends overrides.
FIGURE_BACKENDS = ("pytorch", "cuda", "anthropic", "cueq")
#: constructor kwargs that are channel widths: a D sweep runs over one of them
WIDTHS = ("d_pair", "d_single", "d_hidden", "d_msa", "d_cond", "d_atom", "d_token", "normalized_shape", "n_head")
PANEL_ORDER = {
    ("float32", "eval"): 0,
    ("master", "eval"): 1,
    ("bfloat16", "eval"): 2,
    ("float32", "train"): 3,
    ("master", "train"): 4,
    ("bfloat16", "train"): 5,
}


# ------------------------------------------------------------------------------------------------------------------------- input
def _lines(paths):
    for p in paths:
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    yield json.loads(line)


def load(source: list[str], gpu: str) -> tuple[list[dict], str]:
    """Speed points ``{case fields..., median, p10, p90, status, eager}``; the description of the source."""
    if source == ["tokens"]:
        paths, what = [TOKENS / gpu], f"speed tokens `judge/tokens/{gpu}/`"
    else:
        paths, what = [Path(s) for s in source], "speed records " + ", ".join(f"`{s}`" for s in source)
    out = []
    wanted = None
    if source == ["tokens"]:  # only the cases the requirements ask for now (not the unused tokens)
        from miniworld_engine import judge

        wanted = {judge.case_key("speed", r) for r in judge.required("speed")}
    for rec in _lines(paths):
        if "case" in rec:  # a token
            if rec.get("kind") != "speed" or (wanted is not None and rec.get("case_key") not in wanted):
                continue
            case, res = rec["case"], rec["result"]
        else:  # a speed record (measure/speed.py)
            if "backend" not in rec or "module" not in rec or rec.get("gpu", gpu) != gpu:
                continue
            case, res = rec, rec
        out.append(
            {
                **{
                    k: case.get(k)
                    for k in (
                        "row",
                        "module",
                        "model",
                        "stream",
                        "kwargs",
                        "length",
                        "depth",
                        "samples",
                        "dtype",
                        "mode",
                        "backend",
                        "timing",
                    )
                },
                "median": res.get("median_ms"),
                "p10": res.get("p10_ms"),
                "p90": res.get("p90_ms"),
                "status": res.get("status"),
                "eager": res.get("timed_as") == "eager",
            }
        )
    return out, what


# ------------------------------------------------------------------------------------------------------------------------ slices
def _kw(kwargs: dict) -> str:
    return ";".join(f"{k}={v}" for k, v in sorted((kwargs or {}).items())) or "-"


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9.=_-]+", "_", text).strip("_")
    return s if len(s) <= 110 else s[:96] + "_" + format(sum(map(ord, s)) % 65536, "04x")


def _fixed(p: dict, drop: tuple[str, ...] = ()) -> tuple:
    kw = {k: v for k, v in (p["kwargs"] or {}).items() if k not in drop}
    return (p["model"] or "", p["stream"] or "", _kw(kw), p["depth"], p["samples"], p["timing"])


def l_sweeps(points: list[dict]) -> dict[tuple, list[dict]]:
    """slice -> points, for slices with at least two lengths."""
    groups = collections.defaultdict(list)
    for p in points:
        groups[_fixed(p)].append(p)
    return {k: v for k, v in groups.items() if len({p["length"] for p in v}) >= 2}


def _widths(p: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in (p["kwargs"] or {}).items() if k in WIDTHS))


def d_sweeps(points: list[dict]) -> dict[tuple, tuple[str, int, list[dict]]]:
    """(key, slice without the widths) -> (key, length, points): a channel width with at least three values at one length.

    The other widths may change with it (``n_head`` with ``d_pair``, ``d_hidden`` only at ``d_pair=64``) as long as each value of
    the key comes with one set of widths; the x labels then name them."""
    out, seen = {}, set()
    keys = {k for p in points for k, v in (p["kwargs"] or {}).items() if k in WIDTHS and isinstance(v, int | float)}
    for key in sorted(keys):
        groups = collections.defaultdict(list)
        for p in points:
            if key in (p["kwargs"] or {}):
                groups[("", *_fixed(p, WIDTHS)[1:])].append(p)  # the model set differs per width; the case does not
        for fixed, pts in groups.items():
            per_len = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
            for p in pts:
                per_len[p["length"]][p["kwargs"][key]][_widths(p)] += 1
            length, by_value = max(per_len.items(), key=lambda kv: (len(kv[1]), kv[0]))
            if len(by_value) >= 3:  # one set of widths per value: the most measured one (e.g. d_pair=64 without d_hidden)
                keep = {(v, c.most_common(1)[0][0]) for v, c in by_value.items()}
                sel = [p for p in pts if p["length"] == length and (p["kwargs"][key], _widths(p)) in keep]
                ids = frozenset(map(id, sel))
                if ids not in seen:  # n_head moving with d_pair is the d_pair sweep again
                    seen.add(ids)
                    out[(key, fixed)] = (key, length, sel)
    return out


def _d_label(key: str, pts: list[dict]):
    """x of a D-sweep point: the key's value, with the widths that change alongside it."""
    moving = {k for k in WIDTHS if k != key and len({(p["kwargs"] or {}).get(k) for p in pts}) > 1}

    def x_of(p: dict):
        v = p["kwargs"][key]
        extra = " ".join(f"{k}={p['kwargs'][k]}" for k in sorted(moving) if k in p["kwargs"])
        return (v, extra)

    return x_of


def _xs(x) -> str:
    """a tick / table label: a length, or a D-sweep (value, widths that change with it)"""
    return (f"{x[0]}\n{x[1]}" if x[1] else str(x[0])) if isinstance(x, tuple) else str(x)


def _title(fixed: tuple) -> str:
    model, stream, kw, depth, samples, timing = fixed
    parts = [x for x in (model, stream) if x and x != "-"] + [kw, (f"D{depth}" if depth is not None else ""), f"S{samples}", timing]
    return " | ".join(str(x) for x in parts if x not in (None, ""))


def _panels(pts: list[dict]) -> list[tuple[str, str]]:
    return sorted({(p["dtype"], p["mode"]) for p in pts}, key=lambda dm: (PANEL_ORDER.get(dm, 9), dm))


def _series(pts: list[dict], x_of) -> dict[tuple, dict[str, dict]]:
    """(dtype, mode) -> backend -> x -> point (one per x; a duplicate keeps the faster, never averaged silently)."""
    s: dict[tuple, dict[str, dict]] = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in pts:
        x, cur = x_of(p), s[(p["dtype"], p["mode"])][p["backend"]]
        if x not in cur or (p["median"] or math.inf) < (cur[x]["median"] or math.inf):
            cur[x] = p
    return s


def _ok(p: dict | None) -> float | None:
    return p["median"] if p and p["status"] == "ok" and not p["eager"] and p["median"] else None


# ----------------------------------------------------------------------------------------------------------------------- figures
def figure(pts: list[dict], x_of, x_label: str, title: str, out: Path, card: str, backends=FIGURE_BACKENDS) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E3E6EB",
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#5A6473",
        }
    )
    panels, series = _panels(pts), _series(pts, x_of)
    xs = sorted({x_of(p) for p in pts})
    fig, axes = plt.subplots(2, len(panels), figsize=(max(4.2 * len(panels), 6.0), 6.4), squeeze=False, sharex=True)
    handles = {}
    for j, dm in enumerate(panels):
        lat, spd = axes[0][j], axes[1][j]
        base = series[dm].get("pytorch", {})
        for b in [b for b in backends if b in series[dm]]:
            ys = [_ok(series[dm][b].get(x)) for x in xs]
            if not any(ys):
                continue
            kw = {
                "color": COLOR.get(b, "#444444"),
                "marker": MARKER.get(b, "x"),
                "markersize": 4,
                "linewidth": 1.6,
                "label": LABEL.get(b, b),
            }
            (h,) = lat.plot([i for i, y in enumerate(ys) if y], [y for y in ys if y], **kw)
            handles.setdefault(b, h)
            sp = [(i, b0 / y) for i, (x, y) in enumerate(zip(xs, ys, strict=True)) if y and (b0 := _ok(base.get(x)))]
            if sp and b != "pytorch":
                spd.plot([i for i, _ in sp], [v for _, v in sp], **kw)
        lat.set_yscale("log")
        lat.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        lat.yaxis.set_minor_formatter(ticker.NullFormatter())
        lat.set_title(f"{dm[0]} {dm[1]}", fontsize=10)
        lat.set_ylabel("median latency (ms, log)" if j == 0 else "")
        spd.axhline(1.0, color="#5A6473", linewidth=1.0, linestyle=(0, (4, 2)))
        spd.set_ylabel("speedup vs PyTorch (x)" if j == 0 else "")
        spd.set_xticks(
            range(len(xs)),
            [_xs(x) for x in xs],
            rotation=45 if len(xs) > 6 else 0,
            fontsize=8 if all(not isinstance(x, tuple) or not x[1] for x in xs) else 6.5,
        )
        spd.set_xlabel(x_label)
    order = [b for b in BACKENDS if b in handles] + sorted(set(handles) - set(BACKENDS))
    fig.legend(
        [handles[b] for b in order],
        [LABEL.get(b, b) for b in order],
        loc="upper center",
        ncol=len(order),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(title, y=0.945, fontsize=10)
    fig.text(
        0.01, 0.005, f"{card} | CUDA graph, L2 flushed, median | gaps: unsupported, skipped or eager-only", fontsize=7, color="#5A6473"
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format=out.suffix.lstrip(".") or "svg", dpi=110)
    plt.close(fig)


def table(pts: list[dict], x_of, x_name: str) -> str:
    panels, series = _panels(pts), _series(pts, x_of)
    xs = sorted({x_of(p) for p in pts})
    out = []
    for dm in panels:
        bs = [b for b in BACKENDS if b in series[dm]] + sorted(set(series[dm]) - set(BACKENDS))
        out.append(f"**{dm[0]} {dm[1]}** (median ms; x = speedup vs PyTorch)\n")
        out.append("| " + x_name + " | " + " | ".join(LABEL.get(b, b) for b in bs) + " |")
        out.append("|---" * (len(bs) + 1) + "|")
        for x in xs:
            base = _ok(series[dm].get("pytorch", {}).get(x))
            cells = []
            for b in bs:
                p = series[dm][b].get(x)
                if p is None:
                    cells.append("")
                elif p["status"] != "ok":
                    cells.append(str(p["status"]))
                else:
                    v = f"{p['median']:.4g}" + (" (eager)" if p["eager"] else "")
                    if b != "pytorch" and base and not p["eager"]:
                        v += f" ({base / p['median']:.2f}x)"
                    cells.append(v)
            out.append(f"| {_xs(x).replace(chr(10), ' ')} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


# -------------------------------------------------------------------------------------------------------------------------- main
def render(
    points: list[dict],
    out_dir: Path,
    card: str,
    what: str,
    modules: list[str] | None,
    backends: tuple[str, ...] = FIGURE_BACKENDS,
    model_summaries: tuple[str, ...] = ("af3",),
) -> None:
    by_mod = collections.defaultdict(list)
    for p in points:
        if not modules or p["module"] in modules:
            by_mod[p["module"]].append(p)
    index = []
    for mod in sorted(by_mod):
        pts, mdir, sections = by_mod[mod], out_dir / mod, []
        names: set[str] = set()
        for old in mdir.glob("*.svg") if mdir.exists() else ():
            old.unlink()  # a figure whose slice is gone must not linger

        def name(stem: str, names: set[str] = names) -> str:
            n, i = _slug(stem), 2
            while n in names:
                n, i = f"{_slug(stem)}-{i}", i + 1
            names.add(n)
            return n + ".svg"

        n_l = n_d = 0
        swept = l_sweeps(pts)
        for fixed, sl in sorted(swept.items(), key=lambda kv: str(kv[0])):
            f = name("L__" + _title(("", *fixed[1:])))
            figure(sl, lambda p: p["length"], "L (case length)", f"{mod}: L sweep | {_title(fixed)}", mdir / f, card, backends)
            sections.append(f"### L sweep: {_title(fixed)}\n\n![]({f})\n\n{table(sl, lambda p: p['length'], 'L')}")
            n_l += 1
        for (key, fixed), (_, length, sl) in sorted(d_sweeps(pts).items(), key=lambda kv: str(kv[0])):
            f = name(f"D-{key}__L{length}__" + _title(("", *fixed[1:])))
            figure(sl, _d_label(key, sl), key, f"{mod}: {key} sweep at L{length} | {_title(fixed)}", mdir / f, card, backends)
            sections.append(f"### {key} sweep at L{length}: {_title(fixed)}\n\n![]({f})\n\n{table(sl, _d_label(key, sl), key)}")
            n_d += 1
        single = [p for p in pts if _fixed(p) not in swept]
        if single:
            sections.append(
                "### Single-length cases (no sweep)\n\n"
                + "\n".join(
                    f"- {_title(_fixed(p))} L{p['length']} {p['dtype']} {p['mode']} {p['backend']}: "
                    + (f"{p['median']:.4g} ms" if p["status"] == "ok" else str(p["status"]))
                    for p in sorted(single, key=lambda p: (str(_fixed(p)), p["length"], p["dtype"], p["mode"], p["backend"]))
                )
            )
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "README.md").write_text(
            f"# {mod} on {card}\n\nGenerated by `python -m miniworld_engine.measure.plot` from {what}; "
            "do not edit.\n\n" + "\n\n".join(sections) + "\n"
        )
        index.append(
            f"| [{mod}]({mod}/README.md) | {len({(p['dtype'], p['mode'], _fixed(p), p['length']) for p in pts})} | {n_l} | {n_d} |"
        )
        print(f"[plot] {mod}: {n_l} L sweeps, {n_d} D sweeps", flush=True)
    if not modules:
        stats = summary_stats(points, backends)
        summary_figure(stats, backends, out_dir / "summary.svg", card)
        per_model = []
        for model in model_summaries:
            rows = model_rows(model)
            mstats = summary_stats([p for p in points if p.get("row") in rows], backends)
            if not mstats:
                continue
            summary_figure(mstats, backends, out_dir / f"summary_{model}.png", card, scope=f"{model} only")
            per_model.append(
                f"\n## Summary: {model} only\n\nThe same over the requirement rows whose `source` names {model} "
                f"({len(rows)} rows).\n\n![](summary_{model}.png)\n\n" + summary_table(mstats, backends)
            )
        (out_dir / "README.md").write_text(
            f"# Benchmarks on {card}\n\nGenerated by `python -m miniworld_engine.measure.plot --gpu {card}` from {what} "
            f"(speed measurer v{_speed_version()}); nothing here is measured by the plotter. Timing: CUDA graph, L2 flushed before "
            "every call (docs/measure.md).\n\n## Summary\n\nPer class and mode: the geometric mean over its cases of the speedup vs "
            "PyTorch (PyTorch time / backend time), over the cases both ran (n = cases).\n\n![](summary.svg)\n\n"
            + summary_table(stats, backends)
            + "".join(per_model)
            + "\n## Classes\n\n| class | cases | L sweeps | D sweeps |\n|---|---|---|---|\n"
            + "\n".join(index)
            + "\n"
        )


def model_rows(model: str) -> set[int]:
    """The requirement rows (data-row numbers, 1 = first under the header) whose ``source`` names ``model`` (e.g. ``af3``)."""
    import csv

    with open(cfgmod.REQUIREMENTS, newline="") as fh:
        return {n for n, r in enumerate(csv.DictReader(fh), 1) if model in re.split(r"[\s;,]+", r.get("source") or "")}


# ------------------------------------------------------------------------------------------------------------------------ summary
def summary_stats(points: list[dict], backends) -> dict[tuple[str, str], dict[str, tuple[float, int]]]:
    """(module, mode) -> backend -> (geomean speedup vs pytorch over the cases both ran, n)."""
    by_case = collections.defaultdict(dict)
    for p in points:
        if _ok(p):
            by_case[(p["module"], p["mode"], _fixed(p), p["length"], p["dtype"])][p["backend"]] = p["median"]
    logs = collections.defaultdict(lambda: collections.defaultdict(list))
    for (mod, mode, *_), times in by_case.items():
        base = times.get("pytorch")
        for b in backends:
            if b != "pytorch" and base and b in times:
                logs[(mod, mode)][b].append(math.log(base / times[b]))
    return {k: {b: (math.exp(sum(v) / len(v)), len(v)) for b, v in d.items()} for k, d in logs.items()}


def summary_figure(stats: dict, backends, out: Path, card: str, scope: str = "every requirement row") -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    mpl.rcParams.update({"svg.fonttype": "none", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    bs = [b for b in backends if b != "pytorch"]
    mods = sorted({m for m, _ in stats})
    modes = [m for m in ("eval", "train") if any(k[1] == m for k in stats)]
    h = 0.8 / max(len(bs), 1)
    fig, axes = plt.subplots(1, len(modes), figsize=(7.5 * len(modes), 0.36 * len(bs) * len(mods) + 1.6), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes, strict=True):
        hi = 1.0
        for j, b in enumerate(bs):
            ys, xs = [], []
            for i, m in enumerate(mods):
                v = stats.get((m, mode), {}).get(b)
                if v:
                    ys.append(i + (j - (len(bs) - 1) / 2) * h)
                    xs.append(v[0])
                    hi = max(hi, v[0])
            ax.barh(ys, xs, height=h * 0.92, color=COLOR.get(b, "#444444"), label=LABEL.get(b, b), left=0)
            for y, x in zip(ys, xs, strict=True):
                ax.text(x * 1.04, y, f"{x:.2f}x", va="center", fontsize=6.5, color="#2A2F3A")
        ax.set_xscale("log")
        ax.axvline(1.0, color="#5A6473", linewidth=1.0, linestyle=(0, (4, 2)))
        ax.set_xlim(min(0.1, ax.get_xlim()[0]), hi * 3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.set_title(f"{mode}: speedup vs PyTorch (geomean over cases)", fontsize=10)
        ax.grid(axis="x", color="#E3E6EB", linewidth=0.6)
        ax.set_yticks(range(len(mods)), mods)
    axes[0][0].invert_yaxis()  # once: the y axis is shared (A at the top, the engine bar first in a group)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(bs), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(
        0.01,
        0.002,
        f"{card} | {scope} | CUDA graph, L2 flushed, median per case | a missing bar: the backend serves none of the class's "
        "cases in that mode",
        fontsize=7,
        color="#5A6473",
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.985))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format=out.suffix.lstrip(".") or "svg", dpi=110)
    plt.close(fig)


def summary_table(stats: dict, backends) -> str:
    bs = [b for b in backends if b != "pytorch"]
    modes = [m for m in ("eval", "train") if any(k[1] == m for k in stats)]
    head = "| class | " + " | ".join(f"{mode} {LABEL.get(b, b)}" for mode in modes for b in bs) + " |"
    rows = [head, "|---" * (1 + len(modes) * len(bs)) + "|"]
    for m in sorted({m for m, _ in stats}):
        cells = []
        for mode in modes:
            for b in bs:
                v = stats.get((m, mode), {}).get(b)
                cells.append(f"{v[0]:.2f}x (n={v[1]})" if v else "")
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def _speed_version() -> str:
    from miniworld_engine.measure.speed import SPEED_VERSION

    return SPEED_VERSION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--gpu", default=cfgmod.DEFAULT_GPU, help="the card (default %(default)s)")
    ap.add_argument("--module", nargs="*", default=None, help="only these classes (default: all)")
    ap.add_argument("--source", nargs="+", default=["tokens"], help="'tokens' (default) or speed record files / directories")
    ap.add_argument(
        "--backends",
        nargs="+",
        default=list(FIGURE_BACKENDS),
        help="backends drawn in the figures (default: %(default)s; the tables list all)",
    )
    ap.add_argument(
        "--model-summaries",
        nargs="*",
        default=["af3"],
        help="extra summaries over the rows one model uses (requirements source column; default: %(default)s)",
    )
    ap.add_argument("--out", default=str(REPO / "benchmarks"), help="output root (default <repo>/benchmarks); figures go to <out>/<gpu>/")
    a = ap.parse_args(argv)
    points, what = load(a.source, a.gpu)
    if not points:
        print(f"[plot] no speed results for {a.gpu} in {what}")
        return 1
    render(points, Path(a.out) / a.gpu, a.gpu, what, a.module, tuple(a.backends), tuple(a.model_summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

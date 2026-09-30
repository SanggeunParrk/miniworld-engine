"""Bar charts for the Measurements section of docs/gpus/<gpu>/<module>/<module>.md.

    python -m miniworld_engine.viz.measure_bars docs/gpus/b200/trimul/trimul.md [--length-d 128] [--dim-l 384]

Every table under ``## Measurements`` whose first column is ``(Length, <axis>)`` -- ``Dimension``, or e.g. ``MSA depth``
for the MSA modules -- gets two charts, drawn from the table itself so they cannot disagree with it: a length sweep at one
value of the second axis (``--length-d``, default 128) and a sweep of the second axis at one length (``--dim-l``, default
384). One bar per implementation column (columns that are all "—" are
skipped), latency in ms on a log axis, ours labelled with the table's × column. The figures are written to
``figures/<page>_<table>_{length,dimension}.{svg,png}`` next to the page, and one image line (ending in the
``<!-- measure_bars -->`` marker) is placed or replaced directly under each table. Needs matplotlib (the ``bench`` extra).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from miniworld_engine.viz import style

#: the second axis of a table's first column -> (tick prefix, chart / file name)
AXES = {"Dimension": ("D", "dimension"), "MSA depth": ("S", "msa_depth")}
#: table column -> style identity (colour / legend order)
IDENTITY = {"PyTorch compiled": "torch.compile", "cuEquivariance": "cuequivariance", "Anthropic": "anthropic",
            "ours": "miniworld"}
MARK = "<!-- measure_bars -->"
TIMES = "\u00d7"     # the speed-up column's header (multiplication sign)


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def tables(text: str):
    """(title, second axis, header, rows, end line index) for every (Length, <axis>) table under ## Measurements."""
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("## Measurements"))
    title = None
    i = start
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("## ") and i > start:
            break
        if ln.startswith("### "):
            title = ln[4:].strip()
        m = re.match(r"\| \(Length, ([^)]+)\) \|", ln)
        if m and m.group(1) in AXES and title:
            header = [c.strip() for c in ln.strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].startswith("| ("):
                cells = [c.strip() for c in lines[j].strip("|").split("|")]
                length, dim = map(int, re.findall(r"\d+", cells[0]))
                rows.append((length, dim, cells[1:]))
                j += 1
            yield title, m.group(1), header[1:], rows, j - 1
            i = j
            continue
        i += 1


def value(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def chart(title, header, rows, axis, fixed, out: Path, second: str = "Dimension") -> bool:
    import matplotlib.pyplot as plt

    impls = [h for h in header if h != TIMES]
    xcol = header.index(TIMES) if TIMES in header else None
    sel = [(r[0] if axis == "length" else r[1], r[2]) for r in rows if (r[1] if axis == "length" else r[0]) == fixed]
    if len(sel) < 2:
        return False
    sel.sort()
    impls = [h for h in impls if any(value(c[header.index(h)]) is not None for _, c in sel)]
    style.apply_theme()
    fig, ax = plt.subplots(figsize=(1.1 + 0.95 * len(sel), 3.4))
    width = 0.8 / len(impls)
    for k, h in enumerate(impls):
        ident = IDENTITY.get(h, h)
        xs = [i + (k - (len(impls) - 1) / 2) * width for i in range(len(sel))]
        ys = [value(c[header.index(h)]) for _, c in sel]
        ax.bar([x for x, y in zip(xs, ys, strict=True) if y is not None], [y for y in ys if y is not None], width * 0.92,
               color=style.color_for(ident), label=h, zorder=3)
        if h == "ours" and xcol is not None:
            for x, y, (_, c) in zip(xs, ys, sel, strict=True):
                if y is not None and value(c[xcol]) is not None:
                    ax.text(x, y * 1.08, f"{c[xcol]}{TIMES}", ha="center", va="bottom", fontsize=8, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sel)))
    pre, name = AXES[second]
    ax.set_xticklabels([f"L{v}" if axis == "length" else f"{pre}{v}" for v, _ in sel])
    ax.set_ylabel("latency (ms, log)")
    ax.set_title(f"{title} · {'length sweep, ' + pre if axis == 'length' else name.replace('_', ' ') + ' sweep, L'}{fixed}", fontsize=10.5)
    ax.legend(ncol=len(impls), fontsize=8.5, loc="upper left")
    ax.margins(y=0.15)
    style.save_figure(fig, out, formats=("svg", "png"))
    plt.close(fig)
    return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("page", type=Path)
    ap.add_argument("--length-d", type=int, default=128)
    ap.add_argument("--dim-l", type=int, default=384)
    args = ap.parse_args(argv)
    page: Path = args.page
    figdir = page.parent / "figures"
    figdir.mkdir(exist_ok=True)
    text = page.read_text()
    inserts = []
    for title, second, header, rows, end in tables(text):
        base = f"{page.stem}_{slug(title)}"
        pre, name = AXES[second]
        links = []
        for axis, fixed in (("length", args.length_d), (name, args.dim_l)):
            if chart(title, header, rows, axis, fixed, figdir / f"{base}_{axis}", second):
                what = f"length sweep at {pre}{fixed}" if axis == "length" else f"{name.replace('_', ' ')} sweep at L{fixed}"
                links.append(f"![{title}, {what}](figures/{base}_{axis}.png)")
        if links:
            inserts.append((end, " ".join(links) + " " + MARK))   # the marker last: a line that STARTS with <!-- is raw HTML
    lines = text.split("\n")
    for end, line in sorted(inserts, reverse=True):
        # replace an existing chart line (blank, chart) under the table, else insert one
        if end + 2 < len(lines) and MARK in lines[end + 2]:
            lines[end + 2] = line
        else:
            lines[end + 1:end + 1] = ["", line]
    page.write_text("\n".join(lines))
    print(f"{len(inserts)} tables charted in {page}")


if __name__ == "__main__":
    main()

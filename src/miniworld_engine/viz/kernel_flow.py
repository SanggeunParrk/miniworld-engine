"""Kernel-flow figures for docs/gpus/<gpu>/<module>/<module>.md: one box per kernel, left to right, HBM underneath.

    python -m miniworld_engine.viz.kernel_flow docs/gpus/h100/trimul/figures/trimul_bidir.json

A spec is JSON: ``{"figures": {"<name>": {"title": ..., "panels": [...]}}}``; each figure is
written next to the spec as ``<spec stem>_<name>.svg``. Panels stack top to bottom (training:
forward, then backward), each with its own HBM box. A panel is ``{"label", "kernels", "hbm",
"hbm_title"}``; a kernel is ``{"id", "name", "impl", "code": [lines], "reads", "writes"}``;
``hbm`` lists ``{"name", "shape", "size"}``. In any text, ``a_{b}`` renders ``b`` as a subscript.
Standard library only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

MIN_BOX_W, GAP = 230, 46     # boxes widen to fit their longest code line
HEAD_H, LINE_H, PAD = 46, 15, 9
LEFT, TITLE_H, LABEL_H = 30, 44, 26
IO_H = 90                    # band between the boxes and HBM for the read/write arrows
MONO_W, CHIP_W, CHIP_H = 6.4, 250, 40
PANEL_GAP = 36
IMPL_FILL = {"CUDA": "#d9f2e3", "cuBLAS": "#dbe7fb", "Triton": "#fde8cf", "PyTorch": "#eceef1"}
INK, MUTED, WRITE, READ = "#1f2328", "#57606a", "#b3261e", "#1a5fb4"
SUB = re.compile(r"_\{([^}]*)\}")


def _plain(s: str) -> str:
    """Text without subscript markup, for width estimates."""
    return SUB.sub(r"\1", s)


def _rich(s: str) -> str:
    """Escape ``s`` and turn ``_{...}`` into subscript tspans."""
    out, pos = [], 0
    for m in SUB.finditer(s):
        out.append(escape(s[pos:m.start()]))
        out.append(f'<tspan baseline-shift="sub" font-size="75%">{escape(m.group(1))}</tspan>')
        pos = m.end()
    out.append(escape(s[pos:]))
    return "".join(out)


def _text(x, y, s, size=12, weight="normal", family="Helvetica, Arial, sans-serif",
          fill=INK, anchor="start"):
    keep = ' xml:space="preserve"' if "mono" in family else ""  # code indentation
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{keep}>{_rich(s)}</text>')


def _arrow(x1, y1, x2, y2, color, dash=False):
    d = ' stroke-dasharray="5,4"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
            f'stroke-width="1.6"{d} marker-end="url(#head-{color[1:]})"/>')


def _impl_fill(impl: str) -> str:
    return next((c for k, c in IMPL_FILL.items() if impl.startswith(k)), "#ffffff")


def _panel(panel: dict, top: float, box_w: float, width: float) -> tuple[list[str], float]:
    """Draw one panel starting at ``top``; return its elements and its bottom edge."""
    kernels = panel["kernels"]
    code_lines = max(len(k.get("code", [])) for k in kernels)
    box_h = HEAD_H + PAD * 2 + code_lines * LINE_H
    out = [_text(LEFT, top + 16, panel["label"], size=13, weight="bold", fill=MUTED)]
    y0 = top + LABEL_H
    xs = [LEFT + i * (box_w + GAP) for i in range(len(kernels))]
    labels = max(len(k.get("writes", [])) + len(k.get("reads", [])) for k in kernels)
    hbm_y = y0 + box_h + max(IO_H, 34 + labels * 13)
    for i, k in enumerate(kernels):
        bx = xs[i]
        out.append(f'<rect x="{bx}" y="{y0}" width="{box_w}" height="{box_h}" rx="6" fill="#ffffff" '
                   f'stroke="{INK}" stroke-width="1.2"/>')
        out.append(f'<path d="M{bx + 6},{y0} h{box_w - 12} a6,6 0 0 1 6,6 v{HEAD_H - 6} h-{box_w} '
                   f'v-{HEAD_H - 6} a6,6 0 0 1 6,-6 z" fill="{_impl_fill(k["impl"])}"/>')
        out.append(f'<line x1="{bx}" y1="{y0 + HEAD_H}" x2="{bx + box_w}" y2="{y0 + HEAD_H}" '
                   f'stroke="{INK}" stroke-width="1"/>')
        out.append(_text(bx + 10, y0 + 19, f'{k["id"]} · {k["name"]}', size=13, weight="bold"))
        out.append(_text(bx + 10, y0 + 37, k["impl"], size=11, fill=MUTED))
        out.extend(_text(bx + 10, y0 + HEAD_H + PAD + 11 + j * LINE_H, line, size=10.5,
                         family="Menlo, Consolas, monospace")
                   for j, line in enumerate(k.get("code", [])))
        if i + 1 < len(kernels):
            y = y0 + HEAD_H / 2
            out.append(_arrow(bx + box_w, y, xs[i + 1] - 3, y, INK))
        by = y0 + box_h
        # reads come up on the left (inputs), writes go down on the right (outputs)
        if k.get("reads"):
            rx = bx + box_w * 0.12
            out.append(_arrow(rx, hbm_y, rx, by + 2, READ, dash=True))
            out.extend(_text(rx + 6, by + 18 + j * 13, name, size=10.5, fill=READ)
                       for j, name in enumerate(k["reads"]))
        if k.get("writes"):
            wx = bx + box_w * 0.88
            out.append(_arrow(wx, by, wx, hbm_y - 2, WRITE))
            below = len(k.get("reads", []))  # writes go under the reads, right-aligned to their arrow
            out.extend(_text(wx - 6, by + 18 + (below + j) * 13, name, size=10.5, fill=WRITE, anchor="end")
                       for j, name in enumerate(k["writes"]))
    per_row = max(1, int((width - 2 * LEFT - 12) // (CHIP_W + 10)))
    rows = (len(panel["hbm"]) + per_row - 1) // per_row
    hbm_h = 40 + rows * (CHIP_H + 8)
    out.append(f'<rect x="{LEFT}" y="{hbm_y}" width="{width - 2 * LEFT:.0f}" height="{hbm_h}" rx="8" '
               f'fill="#f6f8fa" stroke="{INK}" stroke-width="1.4"/>')
    out.append(_text(LEFT + 12, hbm_y + 22, panel.get("hbm_title", "HBM"), size=13, weight="bold"))
    for n, t in enumerate(panel["hbm"]):
        cx = LEFT + 12 + (n % per_row) * (CHIP_W + 10)
        cy = hbm_y + 32 + (n // per_row) * (CHIP_H + 8)
        out.append(f'<rect x="{cx}" y="{cy}" width="{CHIP_W}" height="{CHIP_H}" rx="4" fill="#ffffff" '
                   f'stroke="{MUTED}" stroke-width="0.8"/>')
        out.append(_text(cx + 8, cy + 16, t["name"], size=11, weight="bold"))
        out.append(_text(cx + 8, cy + 32, f'{t["shape"]}  {t.get("size", "")}'.strip(), size=10, fill=MUTED))
    return out, hbm_y + hbm_h


def render(fig: dict) -> str:
    panels = fig["panels"]
    longest = max((len(_plain(line)) for p in panels for k in p["kernels"] for line in k.get("code", [])),
                  default=0)
    box_w = max(MIN_BOX_W, int(longest * MONO_W) + 24)
    most = max(len(p["kernels"]) for p in panels)
    width = 2 * LEFT + most * box_w + (most - 1) * GAP
    body, y = [], TITLE_H
    for p in panels:
        els, bottom = _panel(p, y, box_w, width)
        body += els
        y = bottom + PANEL_GAP
    height = y - PANEL_GAP + 24
    head = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
             f'viewBox="0 0 {width:.0f} {height:.0f}">'), "<defs>"]
    head.extend(f'<marker id="head-{c[1:]}" markerWidth="9" markerHeight="7" refX="8" refY="3.5" '
                f'orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{c}"/></marker>'
                for c in (INK, WRITE, READ))
    head += ["</defs>", f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
             _text(LEFT, 28, fig["title"], size=16, weight="bold"),
             _text(width - LEFT, 24, "red: written to HBM", size=11, fill=WRITE, anchor="end"),
             _text(width - LEFT, 38, "blue, dashed: read from HBM", size=11, fill=READ, anchor="end")]
    return "\n".join([*head, *body, "</svg>"]) + "\n"


def main(argv: list[str] | None = None) -> int:
    for spec_path in map(Path, argv if argv is not None else sys.argv[1:]):
        spec = json.loads(spec_path.read_text())
        for name, fig in spec["figures"].items():
            target = spec_path.with_name(f"{spec_path.stem}_{name}.svg")
            target.write_text(render(fig))
            print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

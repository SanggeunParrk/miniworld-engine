"""Shared visualization conventions for miniworld-kernels benchmarks.

One palette, one theme, one set of display labels — used by *every* plotting
path in the repo (``scripts/plot_bench.py`` and ``scripts/bench.py``) so that a
backend always gets the same colour, label and ordering across every figure.
This is what makes the benchmark figures publication-coherent: a reader (or a
paper reviewer) sees "ours" in the same hot colour in every plot, and the
NVIDIA baselines in the same greens, regardless of which figure they look at.

See ``style.py`` for the actual definitions.
"""

from miniworld_kernels.viz.style import (
    FORMATS,
    PALETTE,
    apply_theme,
    canonical,
    color_for,
    label_for,
    save_figure,
    sort_backends,
    style_for,
)

__all__ = [
    "FORMATS",
    "PALETTE",
    "apply_theme",
    "canonical",
    "color_for",
    "label_for",
    "save_figure",
    "sort_backends",
    "style_for",
]

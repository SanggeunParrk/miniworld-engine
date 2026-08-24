"""Canonical benchmark plotting style: palette, labels, ordering, theme.

The single source of truth for how benchmark figures look in this repo. Both
plotting paths import from here:

- ``benchmarks/runners/plot_bench.py`` — matplotlib grouped bars from a parsed ``.out``.
- ``benchmarks/runners/bench.py`` — Triton ``perf_report`` line plots (wants
  ``(colour, linestyle)`` tuples via :func:`style_for`).

Design philosophy (so figures read as one coherent set, paper-ready):

- **MiniWorld / cute family → dark gold.** The thing we built is always the same
  blackened-gold series so it is immediately recognisable.
- **NVIDIA family (cuequivariance / dtv1 / Transformer Engine) → greens & teal.**
  NVIDIA's brand green, kept together so "the NVIDIA kernels" are visually a group.
- **plain baselines (pytorch / torch.compile / triton) → cool greys & blue.**
  The naive baseline recedes; it is the reference, not the story.

Backend names are wildly inconsistent across the repo's logs and reports
(``cuequivariance`` / ``cuequiv``, ``cute`` / ``cute-fused`` / ``ours v4`` /
``v2``, ``nvidia dtv1`` / ``dt-v1`` …). :func:`canonical` normalises any of
these to one identity, so the same *thing* always gets the same colour even when
a different log spells it differently. Unknown names fall back to a
*deterministic* extra colour (stable across figures — never index-dependent).

Pure-Python parts (palette / labels / ordering) import no plotting libraries, so
they are safe to import anywhere. :func:`apply_theme` / :func:`save_figure`
import matplotlib lazily.
"""

from __future__ import annotations

import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# Canonical identities
# --------------------------------------------------------------------------- #
# Every backend label seen in a log/report maps to one of these identities.
# Order here is the canonical legend / bar ordering (baselines first, NVIDIA
# next, ours last so it sits rightmost / on top).
ORDER: list[str] = [
    "pytorch",
    "torch.compile",
    "triton",
    "old-triton",
    "triton-atomic",
    "triton-atomic-compile",
    "triton-partial",
    "triton-partial-compile",
    "te",
    "cuda",
    "cuequivariance",
    "dtv1",
    "layernorm-dispatch",
    "layernorm-dispatch-compile",
    "miniworld",
    "miniworld-alt",
    "miniworld-alt2",
]

# Pretty labels for legends/titles. Keep short — these go on crowded plots.
DISPLAY: dict[str, str] = {
    "pytorch": "PyTorch",
    "torch.compile": "torch.compile",
    "triton": "Triton",
    "old-triton": "old Triton",
    "triton-atomic": "Triton atomic",
    "triton-atomic-compile": "Triton atomic compile",
    "triton-partial": "Triton partial",
    "triton-partial-compile": "Triton partial compile",
    "te": "TransformerEngine",
    "cuda": "CUDA",
    "cuequivariance": "cuEquivariance",
    "dtv1": "NVIDIA dtv1",
    "layernorm-dispatch": "Auto dispatch",
    "layernorm-dispatch-compile": "Auto dispatch compile",
    "miniworld": "MiniWorld",
    "miniworld-alt": "MiniWorld (alt)",
    "miniworld-alt2": "MiniWorld (alt₂)",
}

# Canonical colour per identity. Hex strings (no matplotlib needed to read them).
PALETTE: dict[str, str] = {
    # baselines — cool, recede
    "pytorch": "#B9C0CC",        # light grey — naive reference
    "torch.compile": "#6E7B91",  # slate
    "triton": "#2E6FDB",         # blue
    "old-triton": "#7A86A1",     # muted blue-grey for Team-GM legacy Triton
    "triton-atomic": "#5B8FF9",  # lighter blue variant for the atomic path
    "triton-atomic-compile": "#1F4AA8",  # dark blue compiled atomic path
    "triton-partial": "#F29A38",  # warm orange for the partial-reduction variant
    "triton-partial-compile": "#C66A12",  # darker orange compiled partial path
    # NVIDIA family — greens / teal
    "te": "#3F6B1B",             # dark green (Transformer Engine)
    "cuequivariance": "#11A6A0",  # teal (cuEquivariance lib)
    "dtv1": "#76B900",           # NVIDIA brand green
    "cuda": "#9CCC3C",           # light green (generic CUDA path)
    "layernorm-dispatch": "#E8412B",  # hero red-orange for the shipped dispatch path
    "layernorm-dispatch-compile": "#A8281A",  # darker compiled dispatch path
    # MiniWorld (ours) — gold, fixed across every figure.
    "miniworld": "#D4AF37",
    "miniworld-alt": "#F2C94C",
    "miniworld-alt2": "#8A6A14",
}

# Linestyle per identity for line plots (perf_report). ours = solid & prominent;
# baselines = dashed/dotted so they read as "reference".
#: A named style (``"-"``) or an explicit ``(offset, (on, off))`` dash pattern -- both are
#: what matplotlib's ``linestyle=`` takes.
LINESTYLE: dict[str, str | tuple[int, tuple[int, int]]] = {
    "pytorch": (0, (1, 1)),       # dotted
    "torch.compile": (0, (4, 2)),  # dashed
    "triton": "-",
    "old-triton": (0, (4, 2)),
    "triton-atomic": "-",
    "triton-atomic-compile": (0, (4, 2)),
    "triton-partial": "-",
    "triton-partial-compile": (0, (4, 2)),
    "te": (0, (4, 2)),
    "cuequivariance": "-",
    "dtv1": "-",
    "cuda": "-",
    "layernorm-dispatch": "-",
    "layernorm-dispatch-compile": (0, (4, 2)),
    "miniworld": "-",
    "miniworld-alt": "-",
    "miniworld-alt2": "-",
}

# Deterministic fallback colours for backends we have not catalogued. Picking by
# a stable hash of the name (not its position in a list) guarantees the same
# unknown label gets the same colour in every figure it appears in.
_EXTRAS: list[str] = [
    "#8E5BD0", "#D14F8F", "#5BB0D1", "#C9A227", "#7A9E3A", "#C4633A",
]

# Alias table: normalised raw label -> canonical identity. Lookups are done on a
# lowercased, whitespace/underscore/hyphen-collapsed key (see _norm).
_ALIASES: dict[str, str] = {
    # pytorch
    "pytorch": "pytorch", "torch": "pytorch", "pt": "pytorch", "eager": "pytorch",
    "pytorchnaive": "pytorch", "naive": "pytorch",
    # torch.compile
    "torchcompile": "torch.compile", "compile": "torch.compile",
    "compiled": "torch.compile", "inductor": "torch.compile",
    # triton
    "triton": "triton",
    "oldtriton": "old-triton",
    "teamgmtriton": "old-triton",
    "teamgm": "old-triton",
    "tritonatomic": "triton-atomic",
    "tritonatomiccompile": "triton-atomic-compile",
    "tritonpartial": "triton-partial",
    "tritonpartialcompile": "triton-partial-compile",
    "tritonln": "layernorm-dispatch",
    "tritonkernelln": "layernorm-dispatch",
    "tritondispatchln": "layernorm-dispatch",
    "tritonlnkernel": "layernorm-dispatch",
    "tritonlndispatch": "layernorm-dispatch",
    "tritonpytorchln": "triton",
    # transformer engine
    "te": "te", "transformerengine": "te", "transformer engine": "te",
    # cuequivariance
    "cuequivariance": "cuequivariance", "cuequiv": "cuequivariance",
    "cueq": "cuequivariance", "cue": "cuequivariance",
    # nvidia dtv1
    "dtv1": "dtv1", "nvidiadtv1": "dtv1", "nvidia": "dtv1", "dt": "dtv1",
    # generic cuda
    "cuda": "cuda",
    # miniworld family (canonical = miniworld, displayed "ours"; variants -alt / -alt2).
    # Every authorship spelling we've ever emitted resolves here, so old .out logs and
    # the one-off bench scripts that print "ours"/"cute"/"v4"/... need no changes.
    "miniworld": "miniworld", "mwk": "miniworld", "miniworldkernels": "miniworld",
    "ours": "miniworld", "oursv4": "miniworld", "cute": "miniworld", "cutefused": "miniworld",
    "cutefwd": "miniworld", "v4": "miniworld", "layernormkernel": "miniworld",
    "layernormdispatch": "layernorm-dispatch", "autodispatch": "layernorm-dispatch",
    "layernormdispatchcompile": "layernorm-dispatch-compile",
    "oursv5": "miniworld-alt", "cutetrain": "miniworld-alt", "v5": "miniworld-alt",
    "partialbuffer": "triton-partial", "partialreduction": "triton-partial",
    "v2": "miniworld-alt2", "oursv2": "miniworld-alt2", "v3": "miniworld-alt2",
}

# --------------------------------------------------------------------------- #
# Kernel-level function-benchmark variants (benchmarks/kernels/<op>/).
# --------------------------------------------------------------------------- #
# Each developed OR deprecated implementation of an op is its own CSV row. Register a stable
# identity + colour for every one so sibling variants that share a substring the canonical
# heuristics collapse (`cute`/`miniworld`) never overwrite each other in a single figure.
# (identity, display, colour). Triton kernels → blues; cute/quack → golds; TE → green;
# deprecated/negative-result variants → muted grey/orange.
_KERNEL_VARIANTS: dict[str, tuple[str, str, str]] = {
    # dual_gemm_epil (front) + gemm_gate/tm2 + gemm_epil (LN+linear)
    "trimulfronttriton": ("trimul-front-triton", "Triton front", "#2E6FDB"),
    "tritontm1": ("triton-tm1", "Triton tm1", "#5B8FF9"),
    "tritongated": ("triton-gated", "Triton gated (dep)", "#7FB0FF"),
    "trimulinprojcute": ("trimul-inproj-cute", "cute front", "#D4AF37"),
    "tm1cute": ("tm1-cute", "cute tm1", "#F2C94C"),
    "trimulfrontsm100": ("trimul-front-sm100", "cute SM100 (dep)", "#8A6A14"),
    "layernormlineartriton": ("layernorm-linear-triton", "Triton LN+linear", "#2E6FDB"),
    "layernormlinearcute": ("layernorm-linear-cute", "cute LN+linear M1", "#D4AF37"),
    "layernormlinearcutefused": ("layernorm-linear-cute-fused", "cute LN+linear M2", "#F2C94C"),
    "layernormlinearte": ("layernorm-linear-te", "TE-style", "#3F6B1B"),
    "tm2cute": ("tm2-cute", "cute tm2", "#D4AF37"),
    "tritontm2": ("triton-tm2", "Triton tm2", "#2E6FDB"),
    # transition_b2b
    "tritontransitionfused": ("triton-transition-fused", "Triton transition", "#2E6FDB"),
    "cutetransitionfused": ("cute-transition-fused", "cute transition", "#D4AF37"),
    "transitionb2bktiled": ("transition-b2b-ktiled", "Triton k-tiled (unver)", "#7FB0FF"),
    # layernorm (+ bwd)
    "tritonlayernorm": ("triton-layernorm", "Triton LN", "#2E6FDB"),
    "quackcute": ("quack-cute", "quack cute", "#D4AF37"),
    "tritonlayernormlowreg": ("triton-layernorm-lowreg", "Triton LN low-reg (dep)", "#7A86A1"),
    "tritonpersistent": ("triton-persistent", "Triton persistent", "#11A6A0"),
    # adaln (+ bwd)
    "adalninference": ("adaln-inference", "Triton adaLN inf", "#2E6FDB"),
    "tritonadaln": ("triton-adaln", "Triton adaLN", "#5B8FF9"),
    "adalnfused3": ("adaln-fused3", "Triton adaLN fused3", "#7FB0FF"),
    "adalntrain": ("adaln-train", "Triton adaLN train", "#5B8FF9"),
    # attentions
    "tritontriattn": ("triton-tri-attn", "Triton tri-attn", "#2E6FDB"),
    "tritontriattnminiworld": ("triton-tri-attn-miniworld", "Triton tri-attn (old)", "#7A86A1"),
    "tritontriattnperf": ("triton-tri-attn-perf", "Triton tri-attn (perf)", "#9AA7BF"),
    "tritonbiasattn": ("triton-bias-attn", "Triton bias-attn", "#2E6FDB"),
    "biasonlyfused": ("bias-only-fused", "Triton bias fused (dep)", "#C66A12"),
    "tritonaugattn": ("triton-aug-attn", "Triton aug-attn", "#2E6FDB"),
    "augattncomputeefficient": ("aug-attn-compute-efficient", "Triton aug (comp-eff)", "#7A86A1"),
    # ln_mask, cond_transition, misc bwd
    "fusedlnmask": ("fused-ln-mask", "fused LN+mask", "#D4AF37"),
    "tritoncondtransition": ("triton-cond-transition", "Triton cond-transition", "#2E6FDB"),
    "gateelembwd": ("gate-elem-bwd", "Triton gate bwd", "#2E6FDB"),
    "frontbwdfused": ("front-bwd-fused", "Triton front bwd", "#2E6FDB"),
}
for _k, (_ident, _disp, _col) in _KERNEL_VARIANTS.items():
    _ALIASES.setdefault(_k, _ident)
    DISPLAY.setdefault(_ident, _disp)
    PALETTE.setdefault(_ident, _col)
    LINESTYLE.setdefault(_ident, "-")
    if _ident not in ORDER:
        ORDER.append(_ident)

_NORM_RE = re.compile(r"[\s_\-./]+")


def _norm(name: str) -> str:
    """Lowercase and strip separators so 'cute-fused' == 'cute_fused' == 'cute fused'."""
    return _NORM_RE.sub("", name.strip().lower())


def canonical(name: str) -> str:
    """Map any backend label to its canonical identity.

    Falls through aliases, then substring heuristics, then returns the
    normalised name itself (so unknown backends are still handled consistently).
    """
    key = _norm(name)
    if key in _ALIASES:
        return _ALIASES[key]
    # substring heuristics for compound labels ("ours-trimul-fwd", "cute-v2", …)
    if "cuequiv" in key or "cueq" in key:
        return "cuequivariance"
    if "dtv1" in key:
        return "dtv1"
    if "ours" in key or "cute" in key or "miniworld" in key:
        return "miniworld"
    if "compile" in key:
        return "torch.compile"
    return key


def _fallback_color(key: str) -> str:
    """Stable colour for an uncatalogued identity (hash by content, not order)."""
    h = sum(ord(c) for c in key)
    return _EXTRAS[h % len(_EXTRAS)]


def color_for(name: str) -> str:
    """Canonical hex colour for a backend label (any spelling)."""
    ident = canonical(name)
    return PALETTE.get(ident, _fallback_color(ident))


def label_for(name: str) -> str:
    """Pretty display label for a backend (falls back to the raw name)."""
    ident = canonical(name)
    return DISPLAY.get(ident, name)


def style_for(name: str):  # returns matplotlib (color, linestyle)
    """``(colour, linestyle)`` tuple for line plots (Triton ``perf_report``)."""
    ident = canonical(name)
    return color_for(name), LINESTYLE.get(ident, "-")


def sort_backends(names: list[str]) -> list[str]:
    """Sort backend labels into canonical order (baselines → NVIDIA → ours).

    Stable for unknowns: anything not in :data:`ORDER` is appended in the
    incoming order, after the catalogued ones.
    """
    rank = {ident: i for i, ident in enumerate(ORDER)}
    return sorted(names, key=lambda n: (rank.get(canonical(n), len(ORDER)), n))


# --------------------------------------------------------------------------- #
# Matplotlib theme + vector-friendly saving (imported lazily)
# --------------------------------------------------------------------------- #
# Default output format. SVG is the source artifact; PNG/PDF can be derived from
# it when needed, so benchmark rendering keeps only the canonical vector file.
FORMATS: tuple[str, ...] = ("svg",)


def apply_theme() -> None:
    """Install the repo's publication-grade matplotlib rcParams (idempotent)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        # typography — DejaVu Sans ships with matplotlib (no font hunting)
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        # clean axes: drop the top/right box, light y-grid behind the data
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#D7DCE3",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "#5A6473",
        "axes.linewidth": 1.0,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#D7DCE3",
        # vector text stays text (selectable / searchable in the SVG)
        "svg.fonttype": "none",
    })


def save_figure(fig, out_path: Path, formats: tuple[str, ...] = FORMATS) -> list[Path]:
    """Save ``fig`` to ``out_path`` in every requested format.

    ``out_path``'s suffix is ignored; one file per format is written next to it
    (``foo.svg`` by default). Returns the paths written. SVG stays editable and
    can be converted to PNG/PDF outside the benchmark pipeline when required.
    """
    out_path = Path(out_path)
    written = []
    for fmt in formats:
        target = out_path.with_suffix(f".{fmt}")
        fig.savefig(target, format=fmt)
        written.append(target)
    return written

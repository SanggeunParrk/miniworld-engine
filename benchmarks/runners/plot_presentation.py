"""Presentation figures from the bench.py CSVs of the sweep (jobs named <pass>_<module>_<mode>_<axis>, in the CSV file names).
main_inference / main_training: every module at L = 384 (speedup over PyTorch, per implementation);
sweep_<module>: latency vs L and vs D (or the layout list) for inference and training. Median over passes."""
import csv, glob, math, os, re, sys
from collections import defaultdict
from statistics import median
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOTS = sys.argv[1].split(",")
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

COL = {"pytorch": "#8A8F98", "cuequivariance": "#E8833A", "anthropic": "#2A9D8F", "miniworld": "#1F4E9E"}
LAB = {"pytorch": "PyTorch", "cuequivariance": "cuEquivariance", "anthropic": "Anthropic", "miniworld": "Ours"}
ORDER = ["pytorch", "cuequivariance", "anthropic", "miniworld"]
MK = {"pytorch": "o", "cuequivariance": "s", "anthropic": "^", "miniworld": "D"}
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 14, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E3E5E8", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 220, "savefig.bbox": "tight",
    "axes.edgecolor": "#444", "svg.fonttype": "none",
})

# ---- read ----
DATA = defaultdict(list)          # (job, impl, x_seq, x_d) -> [values over passes]
for f in [g for r in ROOTS for g in glob.glob(os.path.join(r, "benchmarks", "*", "*", "artifacts", "*", "*.csv"))]:
    m = re.search(r"__([a-z]\d+)_(.+)\.csv$", f)
    if not m:
        continue
    pas, job = m.group(1), m.group(2)
    for r in csv.DictReader(open(f, newline="")):
        try:
            v = float(r["value"])
        except (KeyError, ValueError):
            continue
        if not math.isfinite(v) or r.get("status", "ok") != "ok":
            continue
        DATA[(job, r["implementation"], int(float(r["seq_len"])), int(float(r.get("d_pair") or 0)))].append(v)
MED = {k: median(v) for k, v in DATA.items()}
NPASS = max((len(v) for v in DATA.values()), default=0)
print(f"{len(MED)} points, up to {NPASS} passes")


def series(jobs, impl, xkey):
    """{x: ms} for the implementation over the given job names (xkey: 'seq' | 'd')"""
    out = {}
    for (job, im, s, d), v in MED.items():
        if job in jobs and im == impl:
            out[s if xkey == "seq" else d] = v
    return dict(sorted(out.items()))


def layout_series(prefix, mode, layouts, impl, extra=()):
    out = {}
    for lay in layouts:
        names = {f"{prefix}_{mode}_{lay}"} | {f"{prefix}_{mode}_{lay}_anth"}
        for (job, im, s, d), v in MED.items():
            if job in names and im == impl and s == 384:
                out[lay] = v
    return out


M3 = lambda k: (k + "_inf_L", k + "_tra_L")
MODS = [
    dict(key="trimul1", title="TriMul (outgoing)", D=[64, 128, 256, 384, 512], dlabel="d_pair"),
    dict(key="trimul2", title="TriMul (bidirectional)", D=[64, 128, 256, 384, 512], dlabel="d_pair"),
    dict(key="triattn", title="Triangle attention", D=[64, 128, 256, 384, 512], dlabel="d_pair"),
    dict(key="trans", title="Transition (pair)", D=[64, 128, 256, 384, 512], dlabel="width D"),
    dict(key="tdit", title="Token DiT", layouts=["16x48", "24x32", "12x64", "16x64"], dlabel="heads × head width", anth_extra=True),
    dict(key="bodit", title="Bias-only token DiT", layouts=["16x48", "24x32", "12x64", "16x64"], dlabel="heads × head width"),
    dict(key="apb", title="Attention pair bias", layouts=["8x384", "12x384", "16x384", "24x384", "16x512"], dlabel="heads × d_single"),
    dict(key="opm", title="Outer product mean"),
    dict(key="pwa", title="MSA pair-weighted avg."),
    dict(key="ldit", title="AF3 atom DiT (32×128, 3072 atoms)", short="AF3 atom DiT\n(32×128)", atom=True),
    dict(key="swadit3", title="SWA atom DiT (|i−j| ≤ 64, 3072 atoms)", short="SWA atom DiT\n(window 129)", atom=True),
]
ATOM = []


def jobs_L(mod, mode):
    k, md = mod["key"], mode[:3]
    names = {f"{k}_{md}_L", f"{k}_{md}_L_anth", f"{k}_{md}", f"{k}_{md}_anth"}
    return names


def at384(mod, mode, impl):
    s = series(jobs_L(mod, mode), impl, "seq")
    return s.get(384)


# ---- main bars (L = 384) ----
def main_fig(mode):
    slots = [i for i in ORDER if not (mode == "training" and i == "anthropic")]
    fig, ax = plt.subplots(figsize=(18, 6.4))
    n = len(MODS); w = 0.8 / len(slots)
    ours_ms = {}
    for gi, mod in enumerate(MODS):
        base = at384(mod, mode, "pytorch")
        if base is None:
            continue
        for si, impl in enumerate(slots):
            t = at384(mod, mode, impl)
            if t is None:
                continue
            x = gi + (si - (len(slots) - 1) / 2) * w
            sp = base / t
            ax.bar(x, sp, w * 0.92, color=COL[impl], label=LAB[impl] if gi == 0 or True else None, zorder=3)
            if impl == "miniworld":
                ax.text(x, sp * 1.02, f"{sp:.1f}×", ha="center", va="bottom", fontsize=12, fontweight="bold", color=COL[impl])
                ours_ms[gi] = t
    ax.axhline(1.0, color="#444", lw=1.1, zorder=4)
    ax.set_xticks(range(n))
    import textwrap
    ax.set_xticklabels([m.get("short") or "\n".join(textwrap.wrap(m["title"], 12, break_long_words=False)) for m in MODS], fontsize=11.5)
    ax.set_ylabel("Speedup over PyTorch (higher is better)")
    ax.set_title(f"B200 · {mode} · L = 384 tokens (atom blocks: 3072 atoms) · bf16" + ("" if mode == "inference" else " · CUDA graph"), loc="left")
    h, l = ax.get_legend_handles_labels()
    uniq = {}
    for a, b in zip(h, l):
        uniq.setdefault(b, a)
    ax.legend([uniq[LAB[i]] for i in slots if LAB[i] in uniq], [LAB[i] for i in slots if LAB[i] in uniq], ncol=len(slots), loc="upper left", frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    fig.savefig(os.path.join(OUT, f"main_{mode}.png")); fig.savefig(os.path.join(OUT, f"main_{mode}.svg")); plt.close(fig)


# ---- sweeps ----
def best_other(vals_by_impl, x):
    others = [v[x] for i, v in vals_by_impl.items() if i != "miniworld" and x in v]
    return min(others) if others else None


def draw(ax, data, xs, xlabel, categorical=False, annotate=True):
    pos = {x: i for i, x in enumerate(xs)} if categorical else {x: x for x in xs}
    for impl in ORDER:
        s = data.get(impl, {})
        pts = [(pos[x], s[x]) for x in xs if x in s]
        if not pts:
            continue
        ax.plot(*zip(*pts), color=COL[impl], marker=MK[impl], lw=2.6 if impl == "miniworld" else 2.0, ms=8, label=LAB[impl], zorder=3 if impl != "miniworld" else 4)
    if annotate and "miniworld" in data:
        for x in xs:
            if x in data["miniworld"]:
                bo = best_other(data, x)
                if bo:
                    ax.annotate(f"{bo / data['miniworld'][x]:.1f}×", (pos[x], data["miniworld"][x]), textcoords="offset points", xytext=(0, -17),
                                ha="center", fontsize=10.5, color=COL["miniworld"], fontweight="bold")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo / 1.5, hi * 1.25)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=20)); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel); ax.set_ylabel("latency (ms)")
    if categorical:
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([x.replace("x", "×") for x in xs], fontsize=11.5)
    else:
        ax.set_xticks(xs)


def sweep_fig(mod, atom=False):
    atom = atom or mod.get("atom", False)
    k = mod["key"]
    has_d = "D" in mod or "layouts" in mod
    cols = 2 if has_d else 1
    fig, axes = plt.subplots(2, cols, figsize=(7.2 * cols, 9.4), squeeze=False)
    any_data = False
    for ri, mode in enumerate(("inference", "training")):
        md = mode[:3]
        data = {}
        for impl in ORDER:
            if mode == "training" and impl == "anthropic":
                continue
            s = series(jobs_L(mod, mode), impl, "seq")
            if atom:
                s = {x * 8: v for x, v in s.items()}       # the harness runs the atom targets at 8 x the sequence length
            if s:
                data[impl] = s
        xs = sorted({x for s in data.values() for x in s})
        ax = axes[ri][0]
        if data:
            any_data = True
            draw(ax, data, xs, "atoms (8 × tokens)" if atom else "sequence length L")
            if atom and "pytorch" in data:
                miss = [x for x in xs if x not in data["pytorch"] and "miniworld" in data and x in data["miniworld"]]
                for x in miss:
                    ax.annotate("PyTorch: out of memory", (x, data["miniworld"][x]), textcoords="offset points", xytext=(-6, 16), ha="right",
                                fontsize=11, color=COL["pytorch"], fontstyle="italic")
        ax.set_title(f"{mode} · " + ("atom-count sweep" if atom else "length sweep") + ("" if atom else f" ({mod.get('dlabel', '')} fixed)" if has_d else ""), loc="left", fontsize=14)
        if has_d:
            ax = axes[ri][1]
            data = {}
            if "layouts" in mod:
                for impl in ORDER:
                    if mode == "training" and impl == "anthropic":
                        continue
                    s = layout_series(k, md, mod["layouts"], impl)
                    if s:
                        data[impl] = s
                xs = [l for l in mod["layouts"] if any(l in s for s in data.values())]
                if data:
                    draw(ax, data, xs, mod["dlabel"], categorical=True)
            else:
                for impl in ORDER:
                    if mode == "training" and impl == "anthropic":
                        continue
                    jobs = {f"{k}_{md}_D{d}" for d in mod["D"]}
                    s = series(jobs, impl, "d")
                    if s:
                        data[impl] = s
                xs = sorted({x for s in data.values() for x in s})
                if data:
                    draw(ax, data, xs, mod["dlabel"])
            ax.set_title(f"{mode} · {mod['dlabel']} sweep (L = 384)", loc="left", fontsize=14)
    if not any_data:
        plt.close(fig); return
    h, l = {}, []
    for ax in axes.flat:
        for a, b in zip(*ax.get_legend_handles_labels()):
            h.setdefault(b, a)
    fig.legend(list(h.values()), list(h.keys()), loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{mod['title']} · B200 · bf16", y=1.045, fontsize=18, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"sweep_{k}.png")); fig.savefig(os.path.join(OUT, f"sweep_{k}.svg")); plt.close(fig)


main_fig("inference"); main_fig("training")
for mod in MODS:
    sweep_fig(mod)
for mod in ATOM:
    sweep_fig(mod, atom=True)
print("figures:", sorted(os.listdir(OUT)))

# ---- numbers behind the main figures ----
with open(os.path.join(OUT, "summary_L384.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["module", "mode", "implementation", "latency_ms", "speedup_vs_pytorch"])
    for mod in MODS:
        for mode in ("inference", "training"):
            base = at384(mod, mode, "pytorch")
            for impl in ORDER:
                t = at384(mod, mode, impl)
                if t is not None:
                    w.writerow([mod["title"], mode, LAB[impl], f"{t:.4f}", f"{base / t:.2f}" if base else ""])

# ---- every measured point: is ours the fastest? (printed, and written next to the figures) ----
from collections import defaultdict as _dd
grp = _dd(dict)
for (job, impl, s, d), v in MED.items():
    grp[(job, s, d)][impl] = v
rows, worse, total = [], [], 0
for (job, s, d), im in sorted(grp.items()):
    if "miniworld" not in im or len(im) < 2:
        continue
    others = {k: v for k, v in im.items() if k != "miniworld"}
    best = min(others, key=others.get); r = others[best] / im["miniworld"]
    total += 1; rows.append((job, s, d, im["miniworld"], best, others[best], r))
    if r < 1.0:
        worse.append((job, s, d, im["miniworld"], best, others[best], r))
with open(os.path.join(OUT, "ours_vs_fastest_other.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["job", "seq_len", "d_pair", "ours_ms", "fastest_other", "other_ms", "speedup_over_fastest_other"])
    for r in rows: w.writerow([r[0], r[1], r[2], f"{r[3]:.4f}", r[4], f"{r[5]:.4f}", f"{r[6]:.2f}"])
print(f"CHECK: {total} measured points with a comparison; ours slower than the fastest other implementation at {len(worse)}")
for r in sorted(worse, key=lambda t: t[6]):
    print(f"  SLOWER  {r[0]:22s} L={r[1]:<5d} D={r[2]:<4d} ours {r[3]:.4f} ms  {r[4]} {r[5]:.4f} ms  ({r[6]:.2f}x)")
print("min / median speedup over the fastest other:", f"{min(r[6] for r in rows):.2f} / {sorted(r[6] for r in rows)[len(rows)//2]:.2f}")

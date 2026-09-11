"""One page describing the whole autotune sweep: every kernel, one row.

`python -m miniworld_engine.viz.sweep_page` writes `docs/autotune-sweep-grid.html`.

It reads the repository, never a snapshot: `registry.csv` for what a kernel is, `op_units()` for
the shapes a build drives it at, `autotune/configs/grid/` for the ladders it searches, and
`autotune/data/` for what has actually been measured. So the page cannot disagree with the tree it
was generated from -- which is the failure it exists to prevent. A table of this shape was kept by
hand for a while and drifted from the repo twice in one afternoon: once showing a proposed
narrowing the CSVs did not carry, once showing a dtype the registry had already dropped.

The page is committed alongside the generator. Regenerate it in the same commit as any change to a
ladder, a shape list or a dtype column.
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import pathlib
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REGISTRY = PKG / "kernels" / "registry.csv"
EXEMPT = PKG / "kernels" / "tile_order_exempt.csv"
GRID = PKG / "autotune" / "configs" / "grid"
DATA = PKG / "autotune" / "data"
#: Seconds a single (config, shape) pair costs to bench, measured over two full sweeps. Bench is
#: ~97% of a unit's wall time, so this one number turns the table's last column into GPU-hours.
SECONDS_PER_CONFIG = 0.24


def _rows() -> list[dict[str, str]]:
    with REGISTRY.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _ladders(kernel: str) -> dict[str, list[str]]:
    p = GRID / f"{kernel}.csv"
    if not p.is_file():
        return {}
    with p.open(newline="") as fh:
        return {r[0]: r[1].split() for r in csv.reader(fh) if len(r) >= 2 and r[0] != "axis"}


def _exempt_reasons() -> dict[str, str]:
    if not EXEMPT.is_file():
        return {}
    with EXEMPT.open(newline="") as fh:
        return {r["kernel"]: r["reason"].strip() for r in csv.DictReader(fh)}


def _coverage() -> dict[str, float]:
    """Exact usable key coverage for the A6000 plan, never a maximum over other cards."""
    return _coverage_raw()


def _coverage_mismatch() -> dict[str, float]:
    return {}


def _coverage_raw() -> dict[str, float]:
    from miniworld_engine.autotune import derive
    rows = derive.kernel_rows("sm86")
    report = derive.coverage("sm86", "NVIDIA RTX A6000 (sm86)", DATA)
    wanted = collections.Counter(r["kernel"] for r in rows)
    missing = collections.Counter(op for op, _key in report["missing"])
    return {op: (n - missing[op]) / n for op, n in wanted.items()}


def _prune_fn_name(path: pathlib.Path, symbol: str) -> str | None:
    """The function named in ``@triton.autotune(prune_configs_by={'early_config_prune': X})``.

    Read from the SOURCE, not from the decorated object: `autotune/cache.py`'s reader replaces
    `Autotuner.early_config_prune` with its own wrapper at import, so asking the live object gives
    the cache reader's heuristic subset (24 for every kernel) instead of the kernel's own prune.
    """
    import ast

    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None
    want = symbol.split(".")[-1]
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != want:
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            for kw in dec.keywords:
                if kw.arg != "prune_configs_by" or not isinstance(kw.value, ast.Dict):
                    continue
                for k, v in zip(kw.value.keys, kw.value.values, strict=True):
                    if getattr(k, "value", None) == "early_config_prune":
                        return getattr(v, "id", None)
    return None


def _benched_per_unit(op: str, grid: int, sides) -> int:
    """Configs one unit actually compiles and times -- `grid` minus what the KERNEL's own
    `early_config_prune` deletes first.

    `units x grid` is the page's obvious cost model and it is wrong for any kernel that ships a
    prune. `autotune/cache.py`'s reader wraps `early_config_prune` and, on the BUILD path, returns
    `base(configs, nargs)` -- the kernel's own prune still runs, and `capture` makes that pruned
    list the round's work item for both compile and bench. Reporting the unpruned grid made
    `transition_fwd_b2b_triton` read as a quarter of the whole sweep when it is well under one
    percent: `_prefer_covering_b2b` pins BLOCK_K_D, BLOCK_K_ND and GROUP_M at the K its driver
    builds, cutting 28,000 configs to 560. A page that is read to decide where tuning time goes
    must not price 50x of phantom.

    Two kernels in the repository ship a prune (`transition_fwd_b2b_triton`,
    `layernorm_linear_fwd_triton`); everything else returns `grid` unchanged. Falls back to `grid`
    on any failure -- over-reporting is the safe direction for a cost estimate, under-reporting is
    not.
    """
    import importlib

    from miniworld_engine.autotune.configs import configs_for

    row = {r["kernel"]: r for r in _rows()}.get(op)
    if row is None:
        return grid
    name = _prune_fn_name(PKG.parent / row["file"], row["symbol"])
    if name is None:
        return grid
    try:
        prune = getattr(importlib.import_module(row["file"].replace("/", ".")[:-3]), name)
        cfgs = configs_for(op)
        # Evaluate at the SMALLEST width this op is driven at, not the largest. The prune keys on
        # the launch's K, and both kernels that ship one have a driver that PINS K rather than
        # following the unit's width: `drivers/transition.py:99` sets `K_SMALL = ragged(128)` and
        # says so outright ("does NOT follow the swept width, and that is not an oversight"),
        # because `transition_b2b` is dispatched only at K <= _B2B_MAX_K = 128. Taking the max
        # over the unit widths evaluates the prune at a K the driver never builds -- at K=512
        # `_prefer_covering_b2b` keeps all 28,000, and the phantom this function exists to remove
        # comes straight back.
        widths = sorted({w for _s, (_L, W) in sides.items() for w in W}) or [128]
        return min(len(list(prune(cfgs, {"K": w, "D": w, "ND": 4 * w, "M": 1 << 16})))
                   for w in widths) or grid
    except Exception:
        return grid


def _derived_rows(sm: str | None) -> list[dict]:
    """``registry_kernel.csv`` for this card, or [] when there is nothing to read.

    Empty is not an error here: a checkout that has never run `dev derive`, or a card no
    derivation has been run for, should still get a page.
    """
    from miniworld_engine.autotune import derive

    try:
        return derive.kernel_rows(sm or "sm86")
    except (FileNotFoundError, KeyError):
        return []


def collect() -> tuple[list[dict], dict]:
    """One record per kernel a build drives, plus the totals.

    Read from ``registry_kernel.csv`` -- the list `dev derive` writes by RUNNING every row of
    registry_module.csv on fake tensors -- not from ``op_units()``. The page is an inventory of
    what `build all` does, and `op_units` has not been that since the module sweep became the
    default: it enumerates the per-kernel driver ladders, which now cover only the handful of
    kernels no module reaches. Rendering those numbers said 4,948 units and 67 kernels for a build
    that runs 6,526 units over 60 kernels, which is the exact kind of drift this page exists to
    make visible.

    Falls back to ``op_units`` when the derived file is missing or is for another card, so a fresh
    checkout still renders something rather than failing.
    """
    from miniworld_engine.autotune import derive, plan
    from miniworld_engine.autotune.builder import op_units

    reg = {r["kernel"]: r for r in _rows()}
    exempt = _exempt_reasons()
    sides: dict[str, dict[str, tuple[set, set]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: (set(), set())))
    units: collections.Counter = collections.Counter()
    try:
        evidence = plan.load("sm86")
        verification = "verified against current dispatch sources"
    except (OSError, ValueError, KeyError) as exc:
        evidence = None
        verification = f"UNVERIFIED: {exc}"
    cover = _coverage() if evidence else {}
    derived_rows = _derived_rows("sm86")
    if derived_rows:
        for r in derived_rows:
            units[r["kernel"]] += 1
            for stream in r["streams"].split("|"):
                lengths, widths = sides[r["kernel"]][stream]
                lengths.update(json.loads(r.get("shapes") or "{}").get(
                    stream, [int(x) for x in r["lengths"].split("|") if x]))
    else:
        for u in op_units():
            units[u.op] += 1
            lengths, widths = sides[u.op][u.side]
            lengths.add(u.length)
            widths.add(u.width)

    out, total = [], 0
    for op in sorted(units):
        ax = _ladders(op)
        if not ax:
            continue
        grid = 1
        for values in ax.values():
            grid *= len(values)
        r = reg[op]
        # The derivation records keys, not the full per-launch prune inputs. Do not infer a
        # smaller search from a guessed width; show the declared unpruned upper bound.
        benched = grid
        cost = units[op] * benched
        total += cost
        out.append({
            "kernel": op, "kind": r["kind"], "stack": r["stack"], "dtypes": r["dtypes"],
            "where": r["file"].split("miniworld_engine/kernels/", 1)[-1],
            "level": r["level"], "width": (r["width"] or "both").strip(),
            "axes": ax, "units": units[op], "grid": grid, "benched": benched, "cost": cost,
            "cover": cover.get(op, 0.0), "exempt": exempt.get(op, ""),
            "sides": [{"name": s, "L": sorted(L), "W": sorted(W)}
                      for s, (L, W) in sorted(sides[op].items())],
        })
    out.sort(key=lambda r: -r["cost"])
    # Two different numbers, both wanted, and conflating them is how the page came to say 4,948
    # for a build that runs 6,526. `units` is what `build all` LAUNCHES -- one module invocation
    # each, from registry_module.csv. `buckets` is what it PRODUCES -- one cache entry each, from
    # registry_kernel.csv. The search cost tracks buckets, because a unit whose keys the cache
    # already answers benches nothing; the wall-clock tracks units, because each is a process.
    module_units = 0
    if derived_rows:
        module_units = len(derive.units(derive.module_rows(), arch="sm86"))
    totals = {
        "kernels": len(out), "buckets": sum(units.values()), "cost": total,
        "units": module_units or sum(units.values()),
        "hours": total * SECONDS_PER_CONFIG / 3600,
        "derived": sum(1 for r in out if r["cover"] >= 1.0),
        "unmeasured": sum(1 for r in out if r["cover"] == 0.0),
        "verification": verification,
    }
    return out, totals


CSS = """:root{
  --bg:#F4F6F8; --surface:#FFFFFF; --ink:#151A21; --muted:#5B6673; --faint:#8A94A1;
  --rule:#DCE1E7; --rule-soft:#E8ECF1; --accent:#2F6690; --accent-soft:#DCE7F0;
  --warn:#8A5D10; --warn-soft:#F5EBD6;
  --gemm:#2F6690; --reduce:#4A7C59; --elem:#7A5C8E;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0E1217; --surface:#151A21; --ink:#DFE5EC; --muted:#8C97A5; --faint:#6B7684;
  --accent-soft:#1B2733; --warn:#CFA351; --warn-soft:#2A2317;
  --gemm:#71A8D4; --reduce:#7FB18C; --elem:#AC90C4;
}}
:root[data-theme="dark"]{
  --bg:#0E1217; --surface:#151A21; --ink:#DFE5EC; --muted:#8C97A5; --faint:#6B7684;
  --rule:#242B35; --rule-soft:#1C222B; --accent:#71A8D4; --accent-soft:#1B2733;
  --warn:#CFA351; --warn-soft:#2A2317;
  --gemm:#71A8D4; --reduce:#7FB18C; --elem:#AC90C4;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Helvetica Neue",sans-serif;
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:56px 28px 96px;display:flex;flex-direction:column;gap:44px}
.num{font-family:"IBM Plex Mono","SF Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}

header.top{display:flex;flex-direction:column;gap:14px;border-bottom:2px solid var(--ink);padding-bottom:22px}
.eyebrow{font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:600;font-size:12px;
  letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
h1{margin:0;font-size:clamp(30px,4.2vw,44px);font-weight:600;letter-spacing:-.02em;text-wrap:balance;line-height:1.1}
.lede{margin:0;max-width:64ch;color:var(--muted);font-size:16px}

.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule)}
.fig{background:var(--surface);padding:16px 18px;display:flex;flex-direction:column;gap:5px}
.fig dt{font-family:"IBM Plex Sans Condensed",sans-serif;font-size:11.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--faint)}
.fig dd{margin:0;font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;
  font-size:24px;font-weight:500;letter-spacing:-.02em}
.fig .sub{font-size:12px;color:var(--muted);font-family:"IBM Plex Sans",sans-serif}

.note{border-left:3px solid var(--warn);background:var(--warn-soft);padding:14px 18px;
  display:flex;flex-direction:column;gap:6px}
.note b{font-weight:600}
.note p{margin:0;font-size:14px;color:var(--ink)}
.note code{font-family:"IBM Plex Mono",monospace;font-size:13px}

.grp{display:flex;flex-direction:column;gap:0}
.grph{display:flex;flex-direction:column;gap:10px;padding-bottom:14px;border-bottom:1px solid var(--ink)}
.grph h2{margin:0;font-size:22px;font-weight:600;letter-spacing:-.01em;
  font-family:"IBM Plex Mono",monospace}
.gd{margin:0;color:var(--muted);font-size:14px}
.gm{display:flex;flex-wrap:wrap;gap:26px;margin:2px 0 0}
.gm div{display:flex;align-items:baseline;gap:8px}
.gm dt{font-family:"IBM Plex Sans Condensed",sans-serif;font-size:11.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--faint)}
.gm dd{margin:0;font-size:14px;font-weight:500}

.tw{overflow-x:auto;border:1px solid var(--rule);border-top:0;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px}
thead th{position:sticky;top:0;background:var(--surface);z-index:1;
  font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:600;font-size:11px;
  letter-spacing:.09em;text-transform:uppercase;color:var(--faint);
  text-align:left;padding:11px 12px;border-bottom:1px solid var(--rule);white-space:nowrap}
tbody td{padding:9px 12px;border-bottom:1px solid var(--rule-soft);vertical-align:middle}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--accent-soft)}
th.r,td.r{text-align:right}
th.c,td.c{text-align:left}
td.n{color:var(--muted);font-size:12px;white-space:nowrap}
td.sides{min-width:330px}
.sd{display:grid;grid-template-columns:52px 1fr auto;gap:10px;align-items:baseline;padding:1px 0}
.sn{font-family:"IBM Plex Sans Condensed",sans-serif;font-size:10.5px;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
.sl{font-size:11.5px;color:var(--muted);white-space:nowrap}
.sw{font-size:11.5px;color:var(--accent);white-space:nowrap;font-weight:500}

td.k{min-width:270px}
.op{display:block;font-family:"IBM Plex Mono",monospace;font-size:12.5px;font-weight:500}
.fam{display:block;font-size:11px;color:var(--faint);letter-spacing:.02em}

.chip{font-family:"IBM Plex Sans Condensed",sans-serif;font-size:10.5px;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;padding:2px 7px;border:1px solid currentColor;
  border-radius:2px;white-space:nowrap}
.chip.gemm{color:var(--gemm)} .chip.reduce{color:var(--reduce)} .chip.elem{color:var(--elem)}
.lvl{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted)}
.side{margin-left:6px;font-size:10.5px;color:var(--faint);font-family:"IBM Plex Sans Condensed",sans-serif;
  letter-spacing:.06em;text-transform:uppercase}

td.cost{position:relative;min-width:132px;padding-right:14px}
td.cost .bar{position:absolute;left:0;top:50%;transform:translateY(-50%);height:22px;width:var(--w);
  background:var(--accent-soft);border-left:2px solid var(--accent)}
td.cost .v{position:relative;font-weight:500}

footer{border-top:1px solid var(--rule);padding-top:18px;color:var(--faint);font-size:13px;
  display:flex;flex-direction:column;gap:6px}
footer code{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted)}
@media (max-width:760px){.wrap{padding:36px 16px 64px}td.k{min-width:210px}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

table{font-size:12.5px}
th{position:sticky;top:0;background:var(--bg);z-index:2}
td.k .op{font-weight:500}
td.k .fam{display:block;color:var(--faint);font-size:11px;margin-top:2px}
.lad{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;white-space:nowrap}
.lv{display:inline-block;min-width:96px;color:var(--muted);font-size:11px;
  text-transform:uppercase;letter-spacing:.04em}
.gapf{color:var(--warn,#B4762A);font-weight:600;cursor:help}
.ta{display:flex;gap:6px;font-size:11px;white-space:nowrap}
.ta .an{color:var(--faint)} .ta .av{font-family:"IBM Plex Mono",monospace}
.ev{display:inline-block;padding:1px 5px;border-radius:3px;font-size:10.5px;margin-left:5px}
.ev.ok{background:#E3EEE6;color:#3C6B4B} .ev.part{background:var(--accent-soft);color:var(--muted)}
.ev.no{background:#F3E4E4;color:#8C4B4B}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .ev.ok{background:#18251C;color:#8FC29C}
 :root:not([data-theme="light"]) .ev.no{background:#2A1A1A;color:#C98A8A}}
dl.leg{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:2px 18px;
  margin:14px 0 0;font-size:11.5px;color:var(--muted)}
dl.leg>div{display:flex;gap:8px;align-items:baseline;padding:2px 0}
dl.leg dt{flex:0 0 92px;color:var(--fg);font-family:"IBM Plex Mono",monospace;font-size:11px}
dl.leg dd{margin:0}
.dt{display:inline-block;padding:1px 5px;border-radius:3px;font-size:10.5px;margin-right:3px;
  background:var(--accent-soft);color:var(--muted)}
.dt.bf16{background:#E3EEE6;color:#3C6B4B}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .dt.bf16{background:#18251C;color:#8FC29C}}
td.cost{position:relative;padding-right:10px}
td.cost .bar{position:absolute;right:0;top:50%;transform:translateY(-50%);height:20px;
  width:var(--w);background:var(--accent-soft)}
td.cost .v{position:relative}

.faint{color:var(--faint)}
"""

TEMPLATE = """<title>Autotune Sweep Grid</title>
<style>{css}</style>
<div class="wrap">
<header><h1>the autotune sweep, one row per kernel</h1>
<p class="lede">{n} kernels. Each row is what a full <code>build</code> compiles and benches
for that kernel: the shapes it is driven at, the config axes it searches, and the product.
Coverage is for NVIDIA RTX A6000 (sm86), with cache identities checked.
Config counts are unpruned upper bounds for a cold search. Per-shape pruning and existing
measurements reduce the actual work; these counts do not provide a build ETA.</p>
<p class="lede">Plan: {verification}</p>
<dl class="gm">
 <div><dt>buckets × grid</dt><dd class="num">{cost} M</dd></div>
 <div><dt>build ETA</dt><dd class="num">not measured</dd></div>
 <div><dt>module invocations declared</dt><dd class="num">{units}</dd></div>
 <div><dt>cache keys required</dt><dd class="num">{buckets}</dd></div>
 <div><dt>kernels covered</dt><dd class="num">{derived} of {n}</dd></div>
</dl>
<dl class="leg">
 <div><dt>atom_single</dt><dd>an atom count at <code>d_single_atom</code></dd></div>
 <div><dt>token_single</dt><dd>a token count at <code>d_single</code> / <code>d_single_token</code></dd></div>
 <div><dt>token_pair</dt><dd>a token count at <code>d_pair</code> — a pair activation is
   (B, L, L, D), so its L is a token count too</dd></div>
 <div><dt>two shape lines</dt><dd>driven per side: the three DiT families run on the token stream
   and the atom stream at different lengths and different widths</dd></div>
 <div><dt><span class="ev ok">100%</span></dt><dd>every required key has a usable A6000 cache entry; this does not certify numerical accuracy</dd></div>
 <div><dt><span class="ev part">%</span></dt><dd>fraction of required keys with a usable A6000 cache entry</dd></div>
 <div><dt><span class="ev no">none</span></dt><dd>no required key currently has a usable A6000 cache entry</dd></div>
 <div><dt>—</dt><dd>no column-tile axis to order (hover for the reason)</dd></div>
</dl>
</header>
<div class="tw"><table>
<thead><tr><th>kernel</th><th>kind</th><th>dtype</th><th>shape · L · d</th>
<th>tile axes</th><th>GROUP_M</th><th>warps</th><th>stages</th>
<th class="r">keys</th><th class="r">grid</th><th class="r">keys × grid</th></tr></thead>
<tbody>{rows}</tbody></table></div>
</div>
"""


# --------------------------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------------------------- #
#: The stream vocabulary, as `registry_module.csv` spells it. A row read from
#: `registry_kernel.csv` already carries one of these, so it passes straight through; the
#: `pair`/`atom`/`token` spellings are what `OpUnit.side` uses and are translated.
STREAM_NAMES = frozenset({"token_pair", "token_single", "atom_single", "msa_token"})


def shape_name(row: dict, side: str) -> str:
    """What a shape line counts and whose width it carries.

    Three names, not a length word plus a width word: splitting them printed "atom . atom" and made
    two independent axes look like one repeated thing. A pair activation is (B, L, L, D), so its L
    is a token count too -- which is why there is no "pair" on the counting side.

    The fallback below composes a name out of the registry's `level` and `width` columns, and it
    is a LAST resort: when the page started reading `registry_kernel.csv` its side values became
    stream names, none of the branches matched, and every line on the page rendered as
    "both_both" -- two column values glued together, naming nothing.
    """
    if side in STREAM_NAMES:
        return side
    if side == "pair":
        return "token_pair"
    if side == "atom":
        return "atom_single"
    if side == "token":
        return "token_single"
    w = row["width"]
    return f'{row["level"]}_{"single" if w in ("single", "atom") else w}'


def _badge(row: dict) -> str:
    c = row["cover"]
    if c >= 1.0:
        return '<span class="ev ok" title="all required A6000 keys are usable">100%</span>'
    if c == 0.0:
        return ('<span class="ev no" title="no cache entry at the precision this kernel is '
                'declared at">none</span>')
    return (f'<span class="ev part" title="{c:.0%} of required A6000 keys are usable">'
            f'{c:.0%}</span>')


def render(rows: list[dict], totals: dict) -> str:
    e = html.escape
    biggest = max(r["cost"] for r in rows) if rows else 1
    body = []
    for r in rows:
        ax = r["axes"]
        bar = max(2, round(100 * r["cost"] / biggest))
        shapes = "".join(
            f'<div class="lad"><span class="lv">{shape_name(r, s["name"])}</span>'
            f'{" ".join(str(x) for x in s["L"])}'
            + (f'<span class="faint"> &middot; d </span>{" ".join(str(x) for x in s["W"])}'
               if s["W"] else "") + "</div>"
            for s in r["sides"])
        tiles = "".join(
            f'<div class="ta"><span class="an">{e(a.replace("BLOCK_", ""))}</span>'
            f'<span class="av">{e(" ".join(v))}</span></div>'
            for a, v in sorted(ax.items()) if a.startswith("BLOCK"))
        gm = ax.get("GROUP_M")
        gcell = (f'<td class="lad">{e(" ".join(gm))}</td>' if gm else
                 f'<td class="lad faint" title="{e(r["exempt"])}">&mdash;</td>')
        dt = "".join(f'<span class="dt {e(x)}">{e(x)}</span>' for x in r["dtypes"].split("|"))
        body.append(
            f'<tr><td class="k"><span class="op">{e(r["kernel"])}</span>'
            f'<span class="fam">{e(r["where"])} &middot; {e(r["stack"])}</span></td>'
            f'<td><span class="chip {e(r["kind"])}">{e(r["kind"])}</span></td>'
            f'<td class="lad">{dt}</td><td>{shapes}</td><td class="tiles">{tiles}</td>{gcell}'
            f'<td class="lad">{e(" ".join(ax.get("num_warps", [])))}{_badge(r)}</td>'
            f'<td class="lad">{e(" ".join(ax.get("num_stages", [])))}</td>'
            f'<td class="r lad">{r["units"]:,}</td><td class="r lad">{r["grid"]:,}</td>'
            f'<td class="cost r"><span class="bar" style="--w:{bar}%"></span>'
            f'<span class="v lad">{r["cost"]:,}</span></td></tr>')
    return TEMPLATE.format(
        css=CSS, rows="".join(body), n=totals["kernels"], units=f'{totals["units"]:,}',
        buckets=f'{totals["buckets"]:,}',
        verification=e(totals.get("verification", "unverified")),
        cost=f'{totals["cost"] / 1e6:.2f}', hours=f'{totals["hours"]:.0f}',
        derived=totals["derived"], unmeasured=totals["unmeasured"],
        partial=totals["kernels"] - totals["derived"] - totals["unmeasured"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=PKG.parents[1] / "docs" / "autotune-sweep-grid.html")
    args = ap.parse_args(argv)
    rows, totals = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows, totals))
    print(f"{args.out}: {totals['kernels']} kernels, {totals['units']:,} units, "
          f"{totals['buckets']:,} required keys; {totals['verification']}; build ETA not measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

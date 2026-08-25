"""Split the generated grid into config dirs small enough to tune as separate jobs.

``builder.py`` already shards by SHAPE (--case/--dims/--length/--mode), and every such shard
carried the identical full grid. 205k configs cannot be tuned that way -- the split has to be over
the CONFIG SET itself, which is what this produces: N directories, each a drop-in
``--config-dir``.

Two properties the split must have, both learned the hard way:

COMPLETE   A shard dir needs a CSV for EVERY op, not just the ops it targets. ``run_case`` drives a
           whole module, so it touches ops this shard is not tuning; an op with no CSV gets an
           empty config list, Triton substitutes its own ``Config({})``, and the kernel dies at
           launch with ``dynamic_func() missing required positional arguments``. So every non-target
           op gets exactly one FILLER row.

IN-GRID    The filler is taken from that op's OWN generated grid, never from ``configs/accuracy``.
           ``merge_shards`` hashes the UNION of what the shards carried, so a filler from outside
           the grid would put the union above the real grid, the hash would not match a full-grid
           run, and ``store_ranked_configs`` would answer the mismatch by resetting every entry --
           discarding the build. Filling from inside the grid keeps union == full grid exactly.

Balance is by COMBINATION COUNT, not op count: 8 of the 91 ops carry 45% of the space, so one
shard per op would leave most jobs idle while a few ran for days.

    python -m miniworld_engine.tools.gen_shards.py --per-shard 8000 \
        --out src/miniworld_engine/autotune/configs/grid
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from miniworld_engine.tools.classify import SRC, classify
from miniworld_engine.tools.gen_grid import (
    GROUP_M,
    STAGES,
    VALUES,
    WARPS,
    grid_for,
    op_axes,
    role,
)

REG = SRC / "miniworld_engine/kernels/registry.csv"


def full_grid() -> dict[str, list[dict]]:
    """op -> every config in its grid, in generation order."""
    reg = {r["kernel"]: r for r in csv.DictReader(REG.open())}
    out = {}
    for op, axes in sorted(op_axes().items()):
        r = reg.get(op)
        if r is None:
            continue
        kind = classify(SRC / r["file"], r["symbol"])[0]
        out[op] = list(grid_for(op, axes, kind))
    return out


def _write_csv(path: Path, axes: list[str], rows: list[dict]) -> None:
    cols = [*axes, "num_warps", "num_stages"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


def _write_spec(path: Path, axes: list[str], kind: str, sl: tuple[int, int] | None) -> None:
    """The op's grid as `axis,values` (+ an optional slice) instead of one row per config.

    Materialising was costing 13 MB across 26 shard dirs to say something the value sets say in a
    few hundred bytes -- and 15,552 rows of one op are unreviewable, so nobody would ever have
    noticed a wrong value in them.
    """
    vals = {a: (GROUP_M if role(a) == "GROUP" else VALUES[kind][role(a)]) for a in axes}
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "values"])
        for a in axes:                       # file order IS expansion order; see configs._read_spec
            w.writerow([a, " ".join(str(v) for v in vals[a])])
        w.writerow(["num_warps", " ".join(str(v) for v in WARPS[kind])])
        w.writerow(["num_stages", " ".join(str(v) for v in STAGES[kind])])
        if sl is not None:
            w.writerow(["slice", f"{sl[0]}-{sl[1]}"])


def plan(grid: dict[str, list[dict]], per_shard: int) -> list[dict[str, tuple[int, int]]]:
    """Greedy pack: op -> (start, stop) slice, appended until a shard reaches ``per_shard``.

    An op larger than a whole shard is cut across consecutive shards; nothing is dropped, so the
    slices of an op over all shards concatenate back to its full grid.
    """
    shards: list[dict[str, tuple[int, int]]] = [{}]
    used = 0
    for op in sorted(grid, key=lambda o: -len(grid[o])):
        pos, n = 0, len(grid[op])
        while pos < n:
            room = per_shard - used
            if room <= 0:
                shards.append({})
                used, room = 0, per_shard
            take = min(room, n - pos)
            shards[-1][op] = (pos, pos + take)
            pos += take
            used += take
    return shards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-shard", type=int, default=8000, help="configs benched per job")
    ap.add_argument("--out", default="configs/grid", help="directory to hold shard-NNNN dirs")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    ap.add_argument("--grid-only", action="store_true",
                    help="write ONE unsharded config dir holding the whole grid (no slice rows)")
    args = ap.parse_args()

    grid = full_grid()
    axes = op_axes()
    total = sum(len(v) for v in grid.values())
    shards = plan(grid, args.per_shard)
    print(f"{total:,} configs / {len(grid)} ops -> {len(shards)} shards "
          f"of <= {args.per_shard:,}")
    sizes = [sum(b - a for a, b in s.values()) for s in shards]
    print(f"  shard size  min {min(sizes):,}  max {max(sizes):,}  "
          f"mean {sum(sizes) / len(sizes):,.0f}")
    split = [o for o in grid if sum(1 for s in shards if o in s) > 1]
    print(f"  ops split across shards: {len(split)}")
    if args.dry_run:
        return 0

    reg = {r["kernel"]: r for r in csv.DictReader(REG.open())}
    kinds = {op: classify(SRC / reg[op]["file"], reg[op]["symbol"])[0] for op in grid}
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if args.grid_only:
        for op in grid:
            _write_spec(root / f"{op}.csv", axes[op], kinds[op], None)
        print(f"  wrote {len(grid)} grid specs under {root} (no slices -- the whole space)")
        return 0
    manifest = []
    for i, sh in enumerate(shards):
        d = root / f"shard-{i:04d}"
        d.mkdir(exist_ok=True)
        for op in grid:
            # FILLER for a non-target op: slice 0-1, one config from this op's OWN grid, so the
            # union over all shards stays exactly equal to the full grid (see module docstring).
            _write_spec(d / f"{op}.csv", axes[op], kinds[op], sh.get(op, (0, 1)))
        manifest.append((d.name, len(sh), sum(b - a for a, b in sh.values())))

    with (root / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["shard", "target_ops", "configs"])
        w.writerows(manifest)
    # The target ops per shard, so the tuning job knows which measurements are its own.
    with (root / "targets.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["shard", "op", "start", "stop"])
        for i, sh in enumerate(shards):
            for op, (a, b) in sorted(sh.items()):
                w.writerow([f"shard-{i:04d}", op, a, b])
    print(f"  wrote {len(shards)} dirs + manifest.csv + targets.csv under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

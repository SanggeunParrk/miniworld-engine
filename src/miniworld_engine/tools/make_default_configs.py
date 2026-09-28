"""Generate initial v2.1 per-kernel defaults from the retained global domains.

Run on a compute node (`miniworld-engine dev make-default-configs`). This edits ONLY configs/default, never configs/grid or
measured caches. The result is a starting search space, not a tuning result.
"""
import csv
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "autotune" / "configs"


def preference(axis):
    if axis == "num_warps":
        return (4, 8, 2, 1, 16, 32)
    if axis == "num_stages":
        return (2, 3, 1, 4, 5, 6, 8, 10, 12)
    if axis == "GROUP_M":
        return (8, 4, 1, 2, 16, 32)
    if axis == "maxnreg":
        return (128, 192, 256, 64)
    if axis.startswith("BLOCK_K"):
        return (32, 64, 128, 256, 512, 1024, 16, 8, 4, 2, 1)
    return (64, 128, 32, 16, 8, 4, 2, 1, 256, 512, 1024)


def generate():
    target = ROOT / "default"
    target.mkdir(exist_ok=True)
    total = 0
    for source in sorted((ROOT / "grid").glob("*.csv")):
        with source.open() as handle:
            rows = list(csv.DictReader(handle))
        if "axis" not in rows[0]:
            selected = rows[:32]
            names = list(rows[0])
        else:
            names = [r["axis"] for r in rows if r["axis"] != "slice"]
            if len(names) != len(rows):
                raise ValueError(f"global domain must be unsliced: {source}")
            domains = []
            for row in rows:
                vals = [int(v) for v in row["values"].split()]
                pref = preference(row["axis"])
                domains.append(sorted(vals, key=lambda v: (pref.index(v) if v in pref else len(pref), v)))
            # Start with a conventional tile and vary EACH axis before filling
            # coupled alternatives. Reductions retain wide K and small row tiles.
            combos = [tuple(d[0] for d in domains)]
            coupled = [i for i, name in enumerate(names) if name.startswith("BLOCK_K")]
            if coupled:
                common = set.intersection(*(set(domains[i]) for i in coupled))
                for width in sorted(common):
                    c = list(combos[0])
                    for i in coupled:
                        c[i] = width
                    if "GROUP_M" in names and 1 in domains[names.index("GROUP_M")]:
                        c[names.index("GROUP_M")] = 1
                    combos.append(tuple(c))
            for level in range(1, max(map(len, domains))):
                for i, domain in enumerate(domains):
                    if level < len(domain):
                        c = list(combos[0]); c[i] = domain[level]
                        combos.append(tuple(c))
            combos.extend(itertools.product(*(d[:2] for d in domains)))
            selected, seen = [], set()
            for combo in combos:
                values = dict(zip(names, combo, strict=False))
                tiles = [v for k, v in values.items() if k.startswith("BLOCK")]
                product = 1
                for value in tiles:
                    product *= value
                if len(tiles) >= 3 and product <= 16 ** len(tiles):
                    continue
                if combo not in seen:
                    seen.add(combo)
                    selected.append(values)
                if len(selected) == 32:
                    break
        if not selected:
            raise ValueError(f"no default config for {source}")
        with (target / source.name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
            writer.writeheader(); writer.writerows(selected)
        total += len(selected)
    print(f"default: {len(list(target.glob('*.csv')))} kernels, {total} candidates; global domains unchanged")


if __name__ == "__main__":
    generate()

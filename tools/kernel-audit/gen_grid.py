"""Generate an autotune config grid per op, from the axis NAMES the op actually declares.

Why by prefix and not by a fixed list of axis names: the 91 ops declare 18 distinct axis names over
210 axes -- BLOCK_K_D, BLOCK_K_ND, BLOCK_K_NC, BLOCK_K_NX, BLOCK_K_DC, BLOCK_N_NC, BLOCK_N_ROW and
so on -- and 8 gemm ops have FOUR tile axes. A grid written as "BLOCK_M, BLOCK_N, BLOCK_K" cannot
produce a config for those ops at all. The suffixes are not noise: they distinguish several
contraction axes inside one kernel, and a previous audit measured them to be justified.

So the role comes from the prefix, and the value set from the role plus the kernel's kind:

    BLOCK_M*      row / token tile
    BLOCK_N*      output-column tile
    BLOCK_K*      contraction or reduction axis
    BLOCK_E       flat element tile
    GROUP_M       L2 swizzle group -- NOT a tile (see below)

Every value set below contains every value observed to WIN in the historical tuned caches
(sm80/sm86/sm90/sm100, 497 winning configs). Values never observed are kept rather than dropped:
the caches are asymmetric evidence -- a value being present proves it was offered, a value being
absent cannot distinguish "lost" from "never a candidate". Narrowing is the compile probe's job,
not a human's, and it narrows per device.

GROUP_M is timed, not guessed. It only reorders which (pid_m, pid_n) a program takes -- a previous
sweep over {1,2,3,4,16} produced bit-identical output on all four kernels that take it, so a wrong
value costs time and never an answer. Timing it (7 repeats, noise 0.3-3.4%) showed the pinned 8
losing by 12-25% on every case measured, and the optimum moving between kernels: `_dgemm` wants
16/32 while `_gemm_gate` at M=16384 wants 2. That is why it is an axis and not a constant. 3 is
legal (unlike num_warps, GROUP_M has no power-of-two rule) but was never best and is often worst.

num_warps MUST be a power of two -- Triton raises `num_warps must be a power of 2`, measured, up to
32 which compiles. num_stages has no compiler maximum; its ceiling is shared memory, and the cost
is exactly linear at one operand tile per stage (measured: 8192 / 32768 / 49152 B per stage for
64x64x32 / 128x128x64 / 128x256x64), so the ceiling is a division that differs per config -- 12, 4
and 3 respectively on an A6000. That is why no single stage ceiling is written here.
"""
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify import SRC, classify

REG = SRC / "miniworld_engine/kernels/registry.csv"
META = ("num_warps", "num_stages", "maxnreg")

VALUES = {
    "gemm":   {"M": (32, 64, 128, 256), "N": (32, 64, 128, 256), "K": (16, 32, 64),
               "E": (256, 512, 1024, 2048)},
    "reduce": {"M": (1, 2, 4, 8, 16, 32, 64, 128), "N": (32, 64, 128, 256),
               "K": (64, 128, 256, 512, 1024), "E": (256, 512, 1024, 2048)},
    "elem":   {"M": (16, 32, 64, 128, 256), "N": (16, 32, 64, 128, 256),
               "K": (16, 32, 64, 128, 256), "E": (16, 32, 64, 128, 256, 512, 1024, 2048)},
}
WARPS = {"gemm": (1, 2, 4, 8, 16, 32), "reduce": (1, 2, 4, 8, 16, 32),
         "elem": (1, 2, 4, 8, 16)}
STAGES = {"gemm": (1, 2, 3, 4, 5, 6, 8, 10, 12), "reduce": (1, 2, 3, 4, 5, 6),
          "elem": (1, 2, 3, 4, 5, 6)}
GROUP_M = (1, 2, 4, 8, 16, 32)


def role(axis: str) -> str:
    """The prefix decides. GROUP_M is checked first: it starts with GROUP, not BLOCK_M."""
    if axis == "GROUP_M":
        return "GROUP"
    for r in ("M", "N", "K", "E"):
        if axis.startswith(f"BLOCK_{r}"):
            return r
    raise ValueError(f"no role for axis {axis!r} -- add its prefix to role()")


def op_axes() -> dict[str, list[str]]:
    out = {}
    for p in sorted(Path("configs/accuracy").glob("*.csv")):
        rows = list(csv.DictReader(p.open(newline="")))
        if rows:
            out[p.stem] = [k for k in rows[0] if k and k not in META]
    return out


def grid_for(op: str, axes: list[str], kind: str):
    """Every (axis-values, num_warps, num_stages) combination this op's grid contains."""
    vals = [GROUP_M if role(a) == "GROUP" else VALUES[kind][role(a)] for a in axes]
    for combo in itertools.product(*vals):
        for w in WARPS[kind]:
            for s in STAGES[kind]:
                yield dict(zip(axes, combo, strict=False)) | {"num_warps": w, "num_stages": s}


def main() -> int:
    reg = {r["kernel"]: r for r in csv.DictReader(REG.open())}
    axes = op_axes()
    import collections
    total, bykind, worst = 0, collections.Counter(), []
    for op, ax in sorted(axes.items()):
        r = reg.get(op)
        kind = classify(SRC / r["file"], r["symbol"])[0] if r else "?"
        n = sum(1 for _ in grid_for(op, ax, kind))
        total += n
        bykind[kind] += n
        worst.append((n, op, kind, len(ax)))
    worst.sort(reverse=True)
    print(f"ops {len(axes)}   grid entries {total:,}")
    print("by kind: " + "  ".join(f"{k}={v:,}" for k, v in sorted(bykind.items())))
    print()
    print("largest grids:")
    for n, op, kind, na in worst[:8]:
        print(f"  {n:>9,}  {op:<48} {kind:<7} {na} axes")
    print()
    print("smallest:")
    for n, op, kind, na in worst[-4:]:
        print(f"  {n:>9,}  {op:<48} {kind:<7} {na} axes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

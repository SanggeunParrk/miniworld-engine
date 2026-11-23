"""RTX 5090: the atom DiT e4m3 parts (``dit_variants.cuda.atom_fp8_sm120``) against the default block, same process, alternating A-B.

    python token_dit_ab.py [--rounds 6] [--err] [--places expand,qkvg+expand,...] Module:A:L ...

Blocks: registry rows (eval, master weights, bf16 activations, conditioning shared by the A samples; ``dit_common``), every row gate removed
(``MIN_ROWS = 0``). Time: one CUDA graph per variant captured in the same process (default, the default again = same-cubin control, each placement
set), 3 calls per replay with a 128 MB L2 flush before each (subtracted), median of 7 replays; rounds alternate the order; per variant the
median over the rounds; ratio = default / variant (> 1 = e4m3 faster). Error (--err): against the module in float64 on the PyTorch backend,
inputs x 1 / x 0.05 / x 20: update rms / update max (relative to the update y64 - x) and judge = max|y - y64| / max|y64| (docs/fp8.md columns)."""

import argparse

import torch
from dit_common import Rig, ab, case, err, graph

from miniworld_engine.kernels.dit_variants.cuda import atom_fp8_sm120 as F

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--err", action="store_true")
ap.add_argument("--places", default="pre+post,pre,post")
args = ap.parse_args()
F.MIN_ROWS = 0
variants = [v.replace("+", ",") for v in args.places.split(",")]
print(torch.cuda.get_device_name(), "variants", variants, flush=True)

for spec in args.cases:
    mod, A, L, *opt = spec.split(":")
    rig = Rig(case(mod, int(L), int(A), pick=opt[0] if opt else None))
    F.enable_fp8(False)
    y0 = rig.run()
    outs, fired = {}, {}
    for v in variants:
        F.enable_fp8(v)
        F.STATS.update({k: 0 for k in F.STATS})
        outs[v] = rig.run()
        fired[v] = {k: n for k, n in F.STATS.items() if n}
    line = f"{spec} M {int(A) * int(L)}"
    if args.err:
        for scale in (1.0, 0.05, 20.0):
            gold, x = rig.gold(scale)
            F.enable_fp8(False)
            e = [("default", err(rig.run(scale), gold, x))]
            for v in variants:
                F.enable_fp8(v)
                e.append((v, err(rig.run(scale), gold, x)))
            print(f"  err x{scale}: " + "  ".join(f"{k} {a * 100:.2f}/{b * 100:.2f}% ({c:.1e}) rel {r * 100:.2f}%" for k, (a, b, c, r) in e), flush=True)
            del gold, x
    gs = {}
    F.enable_fp8(False)
    gs["default"] = graph(rig.step)
    gs["control"] = graph(rig.step)
    for v in variants:
        F.enable_fp8(v)
        gs[v] = graph(rig.step)
    F.enable_fp8(False)
    t = ab(gs, rounds=args.rounds)
    print(
        f"{line}: default {t['default']:.1f} us control {t['default'] / t['control']:.3f} "
        + " ".join(f"| {v} {t[v]:.1f} us {t['default'] / t[v]:.3f}" for v in variants)
        + f" | fired {fired}",
        flush=True,
    )
    del rig, gs, outs
    torch.cuda.empty_cache()

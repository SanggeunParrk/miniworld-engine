"""Add the `kind` and `level` columns to kernels/registry.csv. Re-runnable.

`kind` is MEASURED from the source by classify.py: it follows the transitive closure of the callees
a kernel inlines, because reading only the kernel's own body misclassifies a flash-attention
forward as elementwise (its `tl.dot`s live in an inlined `@triton.jit` helper) and a CUDA LayerNorm
likewise (its reduction is in a `__device__` helper).

`level` is DECLARED, not derived: where a kernel is used in the model is a property of the
architecture, not of the kernel's text. The mapping below is per family, as stated by the repo
owner, and each entry is corroborated by the module that imports the family:

  both   layernorm, layernorm_linear, fused_ln_mask, transition, gated_projection
  token  trimul_inproj, tm1, tm2, triangle_multiplication, triangle_attention,
         bias_only_attention   -- bias_only_attention is imported by
         modules/triangle_attention/{module,bidirectional}.py, i.e. it IS part of triangle attention
  atom   adaln, conditioned_transition, augmented_attention   -- augmented_attention is imported by
         modules/swa_atom_attention.py (the atom-level sliding-window attention) and by
         modules/conditioned_transition/module.py

A family absent from the map is an error rather than a default, so adding a family forces a
decision instead of silently inheriting one.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify import SRC, classify

REG = SRC / "miniworld_engine/kernels/registry.csv"

LEVEL = {
    "layernorm": "both", "layernorm_linear": "both", "fused_ln_mask": "both",
    "transition": "both", "gated_projection": "both",
    "trimul_inproj": "token", "tm1": "token", "tm2": "token",
    "triangle_multiplication": "token", "triangle_attention": "token",
    "bias_only_attention": "token",
    "adaln": "atom", "conditioned_transition": "atom", "augmented_attention": "atom",
}

ORDER = ["kernel", "backend", "family", "kind", "level", "file", "symbol", "driver", "check"]


def main() -> int:
    rows = list(csv.DictReader(REG.open()))
    unknown = sorted({r["family"] for r in rows} - set(LEVEL))
    if unknown:
        print(f"no level declared for family: {unknown}", file=sys.stderr)
        return 2

    import collections
    kinds, levels, unresolved = collections.Counter(), collections.Counter(), []
    for r in rows:
        kind, _sig, how = classify(SRC / r["file"], r["symbol"])
        if how != "closure":
            unresolved.append((r["kernel"], how))
        r["kind"] = kind
        r["level"] = LEVEL[r["family"]]
        kinds[kind] += 1
        levels[r["level"]] += 1

    if unresolved:
        print(f"symbol not resolved for {len(unresolved)} kernels -- classified on the whole file:",
              file=sys.stderr)
        for k, how in unresolved:
            print(f"   {k}: {how}", file=sys.stderr)
        return 3

    with REG.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ORDER)
        w.writeheader()
        w.writerows({k: r[k] for k in ORDER} for r in rows)
    print(f"{len(rows)} kernels   kind={dict(kinds)}   level={dict(levels)}")
    cross = collections.Counter((r["level"], r["kind"]) for r in rows)
    print("\n        gemm  reduce  elem")
    for lv in ("both", "token", "atom"):
        print(f"  {lv:<6}" + "".join(f"{cross[(lv, k)]:>6}" for k in ("gemm", "reduce", "elem")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

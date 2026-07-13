"""Demonstrate the two scenarios the cache must handle, on the real GPU:
  (A) KNOWN GPU  -> shipped cache loads, its config_space_hash matches the CURRENT kernel grid
      (not stale), and the runtime prune narrows the full grid to the cached top-K.
  (B) UNKNOWN GPU -> no cache, select_config/prune return None/full-grid and warn-once.
One-off driver (not a pytest test)."""
import importlib
import json
import warnings
from pathlib import Path

import triton
from triton.runtime.autotuner import Autotuner

from miniworld_kernels.autotune import cache as C

MODS = [
    "miniworld_kernels.kernels.triangle_attention.triton.main",
    "miniworld_kernels.kernels.augmented_attention.triton.main",
    "miniworld_kernels.kernels.adaln.triton.training",
    "miniworld_kernels.kernels.adaln.triton.inference",
    "miniworld_kernels.kernels.conditioned_transition.triton.inference",
    "miniworld_kernels.kernels.conditioned_transition.triton.composed",
    "miniworld_kernels.kernels.bias_only_attention.triton.gate_out",
    "miniworld_kernels.kernels.layernorm_linear.triton.fused",
    "miniworld_kernels.kernels.layernorm_linear.triton.stats",
    "miniworld_kernels.kernels.tm1.triton.main",
    "miniworld_kernels.kernels.tm2.triton.main",
    "miniworld_kernels.kernels.transition.triton.fused",
    "miniworld_kernels.kernels.transition.triton.main",
]

# op-tag -> the live autotuner (its .configs = the CURRENT grid, and its early_config_prune)
live = {}
for m in MODS:
    mod = importlib.import_module(m)
    for name in dir(mod):
        obj = getattr(mod, name, None)
        if isinstance(obj, Autotuner):
            ecp = getattr(obj, "early_config_prune", None)
            op = getattr(ecp, "_miniworld_op", None)
            if op:
                live[op] = obj

gk = C.gpu_key()
DATA = Path("src/miniworld_kernels/autotune/data")
print(f"gpu_key: {gk}\n")

# ---------------------------------------------------------------------------- #
# (A) KNOWN GPU: shipped cache hash matches current grid + prune narrows
# ---------------------------------------------------------------------------- #
print("=== (A) KNOWN GPU (this A100) — shipped cache HIT-ability ===")
hit = stale = 0
for opdir in sorted(p for p in DATA.iterdir() if p.is_dir()):
    op = opdir.name
    fp = opdir / f"{gk}.json"
    if not fp.exists():
        continue
    data = json.loads(fp.read_text())
    n = len(data.get("entries", {}))
    if op in live:
        want = C.config_space_hash(live[op].configs)
        ok = data.get("config_space_hash") == want
        hit += ok
        stale += (not ok)
        print(f"  {'HIT ' if ok else 'STALE'} {op:32s} buckets={n:2d} "
              f"grid={len(live[op].configs)} {'' if ok else '(hash ' + str(data.get('config_space_hash')) + ' != ' + want + ')'}")
    else:
        print(f"  ---- {op:32s} buckets={n:2d} (no live autotuner; e.g. trimul_bidir_front used by bidir module)")
print(f"  => {hit} ops HIT-able (hash matches current grid), {stale} STALE")

# Prove the runtime prune actually NARROWS on a HIT: transition_split_fwd, a shipped bucket.
print("\n  -- prune narrowing demo (transition_split_fwd) --")
at = live.get("transition_split_fwd")
if at is not None:
    fp = DATA / "transition_split_fwd" / f"{gk}.json"
    data = json.loads(fp.read_text())
    bucket = sorted(data["entries"])[0]              # e.g. "bfloat16|GROUP_M=10,N=256,n=4"
    dtype_s, bk = bucket.split("|")
    dims = dict(kv.split("=") for kv in bk.split(","))
    named = {"x_ptr": type("T", (), {"dtype": getattr(__import__("torch"), dtype_s)})(),
             "n": int(dims["n"]), "N": int(dims["N"])}
    kwargs = {"GROUP_M": int(dims["GROUP_M"])}
    C._load_cache.clear(); C._warned.clear()
    kept = at.early_config_prune(list(at.configs), named, **kwargs)
    cached_n = len(data["entries"][bucket])
    print(f"     bucket {bucket}: full grid={len(at.configs)} -> prune kept={len(kept)} "
          f"(cached top-K={cached_n})  {'NARROWED (HIT)' if len(kept) < len(at.configs) else 'NOT narrowed'}")

# ---------------------------------------------------------------------------- #
# (B) UNKNOWN GPU (simulated): miss -> None + warn-once, still correct
# ---------------------------------------------------------------------------- #
print("\n=== (B) UNKNOWN GPU (simulated 'FUTURE GPU (sm999)') ===")
C._load_cache.clear(); C._warned.clear()
_orig = C.gpu_key
C.gpu_key = lambda *a, **k: "FUTURE GPU (sm999)"
try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = C.select_config("transition_split_fwd", dtype="bfloat16", bucket="GROUP_M=10,N=256,n=4")
        # prune path on unknown gpu: returns the FULL grid (correct, self-tunes)
        named = {"x_ptr": type("T", (), {"dtype": __import__("torch").bfloat16})(), "n": 4, "N": 256}
        kept = at.early_config_prune(list(at.configs), named, GROUP_M=10) if at else []
    print(f"  select_config -> {r}  (expect None)")
    print(f"  prune kept {len(kept)}/{len(at.configs) if at else 0} configs (expect FULL grid = no narrowing)")
    print(f"  warnings emitted: {len(w)}")
    for x in w[:2]:
        print(f"    * {str(x.message)[:110]}")
finally:
    C.gpu_key = _orig

print("\nCACHE HIT/MISS CHECK DONE")

#!/bin/bash
#SBATCH -J atom-flat
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -w gpu04
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o src/miniworld_engine/tools/prof/atomflat-%j.log
set -u
R=${MINIWORLD_REPO:-$(git rev-parse --show-toplevel)}
cd "$R"
export PYTHONPATH=src
export TRITON_CACHE_DIR=${MINIWORLD_TRITON_CACHE:-$HOME/.cache/miniworld-triton}
PY=.pixi/envs/default/bin/python

# The 18 level=both kernels of the transition / trimul families, at the atom buckets that used to
# build M = L*L. L=8192 asked for 67,108,864 rows (16 GiB at D=128) and OOM'd; it should now be
# 8,192 rows and run in milliseconds.
$PY - <<'PY'
import csv, os, time, importlib, torch
reg = list(csv.DictReader(open("src/miniworld_engine/kernels/registry.csv")))
both = [r for r in reg if r["level"] == "both"
        and r["family"] in ("transition", "triangle_multiplication", "trimul_inproj",
                            "tm1", "tm2", "gated_projection")]
print(f"level=both kernels in the transition / trimul families: {len(both)}\n")
for L in (1024, 4096, 8192):
    os.environ["MINIWORLD_DRIVER_LENGTH"] = str(L)
    for m in list(importlib.sys.modules):
        if "miniworld_engine.kernels" in m or "miniworld_engine.autotune" in m:
            del importlib.sys.modules[m]
    ok = bad = 0; worst = 0.0; errs = []
    for r in both:
        mod, _, fn = r["driver"].partition(":")
        try:
            f = getattr(importlib.import_module(mod), fn)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            f()
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            worst = max(worst, dt); ok += 1
        except Exception as e:
            bad += 1; errs.append(f"{fn}: {type(e).__name__}: {str(e)[:70]}")
        torch.cuda.empty_cache()
    print(f"  L={L:<5} ok={ok:<3} failed={bad:<3} slowest_driver={worst:6.2f}s")
    for e in errs[:4]: print(f"      {e}")
PY

"""RTX 5090 FP8 route census: which ``engine-fp8`` routes one FoldForge predictor reaches on this card, per slot (``fp8.route_stats()``).

    python route_census.py --model af3 [--sizes 1 4] [--repeats 1] [--ff-repo ~/ff/FoldForge] [--out DIR] [--guard lazy|sync] [--no-fp8]
    python route_census.py --model af3 --dry          # the slot table only (module per slot, switches): no predictor is loaded

Environment as FoldForge's ``engine_fp8`` arm (``benchmarks/paper/common/ffpaper.apply_arm_env``), set here before the engine is imported:
``MINIWORLD_FP8=safe``, ``MINIWORLD_FP8_MODEL=<model>`` (TriAttn only for a model in ``fp8.TRIATTN_MODELS``), lazy guards (default; ``--guard sync``
keeps the eager redo). Every other ``MINIWORLD_*FP8*`` variable of the caller is cleared first, so the census is the preset's. ``--no-fp8``: no FP8
variable at all (every counter must stay 0: the default route).

The predictor runs through FoldForge's engine backend on the 5I28 harness of the FoldForge checkout (``benchmarks/5i28_ff/run_5i28_ff.py``: real MSA +
templates, ``make_config`` / ``make_spec``; the same inputs as F3), one forward call per size n (chains; 128 tokens a chain), ``--repeats`` warm repeats
inside it. Before each size the counters are reset; after it the script prints, per slot of ``fp8.SLOTS``: the module that serves the slot on this
card (an ``SM120_ROUTES`` module, or ``-`` when the slot keeps the H100 module, whose ``serves()`` is False off sm_90: no 5090 route yet), its switch
(``fp8.status()``) and the non-zero counters; then ``fp8.check()`` (True: a static-scale route saturated during the call). One JSON line per size
(``CENSUS {...}``) carries the same data for collection. The counters bump at Python / capture time: a CUDA-graph replay does not count again.

Run on cssb-rtx under the GPU lock (``~/mwe/gpu.sh SECONDS python route_census.py ...``) with the FoldForge card environment
(``benchmarks/paper/common/cards/rtx5090.env`` of the checkout) and the engine source on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--model", required=True, help="FoldForge model name (af3, protenix1, boltz2, chai1, ...)")
ap.add_argument("--sizes", type=int, nargs="+", default=[1], help="5I28 chain counts (128 tokens a chain)")
ap.add_argument("--repeats", type=int, default=1, help="warm repeats inside each forward call (benchmark_repeats)")
ap.add_argument("--ff-repo", type=Path, default=Path(os.environ.get("FF_REPO", Path.home() / "ff/FoldForge")))
ap.add_argument("--out", type=Path, default=None, help="directory for the specs and census.jsonl (default: a temporary directory)")
ap.add_argument("--guard", choices=["lazy", "sync"], default="lazy")
ap.add_argument("--no-fp8", action="store_true", help="no FP8 variable: the default route (every counter 0)")
ap.add_argument("--dry", action="store_true", help="print the slot table (modules and switches) and exit: no predictor")
ap.add_argument("--any-arch", action="store_true", help="do not refuse a card other than the RTX 5090")
ap.add_argument("--no-compile", action="store_true", help="diagnostics: execution.compile off (CUDA graphs kept)")
ap.add_argument("--patch", action="append", default=[], help="diagnostics: module:ATTR=literal set after the engine is imported")
args = ap.parse_args()

# ---- the engine_fp8 arm's environment, before the engine is imported
for key in [k for k in os.environ if k.startswith("MINIWORLD_") and "FP8" in k]:
    os.environ.pop(key)
if not args.no_fp8:
    os.environ["MINIWORLD_FP8"] = "safe"
    os.environ["MINIWORLD_FP8_MODEL"] = args.model
    if args.guard == "lazy":
        for route in ("TRIMUL", "OPM", "PWA", "TRANSITION"):
            os.environ[f"MINIWORLD_{route}_FP8_GUARD"] = "lazy"

import torch  # noqa: E402

from foldjet_engine import fp8  # noqa: E402

# diagnostics: --patch module:ATTR=python-literal (e.g. foldjet_engine.kernels.trimul_inproj.cuda.sm120_fp8:POLICY=None), after the preset was read
for spec in args.patch:
    import ast
    import importlib

    mod_name, assign = spec.split(":", 1)
    attr, value = assign.split("=", 1)
    setattr(importlib.import_module(mod_name), attr, ast.literal_eval(value))
    print(f"PATCH {mod_name}.{attr} = {value}", flush=True)
ARCH = fp8.arch()
if ARCH != "sm120" and not args.any_arch:
    sys.exit(f"route_census: this card is {ARCH} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'no CUDA'}), not the RTX 5090 (--any-arch to run anyway)")


def slot_modules() -> dict[str, str]:
    """slot -> the module serving it on this card ('-' = the H100 module stays in the slot: no route on this card)."""
    mods = fp8._modules()
    own = fp8.SM120_ROUTES if ARCH == "sm120" else fp8.B200_ROUTES if ARCH == "sm100" else None
    out = {}
    for slot, mod in zip(fp8.SLOTS, mods, strict=True):
        if own is None or slot in own:
            out[slot] = mod.__name__.rsplit(".", 1)[-1]
        else:
            out[slot] = "-"
    return out


def table(stats: dict, status: dict, mods: dict) -> str:
    rows = [f"{'slot':<11} {'module':<20} {'switch':<28} counters (non-zero)"]
    for slot in fp8.SLOTS:
        sw = status.get(slot)
        nz = {k: v for k, v in stats.get(slot, {}).items() if v}
        rows.append(f"{slot:<11} {mods[slot]:<20} {json.dumps(sw):<28} {json.dumps(nz) if nz else '0'}")
    return "\n".join(rows)


MODS = slot_modules()
STATUS = fp8.status()
print(f"CENSUS card={torch.cuda.get_device_name()} arch={ARCH} model={args.model} fp8={'off' if args.no_fp8 else 'safe'} guard={args.guard} "
      f"engine={Path(fp8.__file__).parent}", flush=True)
print(table({}, STATUS, MODS), flush=True)
if args.dry:
    sys.exit(0)

# ---- the predictor through FoldForge's engine backend, on the 5I28 harness of the checkout
sys.path.insert(0, str(args.ff_repo / "benchmarks/5i28_ff"))
import run_5i28_ff as H  # noqa: E402
import yaml  # noqa: E402

import foldforge as ff  # noqa: E402

out = args.out or Path(tempfile.mkdtemp(prefix="route_census_"))
out.mkdir(parents=True, exist_ok=True)
recycles, template_n, extra = H.MODELS[args.model]
config = H.make_config(recycles, extra, args.repeats)
if args.no_compile:
    config["execution"]["compile"] = False
predictor = ff.load_model(args.model, config=config)
fp8.check()  # clear a flag left from loading
for n in args.sizes:
    spec = out / f"x{n}.yaml"
    spec.write_text(yaml.safe_dump(H.make_spec(n, template_n), sort_keys=False))
    fp8.reset_route_stats()
    t0 = time.time()
    rec = {"model": args.model, "n": n, "tokens": H.TOKENS_PER_CHAIN * n, "arch": ARCH, "fp8": not args.no_fp8, "guard": args.guard}
    try:
        result = predictor(ff.load_inputs(spec))
        rec["status"] = "ok"
        result = None
    except Exception as error:  # noqa: BLE001 - a failed size is data
        rec |= {"status": "failed", "error": str(error).splitlines()[0][:300] if str(error) else type(error).__name__}
    # which backend served every engine class (FoldForge's helper, when the checkout has it): a Triton / PyTorch route in an engine run is a bug
    try:
        sys.path.insert(0, str(args.ff_repo / "benchmarks/paper/common"))
        import ffpaper

        bt = ffpaper.backend_table(predictor)
        rec["all_cuda"] = None if bt is None else bt["all_cuda"]
        if bt is not None and not bt["all_cuda"]:
            rec["not_cuda"] = {c: v for c, v in bt["classes"].items() if set(v["set"]) - {"cuda"} or set(v["served"]) - {"cuda"} or v["refused"]}
    except ImportError:
        rec["all_cuda"] = None
    stats = fp8.route_stats()
    rec |= {"saturated": fp8.check(), "wall_s": round(time.time() - t0, 1), "modules": MODS, "status_fp8": fp8.status(), "route_stats": stats}
    print(f"=== {args.model} n={n} ({rec['tokens']} tokens) {rec['status']} wall={rec['wall_s']}s saturated={rec['saturated']}", flush=True)
    print(table(stats, rec["status_fp8"], MODS), flush=True)
    print("CENSUS " + json.dumps(rec, default=str, sort_keys=True), flush=True)
    with open(out / "census.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str, sort_keys=True) + "\n")
    if hasattr(H, "release_size_memory"):
        H.release_size_memory()
print(f"CENSUS done: {out / 'census.jsonl'}", flush=True)

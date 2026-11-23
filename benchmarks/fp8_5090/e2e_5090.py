"""RTX 5090 end level of the whole ``engine-fp8`` set (``MINIWORLD_FP8=safe`` + ``MINIWORLD_FP8_MODEL``) against bf16, per TRUNK seed: the H100 protocol of
docs/fp8.md "End level" (FoldForge ``miniworld`` backend, bf16, eager, bucketing, 200 steps, diffusion_seed 0, 1 sample, trunk seeds 0-3; acceptance =
max same-seed RMSD of FP8 vs bf16 <= max(1.1 x the largest bf16 trunk-seed-pair RMSD, 0.5 A); analysis ``e2e_5090_an.py``).

    python e2e_5090.py --model af3 --arm bf16|fp8 --out DIR --specs DIR TARGET ...  [--seeds 0 1 2 3] [--deadline SECONDS]

One process per (model, arm): the arm's environment is set before the engine is imported (as FoldForge's ``ffpaper.apply_arm_env``, guards ``sync``:
an eager redo of a saturating call). FP8 arm: ``fp8.check()`` after every forward (a saturation of a lazy route: the forward is redone, at most 2 times)
and the route counters per case. Resumable: one .pt (coords, plddt, ptm, iptm) + .json per finished case."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--arm", choices=["bf16", "fp8"], required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--specs", type=Path, required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--deadline", type=float, default=1e9, help="stop starting new cases after this many seconds (resumable)")
ap.add_argument("targets", nargs="+")
args = ap.parse_args()
for key in [k for k in os.environ if k.startswith("MINIWORLD_") and "FP8" in k]:
    os.environ.pop(key)
if args.arm == "fp8":
    os.environ["MINIWORLD_FP8"] = "safe"
    os.environ["MINIWORLD_FP8_MODEL"] = args.model

import torch  # noqa: E402

import foldforge as ff  # noqa: E402
from miniworld_engine import fp8  # noqa: E402

T0 = time.time()
EXTRA = {"rosettafold3": {"debug": {"dense_spec_override": {"use_input_templates": True}}}}
cfg = {"backend": "miniworld", "precision": "bf16", "diffusion": {"steps": 200},
       "execution": {"compile": False, "cuda_graph": False, "bucketing": True, "scope": "trunk"}}  # fmt: skip
for k, v in EXTRA.get(args.model, {}).items():
    cfg.setdefault(k, {}).update(v)
args.out.mkdir(parents=True, exist_ok=True)
print(f"E2E model={args.model} arm={args.arm} engine={Path(fp8.__file__).parent} arch={fp8.arch()} status={json.dumps(fp8.status(), default=str)}", flush=True)
t0 = time.time()
with ff.load_model(args.model, config=cfg, device="cuda:0") as model:
    print("LOADED", round(time.time() - t0), flush=True)
    fp8.check()
    for t in args.targets:
        inputs = None
        for ts in args.seeds:
            name = f"{args.arm}_t{ts}"
            f = args.out / f"{t}.{name}.json"
            if f.exists():
                continue
            if time.time() - T0 > args.deadline:
                print("DEADLINE: stop (resumable)", flush=True)
                sys.exit(0)
            tc = time.time()
            try:
                if inputs is None:
                    inputs = ff.load_inputs(args.specs / f"{t}.yaml")
                fp8.reset_route_stats()
                redo, sat = 0, []
                while True:
                    with torch.no_grad(), model.run_trunk(inputs, seed=ts) as trunk:
                        res = model.sample_diffusion(trunk, samples=1, steps=200, seed=0)
                    s = bool(fp8.check()) if args.arm == "fp8" else False
                    sat.append(s)
                    if not s or redo >= 2:
                        break
                    redo += 1
                    print(f"SATURATED {t} {name}: redo {redo}", flush=True)
                p = res.prediction
                torch.save({"coords": p.coords.float().cpu(), "plddt": None if p.plddt is None else p.plddt.float().cpu(),
                            "ptm": None if p.ptm is None else p.ptm.float().cpu(), "iptm": None if p.iptm is None else p.iptm.float().cpu()},
                           args.out / f"{t}.{name}.pt")  # fmt: skip
                stats = {k: {kk: vv for kk, vv in v.items() if vv} for k, v in fp8.route_stats().items()} if args.arm == "fp8" else {}
                info = {"target": t, "case": name, "arm": args.arm, "model": args.model, "trunk_seed": ts, "secs": round(time.time() - tc, 1),
                        "redo": redo, "saturated": sat, "route_stats": stats}  # fmt: skip
                f.write_text(json.dumps(info, default=str))
                print(f"DONE {t} {name} {time.time() - tc:.0f}s redo={redo} sat={sat} routes={json.dumps(stats, default=str)}", flush=True)
            except Exception as e:  # noqa: BLE001 - a failed case is data
                traceback.print_exc()
                print(f"FAIL {t} {name} {type(e).__name__}: {str(e)[:300]}", flush=True)
                torch.cuda.empty_cache()
print("END", round(time.time() - T0), flush=True)

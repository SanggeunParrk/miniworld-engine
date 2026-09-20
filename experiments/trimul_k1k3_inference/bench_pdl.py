"""Programmatic dependent launch on / off in ONE process with alternating CUDA-graph replays (same clock and thermal state), for a single op
and for the two-op chain.  The host flag ``trimul_native.kernel.PDL`` is read at every launch, so it is toggled between the two captures; the
kernels of a TMN_PDL=1 payload carry the griddepcontrol wait / trigger either way.  Outputs of the two graphs are compared bitwise.

    TRIMUL_NATIVE_BUILD_DIR=<payload>/build python bench_pdl.py --length 384 --output pdl-L384.json"""
import argparse
import importlib
import json
import statistics

import torch

import fixture

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--mask-dtype", default="bf16", choices=sorted(fixture.MASK_DTYPES))
p.add_argument("--rounds", type=int, default=8)
p.add_argument("--output", required=True)
a = p.parse_args()
fx = fixture.make(a.length, a.width, True, a.mask_dtype)
f2 = fixture.second(fx, outgoing=False)
F = fixture.face()
KM = importlib.import_module("trimul_native.kernel")
if not hasattr(KM, "PDL"):
    raise SystemExit("this payload's host package has no PDL switch (build with the overlay of this experiment)")
cache = {}


def single():
    return F.serve(fx.x, fx.pairmask, direction="outgoing", weights=fx.weights, residual=True, cache=cache, eps=fx.eps, config=None)


def chain():
    return F.serve(single(), fx.pairmask, direction="incoming", weights=f2.weights, residual=True, cache=cache, eps=fx.eps, config=None)


res = {}
with torch.no_grad():
    for mode, fn, ref in (("single", single, fx.ref), ("chain", chain, f2.ref)):
        graphs = {}
        for flag in (False, True):
            KM.PDL = flag
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            graphs[flag] = fixture.capture(fn)
        times = {False: [], True: []}
        for rnd in range(a.rounds):
            for flag in ((False, True) if rnd % 2 == 0 else (True, False)):
                times[flag].append(fixture.replay_us(graphs[flag][0]))
        errs = {flag: fixture.error(graphs[flag][1], ref) for flag in (False, True)}
        r = dict(off_us=statistics.median(times[False]), on_us=statistics.median(times[True]), off_all=times[False], on_all=times[True],
                 err_off=errs[False], err_on=errs[True], same_output=bool(torch.equal(graphs[False][1], graphs[True][1])))
        r["gain_us"] = r["off_us"] - r["on_us"]
        res[mode] = r
        print("RESULT", mode, json.dumps(dict(off=round(r["off_us"], 2), on=round(r["on_us"], 2), gain=round(r["gain_us"], 2),
                                              off_spread=round(max(times[False]) - min(times[False]), 2), on_spread=round(max(times[True]) - min(times[True]), 2),
                                              same_output=r["same_output"], err=round(errs[True]["rel_rms"], 6))), flush=True)
json.dump(dict(length=a.length, mask_dtype=a.mask_dtype, res=res), open(a.output, "w"), indent=1)

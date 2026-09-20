"""One TriMul op through the payload face: per-kernel CUPTI durations (K1 / cuBLAS contraction / K3 / other) over eager calls, whole-op time as
a one-call CUDA graph (3 rounds x 100 replays after 20 warm), fp32-reference error before and after the replays.

    TRIMUL_NATIVE_BUILD_DIR=<payload>/build python bench_op.py --length 768 --iters 60 --output op-L768.json [--mask-dtype bf16|bool|fp32]
    --configs baseline | sweep | '<json list of face config dicts>' | @file      --profile-one: bracket one call with cudaProfilerStart/Stop (NCU)

The pair mask dtype selects the K1 instantiation (m1 fp32 / m2 bf16 / m3 bool); the adoption benchmark hands the face a bf16 mask."""
import argparse
import json
import os
import statistics
import sys

import torch

import fixture

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--direction", default="outgoing", choices=["outgoing", "incoming"])
p.add_argument("--configs", default="baseline")
p.add_argument("--iters", type=int, default=60)
p.add_argument("--output", required=True)
p.add_argument("--mask-dtype", default="bf16", choices=sorted(fixture.MASK_DTYPES))
p.add_argument("--profile-one", action="store_true")
p.add_argument("--save", default="", help="torch.save the bf16 output of the first config (bitwise comparisons between payloads)")
a = p.parse_args()

fx = fixture.make(a.length, a.width, a.direction == "outgoing", a.mask_dtype)
F = fixture.face()
print("face", F.__file__, "build", os.environ["TRIMUL_NATIVE_BUILD_DIR"], flush=True)


def call(cfg, cache):
    return F.serve(fx.x, fx.pairmask, direction=fx.direction, weights=fx.weights, residual=True, cache=cache, eps=fx.eps, config=cfg)


def measure(cfg):
    cache = {}
    with torch.no_grad():
        y = call(cfg, cache)
        torch.cuda.synchronize()
        err = fixture.error(y, fx.ref)
        if a.save and not os.path.exists(a.save):
            torch.save(dict(y=y.cpu(), ref=fx.ref.cpu()), a.save)
        sel = {str(k): str(v)[:200] for k, v in cache.items() if isinstance(k, tuple) and k and k[0] == "_sel"}
        for _ in range(5):
            call(cfg, cache)
        torch.cuda.synchronize()
        kernels, names = fixture.cupti_per_kernel(lambda: call(cfg, cache), a.iters)
        g, yg = fixture.capture(lambda: call(cfg, cache))
        rounds = [fixture.replay_us(g) for _ in range(3)]
        err_replay = fixture.error(yg, fx.ref)
    return dict(config=cfg, error=err, error_after_replay=err_replay, kernels=kernels, kernel_names=names,
                op_us=dict(rounds=rounds, median=statistics.median(rounds)), selection=sel)


K1S = [(6, 32, 8, 2), (3, 64, 8, 2), (2, 64, 8, 2), (1, 128, 8, 2), (2, 64, 4, 2)]
K3S = [(3, 64, 8, 1), (2, 64, 8, 1), (2, 64, 8, 2), (1, 128, 8, 1), (2, 64, 4, 1), (1, 64, 8, 1)]
if a.configs == "baseline":
    cfgs = [None]
elif a.configs == "sweep":
    cfgs = [None, {"variant": "exact"}] + [{"k1_cfg": k} for k in K1S] + [{"k3_cfg": k} for k in K3S]
else:
    cfgs = json.loads(open(a.configs[1:]).read() if a.configs.startswith("@") else a.configs)
if a.profile_one:
    cache = {}
    with torch.no_grad():
        for _ in range(6):
            call(cfgs[0], cache)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        call(cfgs[0], cache)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    print("PROFILED one call", flush=True)
    sys.exit(0)
results = []
for cfg in cfgs:
    try:
        r = measure(cfg)
    except Exception as ex:  # a refused or unbuilt configuration is recorded, not fatal
        r = dict(config=cfg, failed=repr(ex)[:400])
    results.append(r)
    ks = r.get("kernels", {})
    print("RESULT", json.dumps(dict(config=cfg, op_us=r.get("op_us", {}).get("median"), k1=ks.get("k1", {}).get("sum_per_call_us"),
                                    k3=ks.get("k3", {}).get("sum_per_call_us"), cublas=ks.get("cublas", {}).get("sum_per_call_us"),
                                    other=ks.get("other", {}).get("sum_per_call_us"), err=r.get("error"), err_replay=r.get("error_after_replay"),
                                    failed=r.get("failed"), names=r.get("kernel_names"))), flush=True)
    json.dump(dict(length=a.length, width=a.width, direction=a.direction, mask_dtype=a.mask_dtype, iters=a.iters, results=results), open(a.output, "w"), indent=1)

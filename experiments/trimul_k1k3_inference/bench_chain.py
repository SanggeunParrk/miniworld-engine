"""Two chained TriMul ops (outgoing, then incoming on its output) as the pairformer issues them, in ONE CUDA graph: chain wall time,
per-kernel CUPTI durations, reference error before and after the replays.  This is where the K3 -> next-op K1 boundary (programmatic
dependent launch) is visible; a single-op graph has no such edge.

    TRIMUL_NATIVE_BUILD_DIR=<payload>/build python bench_chain.py --length 384 --iters 60 --output chain-L384.json"""
import argparse
import json
import statistics

import torch

import fixture

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--iters", type=int, default=60)
p.add_argument("--mask-dtype", default="bf16", choices=sorted(fixture.MASK_DTYPES))
p.add_argument("--output", required=True)
a = p.parse_args()
fx = fixture.make(a.length, a.width, True, a.mask_dtype)
f2 = fixture.second(fx, outgoing=False)
F = fixture.face()
cache = {}


def chain():
    y1 = F.serve(fx.x, fx.pairmask, direction="outgoing", weights=fx.weights, residual=True, cache=cache, eps=fx.eps, config=None)
    return F.serve(y1, fx.pairmask, direction="incoming", weights=f2.weights, residual=True, cache=cache, eps=fx.eps, config=None)


with torch.no_grad():
    y = chain()
    torch.cuda.synchronize()
    err = fixture.error(y, f2.ref)
    for _ in range(5):
        chain()
    torch.cuda.synchronize()
    kernels, names = fixture.cupti_per_kernel(chain, a.iters)
    g, yg = fixture.capture(chain)
    rounds = [fixture.replay_us(g) for _ in range(5)]
    err_replay = fixture.error(yg, f2.ref)
ksum = sum(v["sum_per_call_us"] for v in kernels.values())
out = dict(length=a.length, mask_dtype=a.mask_dtype, chain_us=statistics.median(rounds), rounds=rounds, kernels=kernels, kernel_names=names,
           kernel_sum_us=ksum, error=err, error_after_replay=err_replay)
print("RESULT", json.dumps(dict(chain_us=round(out["chain_us"], 2), kernel_sum_us=round(ksum, 2), err=round(err["rel_rms"], 6),
                                err_replay=round(err_replay["rel_rms"], 6), **{k: round(v["sum_per_call_us"], 2) for k, v in kernels.items()})), flush=True)
json.dump(out, open(a.output, "w"), indent=1)

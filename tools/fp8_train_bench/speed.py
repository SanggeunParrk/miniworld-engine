"""Training step (fwd + bwd of one block, single / cond / pair / params all differentiated) per FP8 variant: one CUDA graph per variant,
same process, interleaved with a same-cubin control (the default captured twice). cool = 2 s idle + one interleaved replay of every
graph, 5 reps, median; sustained = 2 s idle, 1 s back-to-back interleaved, 3 timed interleaved blocks of 5 replays each, mean.
usage: speed.py A:L ...   env VARIANTS="name=places;..." (default set below)"""

import json
import os
import statistics
import sys
import time

import common
import torch

from miniworld_engine.kernels.conditioned_transition.cuda import token_dit_fp8_train as F8

VARS = (
    os.environ.get("VARIANTS")
    or "default=;control=;tr_fwd=expand,squeeze;tr_bwd=dexpand,dsqueeze;transition=transition;attention=attention;all=all"
)
VARS = [v.split("=") for v in VARS.split(";")]
out = open("../res/" + os.environ.get("OUT", "speed.jsonl"), "a")  # noqa: SIM115


def step(m, inp, cl, dout):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(cond=cl.expand(inp["single"].shape[0], -1, -1, -1), **inp)
    out.backward(dout)


def make(A, L, places, e5=False):
    F8.PLACES = F8.parse(places)
    F8.GRAD_E5M2 = e5
    m = common.block()
    inp = common.inputs(m, A, L)
    inp["single"] = inp["single"].detach().clone().requires_grad_(True)
    inp["pair"] = inp["pair"].detach().clone().requires_grad_(True)
    cl = inp.pop("cond").detach()[:1].clone().requires_grad_(True)
    dout = torch.randn(A, 1, L, 768, device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(7))
    ps = [p for p in m.parameters() if p.requires_grad]
    leaves = [*ps, inp["single"], inp["pair"], cl]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for p in leaves:
                p.grad = None
            step(m, inp, cl, dout)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    for p in leaves:
        p.grad = None
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step(m, inp, cl, dout)
        torch.cuda.synchronize()
        return g.replay, (g, m, inp, dout)
    except Exception as ex:
        print("graph capture failed, eager timing:", repr(ex)[:200], flush=True)
        torch.cuda.synchronize()
        return (lambda: step(m, inp, cl, dout)), (m, inp, dout)


def tone(g):
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    g()
    b.record()
    return a, b


for case in sys.argv[1:]:
    if int(os.environ.get("DEADLINE", 1e12)) - time.time() < 90:
        print("DEADLINE")
        break
    A, L = map(int, case.split(":"))
    graphs = {}
    for name, places in VARS:
        try:
            graphs[name] = make(A, L, places, e5=name.endswith("_e5"))
        except Exception as ex:  # noqa: PERF203
            print("capture fail", name, repr(ex)[:300])
            raise
    F8.PLACES = frozenset()
    names = list(graphs)
    # cool
    cool = {n: [] for n in names}
    for _ in range(5):
        time.sleep(2)
        ev = [(n, tone(graphs[n][0])) for n in names]
        torch.cuda.synchronize()
        for n, (a, b) in ev:
            cool[n].append(a.elapsed_time(b))
    # sustained
    time.sleep(2)
    t0 = time.time()
    while time.time() - t0 < 1.0:
        for n in names:
            graphs[n][0]()
    torch.cuda.synchronize()
    sus = {n: [] for n in names}
    for _ in range(3):
        ev = []
        for _r in range(5):
            for n in names:
                ev.append((n, tone(graphs[n][0])))
        torch.cuda.synchronize()
        for n, (a, b) in ev:
            sus[n].append(a.elapsed_time(b))
    base_c, base_s = statistics.median(cool["default"]), statistics.mean(sus["default"])
    row = {"case": case, "M": A * L, "default_ms": [round(base_c, 4), round(base_s, 4)]}
    for n in names:
        row[n] = [round(statistics.median(cool[n]) / base_c, 4), round(statistics.mean(sus[n]) / base_s, 4)]
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n")
    out.flush()
    del graphs
    torch.cuda.empty_cache()

"""PWA training step (forward + backward) FP8 variants vs the 2.5.14 default, same process, CUDA graphs, interleaved.
usage: PWA_BENCH_OUT=path-without-.jsonl pwa_step_bench.py CASE...   CASE = S:N[:d_pair]  (d_msa 64, 8 x 32), p_drop P_DROP (0.15)
Graphs: def (FP8 off), ctl (FP8 off, second capture = same-cubin control), and each VARIANTS entry (name=places/impl/do).
cool = 2 s idle, then a short interleaved burst (~10 ms per graph), 3 reps, median; sustained = 1 s of back-to-back interleaved replays,
then 3 interleaved timed blocks (~20 ms per graph), mean. Also the kernel list of one step per graph (profiler, mean of 3)."""

import json
import os
import statistics
import sys
import time

import torch

specs = sys.argv[1:]
OUT = os.environ.get("PWA_BENCH_OUT", "pwa_step_bench") + ".jsonl"
DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):  # noqa: SIM115
        DONE.add(json.loads(line)["case"])
DEADLINE = int(os.environ.get("DEADLINE", time.time() + 500)) - 40
dev = "cuda"
P_DROP = float(os.environ.get("P_DROP", "0.15"))
PROFILE = os.environ.get("PROFILE", "1") == "1"

from miniworld_engine.kernels.integrations import pwa_fp8_train as F
from miniworld_engine.modules.msa import MSAPairWeightedAveraging

VARIANTS = []
for item in os.environ.get("VARIANTS", "bwd=dv,dw/fused/e4m3;bwdo8=dv,dw/fused/e4m3+o8;f8e4=fwd,dv,dw/fused/e4m3").split(";"):
    name, rest = item.split("=")
    places, impl, fmt = rest.split("/")
    VARIANTS.append((name, frozenset(places.split(",")), impl, fmt))


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def timed(g, n):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / n


def kernels(g):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
    evs = sorted((e for e in prof.events() if e.device_type.name == "CUDA"), key=lambda e: e.time_range.start)
    per = len(evs) // 3
    return [[evs[k].name[:70], round(statistics.fmean(evs[i * per + k].time_range.elapsed_us() for i in range(3)), 1)] for k in range(per)]


for spec in specs:
    if spec in DONE:
        continue
    if time.time() > DEADLINE:
        print("DEADLINE", flush=True)
        sys.exit(0)
    p = list(map(int, spec.split(":")))
    S, N, dp = p[0], p[1], (p[2] if len(p) > 2 else 128)
    torch.manual_seed(0)
    m = MSAPairWeightedAveraging(64, dp, 8, 32, p_drop=P_DROP, master_weights=True, backend="cuda").to(dev).train()
    with torch.no_grad():
        m.to_out.weight.normal_(0, 0.1)
    x = torch.randn(1, S, N, 64, device=dev).bfloat16().requires_grad_()
    z = torch.randn(1, N, N, dp, device=dev).bfloat16().requires_grad_()
    gy = torch.randn(1, S, N, 64, device=dev).bfloat16()
    mask = torch.ones(1, N, dtype=torch.bool, device=dev)
    params = [q for q in m.parameters() if q.requires_grad]

    def step(m=m, x=x, z=z, mask=mask, params=params, gy=gy):
        y = m(x, z, mask)
        return torch.autograd.grad(y, [x, z, *params], gy)

    graphs = {}
    F.PLACES = frozenset()
    graphs["def"] = graph_of(step)
    graphs["ctl"] = graph_of(step)
    for name, places, impl, fmt in VARIANTS:
        F.NATIVE_ONLY = places == frozenset({"native"})  # bf16 through the native kernels (attribution)
        F.O8 = fmt.endswith("+o8")
        fmt = fmt.removesuffix("+o8")
        F.PLACES, F.IMPL, F.DO_FMT = (frozenset() if F.NATIVE_ONLY else places), impl, fmt
        F.reset()
        step()  # eager: seeds the scales
        step()
        graphs[name] = graph_of(step)
        F.PLACES, F.NATIVE_ONLY = frozenset(), False
    names = list(graphs)
    for k in names:
        timed(graphs[k], 3)
    t1 = {k: timed(g, 5) for k, g in graphs.items()}
    n_cool = max(1, min(200, int(10.0 / max(t1.values()))))
    n_sus = max(1, min(400, int(20.0 / max(t1.values()))))
    cool = {k: [] for k in names}
    for rep in range(3):
        time.sleep(2.0)
        order = names if rep % 2 == 0 else names[::-1]
        for k in order:
            cool[k].append(timed(graphs[k], n_cool))
    sus = {k: [] for k in names}
    t_end = time.time() + 1.0
    while time.time() < t_end:
        for k in names:
            timed(graphs[k], max(1, n_sus // 4))
    for rep in range(3):
        order = names if rep % 2 == 0 else names[::-1]
        for k in order:
            sus[k].append(timed(graphs[k], n_sus))
    rec = {
        "case": spec,
        "p_drop": P_DROP,
        "cool": {k: statistics.median(v) for k, v in cool.items()},
        "sus": {k: statistics.fmean(v) for k, v in sus.items()},
        "ovf": F.overflow_count(),
    }
    for k in names[1:]:
        rec[f"cool_{k}"] = round(rec["cool"][k] / rec["cool"]["def"], 4)
        rec[f"sus_{k}"] = round(rec["sus"][k] / rec["sus"]["def"], 4)
    if PROFILE:
        rec["kernels"] = {k: kernels(graphs[k]) for k in ("def", *[v[0] for v in VARIANTS])}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "kernels"}), flush=True)
    if PROFILE:
        for k, rows in rec["kernels"].items():
            print("  ", k, round(sum(r[1] for r in rows), 1), flush=True)
            for r in rows:
                print("     ", r, flush=True)
    del graphs, m, x, z, gy
    torch.cuda.empty_cache()
print("done", flush=True)

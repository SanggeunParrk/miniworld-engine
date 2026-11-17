"""OPM FP8 route vs the 2.5.12 default route, same process, CUDA-graph replay of the module (one call per graph), interleaved.
usage: speed.py OUTNAME [--routes f8b,f8bf] CASE...   CASE = cm:cz:L:S[:a]  (a = divide after the projection, AF3 code / Protenix order)
Routes: def (FP8 off), ctl (FP8 off, a second capture: the same-cubin control), f8b (bound scales), f8bf (bound + cuBLASLt fast accumulation),
f8c (calib scales, lazy guard). Every S gate removed. cool = 2 s idle, then one interleaved burst (~40 ms per route), 3 reps, median;
sustained = 2 s idle, 1 s of back-to-back interleaved replays, then 3 interleaved timed blocks (~60 ms per route), mean.
Appends one json line per case to res/OUTNAME.jsonl (resumable)."""

import json, os, statistics, sys, time
import torch

args = sys.argv[1:]
out_name = args.pop(0)
routes = ["f8b", "f8bf"]
if args and args[0] == "--routes":
    args.pop(0); routes = args.pop(0).split(",")
OUT = f"/home/psk6950/miniworld-engine-scratch/fp8x-opm/res/{out_name}.jsonl"
DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):
        DONE.add(json.loads(line)["case"])
DEADLINE = int(os.environ.get("DEADLINE", time.time() + 500)) - 40
dev = "cuda"
from miniworld_engine.kernels.integrations import opm_fp8 as F  # noqa: E402
from miniworld_engine.modules.msa import OuterProductMean  # noqa: E402

def init_params(m, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))

def setr(r):
    F.MIN_S = 0
    F.GUARD = "lazy"
    F.POLICY = {"def": None, "ctl": None, "f8b": "bound", "f8bf": "bound", "f8c": "calib"}[r]
    F.FAST_ACCUM = r == "f8bf"

def graph_of(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
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
    b.record(); b.synchronize()
    return a.elapsed_time(b) / n

for spec in args:
    if spec in DONE:
        continue
    if time.time() > DEADLINE:
        print("DEADLINE", flush=True); sys.exit(0)
    p = spec.split(":")
    cm, cz, L, S = map(int, p[:4]); after = len(p) > 4 and p[4] == "a"
    torch.manual_seed(0)
    m = OuterProductMean(cm, d_hidden=32, d_pair=cz, normalize_before_proj=not after, master_weights=True, backend="cuda").to(dev).eval()
    init_params(m, cm + cz)
    x = torch.randn(1, S, L, cm, device=dev).bfloat16()
    mask = torch.ones(1, S, L, dtype=torch.bool, device=dev)
    mask[:, S - S // 8:, : L // 2] = False   # ragged MSA rows (a part of the depth covers half the tokens)
    graphs = {}
    t0 = time.time()
    with torch.no_grad():
        for r in ["def", "ctl"] + routes:
            setr(r)
            if r.startswith("f8"):
                m(x, mask); m(x, mask); torch.cuda.synchronize()
            graphs[r] = graph_of(lambda: m(x, mask))
        peak = F.overflow_peak()
        setr("def")
    names = list(graphs)
    t1 = {k: timed(g, 2) for k, g in graphs.items()}
    n_cool = max(1, min(200, int(40.0 / max(t1.values()))))
    n_sus = max(1, min(400, int(60.0 / max(t1.values()))))
    cool = {k: [] for k in names}
    for rep in range(3):
        time.sleep(2.0)
        for k in (names if rep % 2 == 0 else names[::-1]):
            cool[k].append(timed(graphs[k], n_cool))
    sus = {k: [] for k in names}
    time.sleep(2.0)
    t_end = time.time() + 1.0
    while time.time() < t_end:
        for k in names:
            timed(graphs[k], max(1, n_sus // 4))
    for rep in range(3):
        for k in (names if rep % 2 == 0 else names[::-1]):
            sus[k].append(timed(graphs[k], n_sus))
    rec = {"case": spec, "cool": {k: statistics.median(v) for k, v in cool.items()}, "sus": {k: statistics.fmean(v) for k, v in sus.items()},
           "peak": peak, "setup_s": round(time.time() - t0, 1)}
    for k in names[1:]:
        rec[f"cool_{k}"] = round(rec["cool"][k] / rec["cool"]["def"], 4)
        rec[f"sus_{k}"] = round(rec["sus"][k] / rec["sus"]["def"], 4)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    del graphs, m, x
    torch.cuda.empty_cache()
print("done", flush=True)

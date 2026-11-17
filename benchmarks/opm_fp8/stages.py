"""Stage times of the OPM inference chain (bf16 default pieces and the FP8 pieces) on the same operands, CUDA graph per stage, interleaved;
cool (2 s idle, burst ~40 ms) median of 3 and sustained (1 s back-to-back of all stages, then 3 interleaved blocks) mean.
usage: stages.py OUTNAME CASE...   CASE = cm:cz:L:S"""
import json, os, statistics, sys, time
import torch
from miniworld_engine.kernels.integrations import opm_train as ot, opm_fp8 as F

out_name, specs = sys.argv[1], sys.argv[2:]
OUT = f"/home/psk6950/miniworld-engine-scratch/fp8x-opm/res/{out_name}.jsonl"
DONE = {json.loads(l)["case"] for l in open(OUT)} if os.path.exists(OUT) else set()
DEADLINE = int(os.environ.get("DEADLINE", time.time() + 500)) - 40
dev = torch.device("cuda"); bf = torch.bfloat16

def graph_of(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize(); return g

def timed(g, n):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        g.replay()
    b.record(); b.synchronize(); return a.elapsed_time(b) / n

for spec in specs:
    if spec in DONE:
        continue
    if time.time() > DEADLINE:
        print("DEADLINE", flush=True); sys.exit(0)
    cm, cz, L, S = map(int, spec.split(":"))
    torch.manual_seed(0)
    m = torch.randn(S, L, cm, device=dev, dtype=bf); mk = torch.rand(S, L, device=dev) > 0.05
    gam = 1 + 0.1 * torch.randn(cm, device=dev); bet = 0.1 * torch.randn(cm, device=dev)
    wa = (torch.randn(32, cm, device=dev) * cm ** -0.5).to(bf); wb = (torch.randn(32, cm, device=dev) * cm ** -0.5).to(bf)
    wo = (torch.randn(cz, 1024, device=dev) * 1024 ** -0.5).to(bf); ob = 0.1 * torch.randn(cz, device=dev)
    e16 = ot._ext(dev, native=True, tma=False, cm=cm, cz=cz)
    e8 = ot._ext(dev, native=True, tma=False, cm=cm, cz=cz, fp8=True)
    a2, bt, _, mask16, bits = e16.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False)
    s = F.bound_scales(gam, bet, wa, wb).to(dev); q = (1.0 / s).contiguous(); amax = torch.zeros(2, device=dev)
    sab = (s[0:1].clone(), s[1:2].clone())
    a8, b8, *_ = e8.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False, f8q=q, f8amax=amax)
    table = ot._count_table(mask16, 1.0, 0.0)
    rows = max(16, min(L, ot._O_CHUNK_BYTES // (L * 32 * 32 * 2) // 16 * 16)); nr = min(rows, L)
    O = torch.empty((nr * 32, L * 32), device=dev, dtype=bf)
    V = {
        "pro16": lambda: e16.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False),
        "pro8": lambda: e8.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False, f8q=q, f8amax=amax),
        "cnt": lambda: ot._count_table(mask16, 1.0, 0.0),
        "mm16": lambda: ot._product(a2[: nr * 32], bt, L, out=O),
    }
    def mm8(fast):
        def f():
            F.FAST_ACCUM = fast
            F.product(a8[: nr * 32], b8, sab, L, out=O)
        return f
    V["mm8"] = mm8(False); V["mm8f"] = mm8(True)
    V["epi"] = lambda: e16.opm_epilogue(O, table[:nr], wo, ob, nr, L, None, 0)
    if (cm, cz) == (64, 128) and S <= 4096:
        ef = ot._ext(dev, native=True, tma=False, cm=cm, cz=cz, fused=True)
        wp = ot._fused_weight(wo)
        V["fused"] = lambda: ef.opm_fused_infer(a2, bt, bits, wp, ob, None, 0)
    G = {k: graph_of(fn) for k, fn in V.items()}
    names = list(G)
    t1 = {k: max(timed(g, 2), 0.01) for k, g in G.items()}
    cool = {k: [] for k in names}
    for rep in range(3):
        for k in (names if rep % 2 == 0 else names[::-1]):
            time.sleep(1.0 if k != names[0] else 2.0)
            cool[k].append(timed(G[k], max(1, min(200, int(40 / t1[k])))))
    sus = {k: [] for k in names}
    t_end = time.time() + 1.0
    while time.time() < t_end:
        for k in names:
            timed(G[k], max(1, int(10 / t1[k])))
    for rep in range(3):
        for k in (names if rep % 2 == 0 else names[::-1]):
            sus[k].append(timed(G[k], max(1, min(400, int(60 / t1[k])))))
    chunks = -(-L // nr)
    rec = {"case": spec, "chunks": chunks, "rows": nr,
           "cool": {k: statistics.median(v) for k, v in cool.items()}, "sus": {k: statistics.fmean(v) for k, v in sus.items()},
           "flop": {"pro": 2 * S * L * cm * 64, "mm": 2 * S * (32 * L) ** 2, "epi": 2 * L * L * 1024 * cz},
           "bytes": {"pro16": S * L * (cm * 2 + 1 + 2) + 2 * S * L * 32 * 2, "pro8": S * L * (cm * 2 + 1 + 2) + 2 * S * L * 32,
                     "mm16_in": 2 * S * L * 32 * 2, "mm8_in": 2 * S * L * 32, "O": L * L * 1024 * 2, "z": L * L * cz * 2},
           "amax": amax.tolist(), "scale": s.tolist()}
    # per-chunk stages (mm*, epi) -> whole-call times
    for d in ("cool", "sus"):
        for k in ("mm16", "mm8", "mm8f", "epi"):
            rec[d][k + "_all"] = rec[d][k] * chunks if L % nr == 0 else None
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    del G, V, a2, bt, a8, b8, O, m
    torch.cuda.empty_cache()
print("done", flush=True)

"""Kernel A/B of opm_fused_infer variants on the same operands (prologue once), interleaved; cool (idle before a short burst) and sustained
(back-to-back for --sustain s, then timed) per variant per round; SM clock + power (NVML) after the sustained burst; bitwise compare to
cluster 1.  python kbench.py L S OUT.jsonl [--rounds 4] [--variants cl1,cl2,cublas]"""
import argparse, json, statistics, sys, time
import torch
from miniworld_engine.kernels.integrations import opm_train as ot

ap = argparse.ArgumentParser()
ap.add_argument("L", type=int); ap.add_argument("S", type=int); ap.add_argument("out")
ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--variants", default="cl1,cl2,cublas")
ap.add_argument("--sustain", type=float, default=0.6); ap.add_argument("--idle", type=float, default=2.0)
ap.add_argument("--residual", action="store_true")
a = ap.parse_args()
T0 = time.time(); L, S = a.L, a.S; dev = torch.device("cuda"); bf = torch.bfloat16
torch.manual_seed(0)
m = torch.randn(S, L, 64, device=dev, dtype=bf); mk = torch.rand(S, L, device=dev) > 0.1
gam = torch.ones(64, device=dev); bet = torch.zeros(64, device=dev)
wa = (torch.randn(32, 64, device=dev) * 0.1).to(bf); wb = (torch.randn(32, 64, device=dev) * 0.1).to(bf)
wo = (torch.randn(128, 1024, device=dev) * 0.02).to(bf); ob = torch.randn(128, device=dev) * 0.1
ext = ot._ext(dev, native=True, tma=False, cm=64, cz=128, fused=True)
print("ext ready", round(time.time() - T0, 1), flush=True)
a2, bt, stats, mask16, norm = ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False)
del m, stats
wp = ot._fused_weight(wo)
res = torch.randn(1, L, L, 128, device=dev, dtype=bf) if a.residual else None
V = {f"cl{c}": (lambda c=c: ext.opm_fused_infer(a2, bt, norm, wp, ob, res, 1, cluster=c)) for c in (1, 2, 3, 4)}   # 1 none, 2 2x1, 3 1x2, 4 2x2
V["cublas"] = lambda: torch.mm(a2, bt.t())
V["unfused"] = lambda: ext.opm_epilogue(torch.mm(a2, bt.t()), norm, wo, ob, L, L, res, 1)   # cuBLAS product + the CUDA epilogue kernel
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from builds import build   # noqa: E402
for k in a.variants.split(","):
    if "." in k:   # "<build>.cl<code>": a macro variant of the extension (builds.py)
        bname, cl = k.split(".")
        bext = build(bname)
        V[k] = (lambda bext=bext, c=int(cl[2:]): bext.opm_fused_infer(a2, bt, norm, wp, ob, res, 1, cluster=c))
V = {k: V[k] for k in a.variants.split(",")}
ref = V["cl1"]() if "cl1" in V else None
for k in V:
    if (k != "cublas") and ref is not None:
        o = V[k](); o2 = V[k]()
        print(k, "bitwise vs cl1", bool(torch.equal(o, ref)), "run-twice", bool(torch.equal(o, o2)), "max|d|", float((o.float() - ref.float()).abs().max()), flush=True)
print("max clusters per code", ext.opm_fused_max_clusters(), flush=True)
flush = torch.empty(256 << 20, dtype=torch.int8, device=dev)
fl = 2 * S * L * L * 1024

def timed(fn, k):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(k)]; en = [torch.cuda.Event(enable_timing=True) for _ in range(k)]
    for s_, e_ in zip(st, en):
        flush.zero_(); s_.record(); fn(); e_.record()
    en[-2 if k > 1 else -1].synchronize()
    try: ghz, w = torch.cuda.clock_rate() / 1000.0, torch.cuda.power_draw() / 1000.0
    except Exception: ghz = w = None
    torch.cuda.synchronize()
    return statistics.median(s_.elapsed_time(e_) for s_, e_ in zip(st, en)), ghz, w

est = {}
for k, fn in V.items():
    fn(); torch.cuda.synchronize(); est[k] = max(timed(fn, 2)[0], 0.05)
with open(a.out, "a") as f:
    for r in range(a.rounds):
        names = list(V); names = names[r % len(names):] + names[: r % len(names)]
        for k in names:
            torch.cuda.synchronize(); time.sleep(a.idle)
            c, cg, cw = timed(V[k], max(2, min(5, int(40 / est[k]))))
            n = max(2, int(a.sustain * 1000 / est[k]))
            for _ in range(n):
                flush.zero_(); V[k]()
            s, g, w = timed(V[k], max(3, min(10, int(100 / est[k]))))
            rec = {"L": L, "S": S, "variant": k, "round": r, "cool_ms": c, "sus_ms": s, "sus_ghz": g, "sus_w": w, "residual": a.residual}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{k:<7} r{r} cool {c:.3f} sus {s:.3f} ms  TF {fl / s / 1e9:.0f}  {g} GHz {w} W", flush=True)
print("done", round(time.time() - T0, 1), flush=True)

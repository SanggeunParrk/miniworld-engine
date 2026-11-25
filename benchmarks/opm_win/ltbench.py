"""cuBLASLt algorithm sweep for the OPM product at one (L, S): every heuristic algorithm (config attributes), its error vs torch.mm,
cool and sustained ms, the SM clock and the energy per call (NVML total-energy counter over the sustained burst).
   python ltbench.py L S OUT.jsonl [--n 16] [--rounds 2]"""
import argparse, json, statistics, time
from pathlib import Path
import torch
import pynvml
from foldjet_engine.kernels._nvcc import load_extension, ensure_cuda_home

ap = argparse.ArgumentParser()
ap.add_argument("L", type=int); ap.add_argument("S", type=int); ap.add_argument("out")
ap.add_argument("--n", type=int, default=16); ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--sustain", type=float, default=1.0)
a = ap.parse_args()
dev = torch.device("cuda"); L, S = a.L, a.S
ext = load_extension(name="opm_lt_sweep", sources=[str(Path(__file__).with_name("csrc") / "lt_sweep.cpp")], extra_cflags=["-O3"], extra_include_paths=[str(Path(ensure_cuda_home()) / "targets" / "x86_64-linux" / "include")],
                     extra_ldflags=["-lcublasLt"])
pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
torch.manual_seed(0)
a2 = (torch.randn(L * 32, S, device=dev) * 0.5).to(torch.bfloat16); bt = (torch.randn(L * 32, S, device=dev) * 0.5).to(torch.bfloat16)
ref = torch.mm(a2, bt.t())
algos = ext.lt_algos(a2, bt, False, a.n)
out = torch.empty_like(ref)
flush = torch.empty(256 << 20, dtype=torch.int8, device=dev)
names = ["config_id", "tile", "stages", "splitk", "swizzle", "cluster", "inner", "ws", "waves1000"]
V = {"torch.mm": lambda: torch.mm(a2, bt.t(), out=out)}
for i, row in enumerate(algos):
    V[f"lt{i}"] = (lambda i=i: ext.lt_run(a2, bt, out, i))
    print(f"lt{i}", dict(zip(names, row)), flush=True)
info = {f"lt{i}": dict(zip(names, row)) for i, row in enumerate(algos)}
good = {}
for k, fn in V.items():
    try:
        fn(); torch.cuda.synchronize()
        d = float((out.float() - ref.float()).abs().max() / ref.float().abs().max())
        good[k] = fn
        print(k, "rel max diff vs torch.mm", f"{d:.2e}", "bitwise", bool(torch.equal(out, ref)), flush=True)
    except Exception as e:
        print(k, "FAILED", str(e)[:200], flush=True)

def timed(fn, k):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(k)]; en = [torch.cuda.Event(enable_timing=True) for _ in range(k)]
    for s_, e_ in zip(st, en):
        flush.zero_(); s_.record(); fn(); e_.record()
    torch.cuda.synchronize()
    return statistics.median(s_.elapsed_time(e_) for s_, e_ in zip(st, en))

est = {k: timed(fn, 2) for k, fn in good.items()}
with open(a.out, "a") as f:
    for r in range(a.rounds):
        ks = list(good); ks = ks[r % len(ks):] + ks[: r % len(ks)]
        for k in ks:
            fn = good[k]
            torch.cuda.synchronize(); time.sleep(2.0)
            c = timed(fn, max(2, min(5, int(40 / est[k]))))
            n = max(4, int(a.sustain * 1000 / est[k]))
            for _ in range(n // 2):
                flush.zero_(); fn()
            torch.cuda.synchronize()
            e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h); t0 = time.time()
            for _ in range(n // 2):
                flush.zero_(); fn()
            torch.cuda.synchronize()
            e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h); t1 = time.time()
            s = timed(fn, max(3, min(10, int(100 / est[k]))))
            ghz = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM) / 1000
            rec = {"L": L, "S": S, "variant": k, "round": r, "cool_ms": c, "sus_ms": s, "J_per_call": (e1 - e0) / 1000 / (n // 2),
                   "W": (e1 - e0) / 1000 / (t1 - t0), "ghz": ghz, **info.get(k, {})}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{k:<9} r{r} cool {c:.3f} sus {s:.3f} ms  {rec['J_per_call']:.3f} J/call  {rec['W']:.0f} W  {ghz} GHz  {info.get(k, '')}", flush=True)
print("done", flush=True)

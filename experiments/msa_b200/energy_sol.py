"""Power-capped SoL on this B200 (1000 W limit: sustained kernels run at ~985 W, clocks drop to fit).
Everything is timed the same way -- a CUDA graph of back-to-back calls replayed for ~3 s after a settle -- with the NVML energy
counter read across the window.  The ceilings are measured in that same regime:
  HBM : a vectorised elementwise stream (torch.mul, read + write) over 1 GiB tensors
  TC  : cuBLAS on each module's own dense GEMM with the module's own operands (OPM: A2 @ BT^T; PWA: w @ v per head)
Floors = bench.sol_floor with those ceilings (per kernel max(bytes / HBM, FLOPs / TC), summed).
    GPU=0 python energy_sol.py --out esol.json"""
import argparse, ctypes, json, os, sys, time, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import make_inputs, make_module, module_fn, anthropic_fn, sol_floor, HBM, TC

nvml = ctypes.CDLL("libnvidia-ml.so.1"); nvml.nvmlInit_v2()
hdl = ctypes.c_void_p(); nvml.nvmlDeviceGetHandleByIndex_v2(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]), ctypes.byref(hdl))
def energy_j():
    e = ctypes.c_ulonglong(); nvml.nvmlDeviceGetTotalEnergyConsumption(hdl, ctypes.byref(e)); return e.value / 1000.0

def sustained(fn, secs=3.0, target_ms=float(os.environ.get("ESOL_GRAPH_MS", "20"))):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.time(); fn(); torch.cuda.synchronize(); one = max(time.time() - t0, 1e-5)
    reps = max(1, min(200, int(target_ms / 1e3 / one)))
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    t0 = time.time()
    while time.time() - t0 < 1.0: g.replay()          # settle the clocks / power state
    torch.cuda.synchronize()
    n = 0; e0 = energy_j(); t0 = time.time()
    while time.time() - t0 < secs:
        for _ in range(5): g.replay()
        n += 5 * reps
        torch.cuda.synchronize()
    dt = time.time() - t0; de = energy_j() - e0
    del g
    return {"ms": dt / n * 1e3, "J": de / n, "W": de / dt}

def ceilings(L, S):
    bf = torch.bfloat16
    x = torch.randn(512 * 1024 * 1024, device="cuda", dtype=bf); y = torch.empty_like(x)
    r = sustained(lambda: torch.mul(x, 1.0, out=y)); hbm = 2 * x.numel() * 2 / (r["ms"] * 1e-3)
    del x, y; torch.cuda.empty_cache()
    out = {"hbm_Bps": hbm, "hbm_W": r["W"]}
    with torch.no_grad():
        # OPM: the grouped outer product A2 [(i,c), s] @ BT^T, from the module's own prologue math
        mod = make_module("opm", "pytorch"); msa, pair, mask = make_inputs("opm", L, S)
        yl = mod.ln_msa(msa); m = mask[..., None].to(bf)
        a = (mod.to_left(yl) * m)[0]; b = (mod.to_right(yl) * m)[0]              # [S, N, 32]
        A2 = a.permute(1, 2, 0).reshape(L * 32, S).contiguous(); BT = b.permute(1, 2, 0).reshape(L * 32, S).contiguous()
        O = torch.empty(L * 32, L * 32, device="cuda", dtype=bf)
        r = sustained(lambda: torch.mm(A2, BT.t(), out=O)); out["tc_opm"] = 2 * (L * 32) ** 2 * S / (r["ms"] * 1e-3); out["tc_opm_W"] = r["W"]
        del A2, BT, O, a, b, yl, mod; torch.cuda.empty_cache()
        # PWA: the per-head contraction w [H, N, N] @ v [H, N, S*C], from the module's own pair softmax and value projection
        mod = make_module("pwa", "pytorch"); msa, pair, mask = make_inputs("pwa", L, S)
        lg = mod.to_bias(mod.ln_pair(pair))[0].permute(2, 0, 1).float()            # [H, N, N]
        lg = lg.masked_fill(~mask[0][None, None, :], -1e9)
        w = torch.softmax(lg, -1).to(bf).contiguous()
        v = mod.to_value(mod.ln_msa(msa))[0].view(S, L, 8, 32).permute(2, 1, 0, 3).reshape(8, L, S * 32).contiguous()
        o = torch.empty(8, L, S * 32, device="cuda", dtype=bf)
        r = sustained(lambda: torch.bmm(w, v, out=o)); out["tc_pwa"] = 2 * 8 * L * L * S * 32 / (r["ms"] * 1e-3); out["tc_pwa_W"] = r["W"]
    return out

def case(op, impl, mode, L, S):
    mod = make_module(op, impl); msa, pair, mask = make_inputs(op, L, S)
    if mode == "infer":
        mod.eval()
        with torch.no_grad():
            fn = anthropic_fn(op, mod, msa, pair, mask) if impl == "anthropic" else module_fn(op, mod, msa, pair, mask)
            return sustained(fn)
    if impl == "anthropic":
        return None
    mod.train(); msa.requires_grad_(True); pair.requires_grad_(True)
    f = module_fn(op, mod, msa, pair, mask); params = [msa, pair, *mod.parameters()]
    gout = torch.randn(msa.shape if op == "pwa" else pair.shape, device="cuda", dtype=torch.bfloat16)
    return sustained(lambda: torch.autograd.grad(f(), params, gout))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int, default=384); ap.add_argument("--S", type=int, default=1024); ap.add_argument("--out")
    ap.add_argument("--cal", help="reuse the ceilings of a previous --out json"); ap.add_argument("--only", nargs="*", default=[], help="op:mode:impl filters, e.g. opm:train:ours")
    a = ap.parse_args()
    cal = json.load(open(a.cal))["ceilings"] if a.cal else ceilings(a.L, a.S)
    cal["tc_pwa_cublas_bmm"] = cal.get("tc_pwa_cublas_bmm", cal["tc_pwa"])
    cal["tc_pwa"] = cal["tc_opm"]   # the PWA-shaped bmm is cuBLAS being slow on that shape, not the card's ceiling: use the dense rate
    print(json.dumps({k: (f"{v:.4g}") for k, v in cal.items()}), flush=True)
    rows = []
    for op in ("opm", "pwa"):
        tc = cal["tc_" + op]
        for mode in ("infer", "train"):
            floor_cap = sol_floor(op, mode, a.L, a.S, hbm=cal["hbm_Bps"], tc=tc) * 1e3
            floor_clk = sol_floor(op, mode, a.L, a.S) * 1e3
            en = (float(os.environ.get("ESOL_EB", "104e-12")), float(os.environ.get("ESOL_EF", "0.578e-12")), float(os.environ.get("ESOL_PDYN", "752")))
            floor_en = sol_floor(op, mode, a.L, a.S, hbm=cal["hbm_Bps"], tc=tc, energy=en) * 1e3
            for impl in ("pytorch", "anthropic", "ours"):
                if a.only and f"{op}:{mode}:{impl}" not in a.only:
                    continue
                try:
                    r = case(op, impl, mode, a.L, a.S)
                except Exception as exc:
                    r = None; print(op, mode, impl, "failed:", str(exc)[:200])
                torch.cuda.empty_cache()
                row = dict(op=op, mode=mode, impl=impl, floor_cap_ms=floor_cap, floor_clk_ms=floor_clk, floor_energy_ms=floor_en, **(r or {}))
                rows.append(row)
                if r:
                    sol = (f"  SoL(energy) {100 * floor_en / r['ms']:5.1f}% [{floor_en:.3f}]  SoL(power-capped) {100 * floor_cap / r['ms']:5.1f}%"
                       f"  SoL(clock-peak) {100 * floor_clk / r['ms']:5.1f}%") if impl == "ours" else ""
                    print(f"{op} {mode:5s} {impl:9s} {r['ms']:.4f} ms  {r['J']*1e3:.3f} mJ  {r['W']:.0f} W{sol}", flush=True)
                else:
                    print(f"{op} {mode:5s} {impl:9s} n/a", flush=True)
    if a.out:
        json.dump({"ceilings": cal, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""End level, FoldForge + miniworld backend, TriAttn FP8 on RoseTTAFold3, per TRUNK seed.
usage: e2e3.py OUT TARGET [TARGET...]
modes (E2E_MODES, comma list): bf16 = every FP8 route off; f8 = TriAttn PV only (enable_fp8_pv_for, kinds of the table / E2E_FORCE);
  f8qk = PV + e4m3 q / k on the fused route; safe = fp8.enable('safe', model=MODEL): every `safe` route + the TriAttn part the table lists for MODEL.
env: E2E_MODEL (rosettafold3)  E2E_TS (0,1,2,3)  E2E_FORCE (kinds)  E2E_DEADLINE (s, stop starting new cases; resumable: one .json per finished case)
Per case: coords / ipTM / pTM / pLDDT (.pt), the FP8 call counters (fp8.route_stats() when present), V^T peak statistics, q / k scale feasibility."""
import json, os, sys, time, traceback, collections
from pathlib import Path
import torch
SPECS = "/home/psk6950/foldforge-scratch/foldbench/specs_all"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
MODES = os.environ.get("E2E_MODES", "bf16,f8,f8qk,safe").split(","); TS = [int(x) for x in os.environ.get("E2E_TS", "0,1,2,3").split(",")]
MODEL = os.environ.get("E2E_MODEL", "rosettafold3")
DEADLINE = float(os.environ.get("E2E_DEADLINE", "440"))
T0 = time.time()
class Prox:
    """Records the V^T planes' absmax (fraction of the e4m3 range, 448) per (kind, L, H, D) of the calls that go through the PV core."""
    def __init__(self, real, st): self.r, self.st = real, st
    def __getattr__(self, n): return getattr(self.r, n)
    def triattn_fwd_h16(self, q, *a, **k):
        vt = k.get("vt")
        if vt is None and len(a) >= 11 and a[10] is not None and hasattr(a[10], "dtype"): vt = a[10]
        if vt is not None:
            p = vt.view(torch.float8_e4m3fn) if vt.dtype == torch.uint8 else vt
            kind = "fused" if "vt" in k else "pass"
            self.st[(kind, q.size(2), q.size(3), q.size(4))].append(p.float().abs().amax() / 448.0)
        return self.r.triattn_fwd_h16(q, *a, **k)
def main():
    import foldforge as ff
    import foldforge.models.checkpoints.lock as _LK
    _LK.check_size = lambda _p: None
    from miniworld_engine import fp8
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as HS
    print("engine at", HS.__file__, flush=True)
    st = collections.defaultdict(list)
    rf8 = HS._core_f8pv; HS._core_f8pv = lambda: Prox(rf8(), st)
    force = os.environ.get("E2E_FORCE")
    if force: HS.FP8_PV_WHITELIST[MODEL] = frozenset(force.split(","))
    cfg = {"backend": "miniworld", "precision": "bf16", "diffusion": {"steps": 200}, "execution": {"compile": False, "cuda_graph": False, "bucketing": True, "scope": "trunk"}}
    t0 = time.time()
    with ff.load_model(MODEL, config=cfg, device="cuda:0") as model:
        print("LOADED", round(time.time() - t0), flush=True)
        for t in sys.argv[2:]:
            inputs = None
            for m in MODES:
                for ts in TS:
                    name = f"{m}_t{ts}"; f = OUT / f"{t}.{name}.json"
                    if f.exists(): continue
                    if time.time() - T0 > DEADLINE: print("DEADLINE: stop (resumable)", flush=True); print("END", flush=True); return
                    tc = time.time()
                    try:
                        fp8.disable(); HS.disable_fp8_pv(); HS.FP8_QK = False
                        if m in ("f8", "f8qk"):
                            assert HS.enable_fp8_pv_for(MODEL), f"{MODEL} not whitelisted (set E2E_FORCE)"
                            HS.FP8_QK = m == "f8qk"
                        elif m == "safe":
                            fp8.enable("safe", model=MODEL)
                        elif m == "safenota":  # every `safe` route, TriAttn left on the default fp16 core (attribution)
                            fp8.enable("safe")
                        elif m != "bf16": raise ValueError(m)
                        if inputs is None: inputs = ff.load_inputs(Path(SPECS) / f"{t}.yaml")
                        st.clear(); redo = 0; ovf_events = 0
                        while True:
                            if hasattr(fp8, "reset_route_stats"): fp8.reset_route_stats()
                            with torch.no_grad(), model.run_trunk(inputs, seed=ts) as trunk:
                                res = model.sample_diffusion(trunk, samples=1, steps=200, seed=0)
                            if m == "bf16": break
                            hit = fp8.check() if m == "safe" else HS.take_overflow()
                            if not hit or redo >= 3: break
                            ovf_events += 1; redo += 1; st.clear()
                            if m != "safe": HS.rescale_all(4.0)
                            print(f"OVERFLOW {t} {name}: rescaled, redo {redo}", flush=True)
                        rs = fp8.route_stats() if hasattr(fp8, "route_stats") else {}
                        stats = {}
                        for key, v in st.items():
                            x = torch.stack(v).cpu(); y = x[x < 0.9995]
                            stats[f"{key[0]}:L{key[1]}H{key[2]}D{key[3]}"] = {"calls": len(v), "n_eq1": int((x >= 0.9995).sum()), "n_nan": int(x.isnan().sum()), "peak_frac_max": round(float(y.max()), 4) if y.numel() else None,
                                                                              "peak_frac_p99": round(float(y.quantile(0.99)), 4) if y.numel() else None}
                        vs = [torch.log2(s.float()).cpu() for s in HS._VS_AUTO.values()]
                        vsc = torch.stack(vs) if vs else torch.zeros(0)
                        qk_sites = {"calibrated": len(HS._QK_AUTO), "infeasible": sum(1 for v in HS._QK_AUTO.values() if v is None)}
                        p = res.prediction
                        torch.save({"coords": p.coords.float().cpu(), "plddt": None if p.plddt is None else p.plddt.float().cpu(),
                                    "ptm": None if p.ptm is None else p.ptm.float().cpu(), "iptm": None if p.iptm is None else p.iptm.float().cpu()}, OUT / f"{t}.{name}.pt")
                        info = {"target": t, "case": name, "mode": m, "trunk_seed": ts, "secs": round(time.time() - tc, 1), "redo": redo, "overflow_events": ovf_events,
                                "sites": len(HS._VS_AUTO), "log2scale_min": float(vsc.min()) if vsc.numel() else None, "log2scale_max": float(vsc.max()) if vsc.numel() else None,
                                "heads_le_-20": int((vsc <= -20).sum()) if vsc.numel() else 0, "qk_sites": qk_sites, "route_stats": rs, "peak": stats,
                                "f8_calls": sum(s["calls"] for s in stats.values())}
                        f.write_text(json.dumps(info))
                        print(f"DONE {t} {name} {time.time() - tc:.0f}s redo={redo} ta={json.dumps(rs.get('triattn'))} qk_sites={qk_sites} heads<=-20:{info['heads_le_-20']}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc(); print(f"FAIL {t} {name} {type(e).__name__}: {str(e)[:300]}", flush=True); torch.cuda.empty_cache()
        fp8.disable(); HS.disable_fp8_pv()
    print("END", round(time.time() - t0), flush=True)
main()

"""End level, FoldForge + miniworld backend, FP8-PV TriAttn on the SHIPPED route (enable_fp8_pv_for, S gates as shipped) vs bf16, per TRUNK seed.
usage: e2e2.py OUT TARGET [TARGET...]
env: E2E_MODEL (protenix1)   E2E_MODES (bf16,f8: f8 = enable_fp8_pv_for(model); f8g768 = + the 384:12:32 gate at 768)   E2E_TS (0,1,2,3)
     E2E_FORCE (comma kinds, e.g. fused,unfused: whitelist override for a model that is not on the whitelist; evaluation only)   E2E_NOGATE=1 (remove all S gates)
cases: <mode>_t<trunk seed>, diffusion_seed 0, 200 steps, 1 sample; resumable (a .json per finished case); per-case log lines incl. overflow + peak / scale stats."""
import json, os, sys, time, traceback, collections
from pathlib import Path
import torch
SPECS = "/home/psk6950/foldforge-scratch/foldbench/specs_all"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
MODES = os.environ.get("E2E_MODES", "bf16,f8").split(","); TS = [int(x) for x in os.environ.get("E2E_TS", "0,1,2,3").split(",")]
MODEL = os.environ.get("E2E_MODEL", "protenix1")
DEADLINE = float(os.environ.get("E2E_DEADLINE", "540"))  # stop starting new cases after this many seconds (job limit 600)
T0 = time.time()
class Prox:
    """Counts the FP8 core calls and records the V^T planes' absmax (fraction of the e4m3 range, 448) per (L, H, D)."""
    def __init__(self, real, st): self.r, self.st = real, st
    def __getattr__(self, n): return getattr(self.r, n)
    def triattn_fwd_h16(self, q, *a, **k):
        vt = k.get("vt")
        if vt is None and len(a) >= 11 and a[10] is not None and hasattr(a[10], "dtype"): vt = a[10]  # positional: (k, k, sbias, scale, want_lse, gate, want_out, key_tiles, key_mask, bias_t, vt, vs)
        if vt is not None:
            p = vt.view(torch.float8_e4m3fn) if vt.dtype == torch.uint8 else vt
            kind = "fused" if "vt" in k else "pass"  # fused: the d_pair-128 front wrote the planes (kwargs); pass: standalone pass / front-made planes (positional)
            self.st[(kind, q.size(2), q.size(3), q.size(4))].append(p.float().abs().amax() / 448.0)
        return self.r.triattn_fwd_h16(q, *a, **k)
def main():
    import foldforge as ff
    import foldforge.models.checkpoints.lock as _LK
    _LK.check_size = lambda _p: None
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
                        HS.disable_fp8_pv()
                        if m != "bf16":
                            assert HS.enable_fp8_pv_for(MODEL), f"{MODEL} not whitelisted (set E2E_FORCE)"
                            if os.environ.get("E2E_NOGATE"): HS.FP8_PV_MIN_S = {}; HS.FP8_PV_FRONT_MIN_S = {}
                            if m == "f8g768": HS.FP8_PV_FRONT_MIN_S[(384, 12, 32)] = 768; HS.FP8_PV_MIN_S[32] = 768
                        if inputs is None: inputs = ff.load_inputs(Path(SPECS) / f"{t}.yaml")
                        st.clear(); redo = 0; ovf_events = 0
                        while True:
                            with torch.no_grad(), model.run_trunk(inputs, seed=ts) as trunk:
                                res = model.sample_diffusion(trunk, samples=1, steps=200, seed=0)
                            if m == "bf16" or not HS.take_overflow() or redo >= 3: break
                            ovf_events += 1; HS.rescale_all(4.0); redo += 1; st.clear(); print(f"OVERFLOW {t} {name}: rescale x4, redo {redo}", flush=True)
                        # peak / scale statistics of the last (accepted) forward
                        stats = {}
                        for key, v in st.items():
                            x = torch.stack(v).cpu(); y = x[x < 0.9995]  # exactly 1.0 = a calibration call (dynamic scale, amax -> 448) or a saturation (the flag counts those)
                            stats[f"{key[0]}:L{key[1]}H{key[2]}D{key[3]}"] = {"calls": len(v), "n_eq1": int((x >= 0.9995).sum()), "n_nan": int(x.isnan().sum()), "peak_frac_max": round(float(y.max()), 4) if y.numel() else None,
                                                                              "peak_frac_p99": round(float(y.quantile(0.99)), 4) if y.numel() else None, "peak_frac_mean": round(float(y.mean()), 4) if y.numel() else None}
                        sc = [float(torch.log2(s.float()).min()) for s in HS._VS_AUTO.values()] + [float(torch.log2(s.float()).max()) for s in HS._VS_AUTO.values()]
                        sk = collections.defaultdict(list)
                        for k_, s_ in HS._VS_AUTO.items(): sk["fused" if not isinstance(k_, tuple) else f"vq_H{k_[2]}D{k_[3]}"].append(torch.log2(s_.float()).cpu())
                        scale_kinds = {k_: {"sites": len(v_), "log2_min": float(torch.stack(v_).min()), "log2_max": float(torch.stack(v_).max()), "heads_le_-20": int((torch.stack(v_) <= -20).sum())} for k_, v_ in sk.items()}
                        p = res.prediction
                        torch.save({"coords": p.coords.float().cpu(), "plddt": None if p.plddt is None else p.plddt.float().cpu(),
                                    "ptm": None if p.ptm is None else p.ptm.float().cpu(), "iptm": None if p.iptm is None else p.iptm.float().cpu()}, OUT / f"{t}.{name}.pt")
                        info = {"target": t, "case": name, "mode": m, "trunk_seed": ts, "secs": round(time.time() - tc, 1), "f8_calls": sum(s["calls"] for s in stats.values()), "redo": redo,
                                "overflow_events": ovf_events, "sites": len(HS._VS_AUTO), "log2scale_min": min(sc) if sc else None, "log2scale_max": max(sc) if sc else None, "scale_kinds": scale_kinds, "peak": stats}
                        f.write_text(json.dumps(info))
                        print(f"DONE {t} {name} {time.time() - tc:.0f}s f8_calls={info['f8_calls']} redo={redo} sites={info['sites']} peak={json.dumps(stats)}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc(); print(f"FAIL {t} {name} {type(e).__name__}: {str(e)[:300]}", flush=True); torch.cuda.empty_cache()
        HS.disable_fp8_pv()
    print("END", round(time.time() - t0), flush=True)
main()

"""Call-site probe: which TriAttn core calls a FoldForge model's trunk makes (path f8 / fp16, B, L, H, D) with the FP8 PV route on for ALL site kinds
(fused + unfused, shipped S gates).  usage: probe.py MODEL TARGET [TARGET...]; E2E_GATES=none removes every S gate."""
import os, sys, time, collections
from pathlib import Path
import torch
SPECS = "/home/psk6950/foldforge-scratch/foldbench/specs_all"
MODEL = sys.argv[1]
_IN = [False]
class Prox:
    def __init__(self, real, tag, hist): self.r, self.tag, self.h = real, tag, hist
    def __getattr__(self, n): return getattr(self.r, n)
    def triattn_fwd_h16(self, q, *a, **k):
        if _IN[0]: return self.r.triattn_fwd_h16(q, *a, **k)  # the f8 core is built through _core_h16: not a separate fp16 call
        self.h[(self.tag, q.size(0), q.size(2), q.size(3), q.size(4))] += 1
        _IN[0] = self.tag == "f8"
        try: return self.r.triattn_fwd_h16(q, *a, **k)
        finally: _IN[0] = False
def main():
    import foldforge as ff
    import foldforge.models.checkpoints.lock as _LK
    _LK.check_size = lambda _p: None
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as HS
    print("engine at", HS.__file__, flush=True)
    cfg = {"backend": "miniworld", "precision": "bf16", "execution": {"compile": False, "cuda_graph": False, "bucketing": True, "scope": "trunk"}}
    hist = collections.Counter(); fv = collections.Counter()
    r16, rf8, fva = HS._core_h16, HS._core_f8pv, HS.front_vt_args
    HS._core_h16 = lambda *a, **k: Prox(r16(*a, **k), "fp16", hist); HS._core_f8pv = lambda: Prox(rf8(), "f8", hist)
    def fv_wrap(device, length, heads, channels, batch=1, dpair=0):
        r = fva(device, length, heads, channels, batch, dpair)
        fv[(dpair, length, heads, channels, "none" if r is None else ("calib" if r == HS.CALIBRATE else "planes"))] += 1; return r
    HS.front_vt_args = fv_wrap
    HS.enable_fp8_pv_for("protenix1"); HS.FP8_PV_UNFUSED = True
    HS.FP8_PV_MIN_S = {} if os.environ.get("E2E_GATES") == "none" else dict(HS.FP8_PV_UNFUSED_MIN_S)
    HS.FP8_PV_FRONT_MIN_S = {} if os.environ.get("E2E_GATES") == "none" else dict(HS.FP8_PV_UNFUSED_FRONT_MIN_S)
    if os.environ.get("E2E_G768"): HS.FP8_PV_FRONT_MIN_S[(384, 12, 32)] = 768; HS.FP8_PV_MIN_S[32] = 768
    t0 = time.time()
    with ff.load_model(MODEL, config=cfg, device="cuda:0") as model:
        print("LOADED", round(time.time() - t0), flush=True)
        for t in sys.argv[2:]:
            hist.clear(); fv.clear()
            inputs = ff.load_inputs(Path(SPECS) / f"{t}.yaml")
            with torch.no_grad(), model.run_trunk(inputs, seed=0) as trunk:
                pass
            print(f"PROBE {MODEL} {t} ovf={HS.take_overflow()}", flush=True)
            for k, n in sorted(hist.items(), key=lambda x: (x[0][0], x[0][2], x[0][3])): print(f"  core path={k[0]} B={k[1]} L={k[2]} H={k[3]} D={k[4]} calls={n}", flush=True)
            for k, n in sorted(fv.items()): print(f"  front_vt dpair={k[0]} L={k[1]} H={k[2]} D={k[3]} -> {k[4]} calls={n}", flush=True)
    print("END", flush=True)
main()

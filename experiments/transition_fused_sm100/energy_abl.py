"""Energy attribution of the forward: sustained energy per call of ablation builds (inference, L384)."""
import sys, json, torch
sys.argv = sys.argv[:1]
from energy import measure
from common import make_inputs
from fwd_op import FusedFwd
x, wa, wb, ws, g, b = make_inputs(384)
M, D, H = 384 * 384, 128, 512
res = {}
for name in ("tfwd", "abl_nomath", "abl_noln", "abl_noswld", "abl_now", "abl_all"):
    f = FusedFwd(f"build/{name}.cubin"); f.set_weights(wa, wb, ws)
    run, *_ = f.bind(x, g, b, save=False)
    r = measure(run, 6 * M * D * H)
    res[name] = r
    print(f"{name:12s} {r['us_per_call']:6.1f} us  {r['W']:5.0f} W  {r['J_per_call']*1e3:6.2f} mJ  {r['pJ_per_flop']:.3f} pJ/FLOP", flush=True)

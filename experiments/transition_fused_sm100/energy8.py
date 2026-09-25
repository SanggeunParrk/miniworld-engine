"""Sustained energy / time per backward call: e4m3 tbwd8 builds vs the bf16 v9 backward (same inputs, L384)."""
import sys, torch
names = sys.argv[1:] or ["tbwd8"]
sys.argv = sys.argv[:1]
from energy import measure
from common import make_inputs, graph_time
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
from bwd8_op import FusedBwd8, FusedBwd8x, Quant8, scales_for
M, D, H = 384 * 384, 128, 512
x, wa, wb, ws, g, b = make_inputs(384)
dy = torch.randn_like(x) * 0.1
f = FusedFwd2(); f.set_weights(wa, wb, ws)
run_f, out, xn, rstd, c1 = f.bind(x, g, b, save=True); run_f(); torch.cuda.synchronize()
sc = scales_for(x, wa, wb, ws, g, b, dy)
xq = (xn.float() / sc[0]).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
st16 = FusedTrain(f).bind(x, g, b, dy); st16()
r = measure(st16.keep[1], 22 * M * D * H)
print(f"bf16 v9   : sustained {r['us_per_call']:6.1f} us {r['W']:4.0f} W {r['J_per_call']*1e3:6.1f} mJ/call", flush=True)
import os
for n, R in [(n, int(R)) for n in names for R in os.environ.get("REPLS", "9").split(",")]:
    qrun, wab_q, wst_q, ws_q = Quant8(f"build/{n}.cubin").bind(wa, wb, ws, sc); qrun()
    rb, _ = (FusedBwd8x if "8x" in n else FusedBwd8)(f"build/{n}.cubin", R).bind(dy, xq, x, rstd, c1, g, wab_q, wst_q, sc, wa, wb, ws); rb(); torch.cuda.synchronize()
    r = measure(rb, 22 * M * D * H)
    print(f"{n:10s} R{R}: sustained {r['us_per_call']:6.1f} us {r['W']:4.0f} W {r['J_per_call']*1e3:6.1f} mJ/call   graph20 cold {graph_time(rb):6.1f}", flush=True)

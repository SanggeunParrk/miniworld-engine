"""Read the TRACE build's clock64 stamps (CTA 0..3) and print per-stage cycle breakdowns for the forward."""
import argparse, ctypes
import numpy as np, torch
from common import make_inputs
from fwd_op import FusedFwd
from cuda.bindings import driver as cu
import drv

p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=384); p.add_argument("--cubin", default="build/tfwd_trace.cubin")
a = p.parse_args()
x, wa, wb, ws, gamma, beta = make_inputs(a.length)
f = FusedFwd(a.cubin); f.set_weights(wa, wb, ws)
run, *_ = f.bind(x, gamma, beta, save=True)
for _ in range(3):
    run()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(f.k.module, b"g_trace"), "global")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda")
drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "copy")
t = buf.cpu().numpy().reshape(4, 3, 512).astype(np.int64)
for cta in range(2):
    m, sw, ln = t[cta, 0], t[cta, 1], t[cta, 2]
    t0 = min(v for v in (m[0], sw[0], ln[0]) if v > 0)
    ntiles = int((ln[::8] > 0).sum())
    nch = ntiles * 8
    print(f"CTA {cta}: tiles {ntiles}, total {(ln[8*(ntiles-1)+6]-t0)} clk")
    sw_wait = [sw[4*c+1]-sw[4*c] for c in range(nch)]; sw_comp = [sw[4*c+2]-sw[4*c+1] for c in range(nch)]; sw_st = [sw[4*c+3]-sw[4*c+2] for c in range(nch)]
    print(f"  SwiGLU per chunk: wait ab {np.median(sw_wait):.0f}  ld+compute {np.median(sw_comp):.0f}  wait h_empty+st {np.median(sw_st):.0f}  (median clk)")
    print(f"  SwiGLU chunk period {np.median(np.diff([sw[4*c] for c in range(nch)])):.0f}")
    ex = [m[2*c] for c in range(nch)]; sq = [m[2*c+1] for c in range(nch)]
    print(f"  MMA expand period {np.median(np.diff(ex)):.0f}, squeeze period {np.median(np.diff(sq)):.0f}")
    for i in range(min(ntiles, 3)):
        e = ln[8*i:8*i+8] - t0
        print(f"  tile {i}: LN wait {e[1]-e[0]} LN {e[2]-e[1]} | epi wait {e[4]-e[3]} epi {e[5]-e[4]} store {e[6]-e[5]}  (start {e[0]})")

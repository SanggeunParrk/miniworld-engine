"""Per-role CTA exit times of a SPAN build of tbwd8 (after the bf16 forward), median over 20 runs."""
import os, sys, numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make_inputs
from fwd_op import FusedFwd2
from bwd8_op import FusedBwd8, FusedBwd8x, Quant8, scales_for
x, wa, wb, ws, g, b = make_inputs(384)
dy = torch.randn_like(x) * 0.1
f = FusedFwd2(); f.set_weights(wa, wb, ws)
run_f, out, xn, rstd, c1 = f.bind(x, g, b, save=True); run_f(); torch.cuda.synchronize()
sc = scales_for(x, wa, wb, ws, g, b, dy)
xq = (xn.float() / sc[0]).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
import os
REPLS = [int(v) for v in os.environ.get("REPLS", "9").split(",")]
for name, R in [(n, R) for n in sys.argv[1:] for R in REPLS]:
    qrun, wab_q, wst_q, ws_q = Quant8(f"build/{name}.cubin").bind(wa, wb, ws, sc); qrun()
    bb = (FusedBwd8x if "8x" in name else FusedBwd8)(f"build/{name}.cubin", R); rb, _ = bb.bind(dy, xq, x, rstd, c1, g, wab_q, wst_q, sc, wa, wb, ws)
    res = []
    for rep in range(20):
        run_f(); rb(); torch.cuda.synchronize()
        dptr, size = drv._chk(cu.cuModuleGetGlobal(bb.k.module, b"g_spanb"), "g")
        buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
        t = buf.cpu().numpy().reshape(256, 2)[:148].astype(np.int64)
        e = (t[:, 1] - t[:, 0].min()) / 1000
        nd = 8 * R
        res.append((np.median(e[:nd]), e[:nd].max(), np.median(e[nd:]), e[nd:].max()))
    r = np.median(np.array(res), axis=0)
    print(f"{name:12s} R{R}: DW exit median {r[0]:6.1f} max {r[1]:6.1f} | DX exit median {r[2]:6.1f} max {r[3]:6.1f} us", flush=True)

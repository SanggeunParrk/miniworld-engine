"""Per-role CTA exit times (globaltimer, us from the first CTA entry) of SPAN builds of tbwd.cu; the forward runs first as in a step.
  NAMES=bspan_x,bspan_y python span_sim.py"""
import os, numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make_inputs
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
x, wa, wb, ws, g, b = make_inputs(384)
f = FusedFwd2(os.environ.get("FWD", "build/tfwd2.cubin")); f.set_weights(wa, wb, ws)
for name in os.environ["NAMES"].split(","):
    tr = FusedTrain(f, repl=9, cubin=f"build/{name}.cubin"); tr.b.xch = "xch" in name; st = tr.bind(x, g, b, torch.randn_like(x) * 0.1)
    res = []
    for rep in range(20):
        st(); torch.cuda.synchronize()
        dptr, size = drv._chk(cu.cuModuleGetGlobal(tr.b.k.module, b"g_spanb"), "g")
        buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
        t = buf.cpu().numpy().reshape(256, 2)[:148].astype(np.int64)
        e = (t[:, 1] - t[:, 0].min()) / 1000
        res.append((np.median(e[:72]), e[:72].max(), np.median(e[72:]), e[72:].max()))
    r = np.median(np.array(res), axis=0)
    print(f"{name:12s}: DW exit median {r[0]:6.1f} max {r[1]:6.1f} | DX exit median {r[2]:6.1f} max {r[3]:6.1f} us", flush=True)

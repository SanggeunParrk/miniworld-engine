"""Per-role exit times of a SPAN build of tbwdx (bf16 exchange backward) after the bf16 forward; also energy."""
import os, sys, numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make_inputs, graph_time
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
x, wa, wb, ws, g, b = make_inputs(384)
dy = torch.randn_like(x) * 0.1
f = FusedFwd2(); f.set_weights(wa, wb, ws)
name = sys.argv[1]
for R in [int(v) for v in os.environ.get("REPLS", "14").split(",")]:
    tr = FusedTrain(f, repl=R, cubin=f"build/{name}.cubin", x=True); st = tr.bind(x, g, b, dy)
    res = []
    for rep in range(10):
        st(); torch.cuda.synchronize()
        dptr, size = drv._chk(cu.cuModuleGetGlobal(tr.b.k.module, b"g_spanb"), "g")
        buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
        t = buf.cpu().numpy().reshape(256, 2)[:148].astype(np.int64)
        e = (t[:, 1] - t[:, 0].min()) / 1000; nd = 8 * R
        res.append((np.median(e[:nd]), e[:nd].max(), np.median(e[nd:]), e[nd:].max()))
    r = np.median(np.array(res), axis=0)
    print(f"{name} R{R}: DW exit median {r[0]:6.1f} max {r[1]:6.1f} | DX exit median {r[2]:6.1f} max {r[3]:6.1f} us | bwd graph {graph_time(st.keep[1]):.1f}", flush=True)

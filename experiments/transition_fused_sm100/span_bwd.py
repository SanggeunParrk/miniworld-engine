import numpy as np, torch, drv, sys
from cuda.bindings import driver as cu
from common import make_inputs
from fwd_op import FusedFwd
from bwd_op import FusedTrain
R = int(sys.argv[1]) if len(sys.argv) > 1 else 9
V2 = len(sys.argv) > 2 and sys.argv[2] == "v2"
x, wa, wb, ws, g, b = make_inputs(384)
f = FusedFwd(); f.set_weights(wa, wb, ws)
tr = FusedTrain(f, repl=R, cubin="build/tbwd2_trace.cubin" if V2 else "build/tbwd_trace.cubin", v2=V2); st = tr.bind(x, g, b, torch.randn_like(x) * 0.1)
for _ in range(3): st()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(tr.b.k.module, b"g_spanb"), "g")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
t = buf.cpu().numpy().reshape(256, 2)[:148].astype(np.int64)
t0 = t[:, 0].min(); e = (t[:, 1] - t0) / 1000
ndw = 8 * R
print("DW exits (us): median %.1f max %.1f" % (np.median(e[:ndw]), e[:ndw].max()))
print("DX exits (us): median %.1f max %.1f" % (np.median(e[ndw:]), e[ndw:].max()))
print("DX exit per CTA:", " ".join("%.0f" % v for v in e[ndw:]))

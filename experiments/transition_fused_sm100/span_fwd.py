import numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make_inputs
from fwd_op import FusedFwd
import sys
L = int(sys.argv[1]) if len(sys.argv) > 1 else 384
x, wa, wb, ws, g, b = make_inputs(L)
f = FusedFwd("build/tfwd_trace.cubin"); f.set_weights(wa, wb, ws)
run, *_ = f.bind(x, g, b, save=False)
for _ in range(5): run()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(f.k.module, b"g_span"), "g")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
t = buf.cpu().numpy().reshape(256, 3)[:148].astype(np.int64)
t0 = t[:, 0].min()
print("entry spread (ns): %d .. %d" % (t[:,0].min()-t0, t[:,0].max()-t0))
print("setup (ns): median %d max %d" % (np.median(t[:,1]-t[:,0]), (t[:,1]-t[:,0]).max()))
e = t[:, 2] - t0
tiles = [(1152 - c + 147) // 148 for c in range(148)]
print("exit (ns): min %d median %d max %d" % (e.min(), np.median(e), e.max()))
print("exit by tile count:", {n: int(np.median([e[c] for c in range(148) if tiles[c] == n])) for n in set(tiles)})
print("slowest 5 CTAs:", np.argsort(e)[-5:].tolist(), e[np.argsort(e)[-5:]].tolist())
smid = None
print("exit per CTA (us):", " ".join("%.1f" % (v / 1000) for v in e.tolist()))
sm = (t[:, 1] & 0xFF)
order = np.argsort(sm)
print("exit by SM id (us):", " ".join("%d:%.1f" % (sm[c], e[c] / 1000) for c in order))

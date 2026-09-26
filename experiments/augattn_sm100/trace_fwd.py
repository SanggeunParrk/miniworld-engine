import sys, numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make
from ops import Fwd
L = int(sys.argv[2]); A = 48
q, k, v, bias = make(A, L)
f = Fwd(sys.argv[1]); run, O, LSE = f.bind(q, k, v, bias)
for _ in range(3): run()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(f.k.module, b"g_tr"), "g")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
t = buf.cpu().numpy().reshape(9, 64).astype(np.int64); t0 = t[7][0]; nb = L // 128
print("q_full seen at", t[8][0] - t0)
for n in range(nb):
    print(f"blk {n}: prod issued {t[0][n]-t0:7d}  mma kv seen {t[1][n]-t0 if n else 0:7d}  QK(w0) {t[2][n]-t0:7d}  sm s_full {t[3][n]-t0:7d}  sm P done {t[4][n]-t0:7d}  PV(w0) {t[5][n]-t0:7d}")
print("epilogue start", t[6][0] - t0)

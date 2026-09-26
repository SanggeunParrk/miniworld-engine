import sys, numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make
from ops import Fwd2
L = int(sys.argv[2]); A = 48
q, k, v, bias = make(A, L)
f = Fwd2(sys.argv[1]); run, O, LSE = f.bind(q, k, v, bias)
for _ in range(3): run()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(f.k.module, b"g_tr"), "g")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
t = buf.cpu().numpy().reshape(8, 256).astype(np.int64); t0 = t[1][0]
n = int((t[5] > 0).sum())
md = lambda x: float(np.median(x))
r = slice(4, n - 1)
print(f"blocks {n}; per block (sm0 P done diff) median {md(np.diff(t[5][:n])[4:]):.0f} clk")
print(f"  sm0: s_full->max {md((t[3]-t[2])[r]):.0f}  max->p_free seen {md((t[4]-t[3])[r]):.0f}  p_free->P done {md((t[5]-t[4])[r]):.0f}")
print(f"  P done(G) -> PV(G) issued {md((t[6]-t[5])[r]):.0f}   PV(G) issued -> s_full(G+2) seen {md((t[2][6:n] - t[6][4:n-2])):.0f}  QK(G) issued -> s_full(G) seen {md((t[2]-t[1])[r]):.0f}")
print(f"  prod issue(G) -> sm0 s_full(G) {md((t[2]-t[0])[r]):.0f}")
for G in range(8, 14):
    print(G, [int(t[e][G] - t0) for e in range(8)])

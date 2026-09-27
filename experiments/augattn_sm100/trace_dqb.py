import sys, numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make, H, D
from ops import Fwd2, Dqb
L = int(sys.argv[2]); A = 48
q, k, v, bias = make(A, L)
do = torch.randn_like(q, dtype=torch.float32).to(torch.bfloat16)
frun, O, LSE = Fwd2("build/attn_fwd2.cubin").bind(q, k, v, bias); frun()
Dd = torch.randn(A, H, L, device="cuda")
f = Dqb(sys.argv[1]); run, DQ, DB = f.bind(q, k, v, do, bias, LSE, Dd)
for _ in range(3): run()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(f.k.module, b"g_tr"), "g")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
t = buf.cpu().numpy().reshape(12, 256).astype(np.int64); t0 = t[1][0]
n = min(int((t[6] > 0).sum()), 256)
md = lambda x: float(np.median(x))
r = slice(4, n - 1)
print(f"steps {n}; per step (wg0 dS done diff) median {md(np.diff(t[5][:n])[4:]):.0f} clk")
print(f"  wg0: s_full->loaded {md((t[3]-t[2])[r]):.0f}  loaded->ds_free {md((t[4]-t[3])[r]):.0f}  ds_free->dS done {md((t[5]-t[4])[r]):.0f}  dS done->epi done {md((t[6]-t[5])[r]):.0f}")
print(f"  epi done(g) -> s_full(g+1) {md(t[2][5:n]-t[6][4:n-1]):.0f}   S(g,0) issued -> s_full seen {md((t[2]-t[1])[r]):.0f}   dS done(g)->dQ(g,1) issued {md((t[7]-t[5])[r]):.0f}")
print(f"  S(g,0) issued -> s_full landed {md((t[0]-t[1])[r]):.0f}   landed -> wg0 sees {md((t[2]-t[0])[r]):.0f}")
e = lambda a, b: md((t[b][:n - 3] - t[a][:n - 3])[4:])
print(f"  epi(g): enter->dq_full {e(5, 8) if False else 0:.0f}  dq_full->tmem done {e(8, 9):.0f}  ->bar1 {e(9, 10):.0f}  ->bar2 {e(10, 11):.0f}")
for G in range(8, 14):
    print(G, [int(t[k][G] - t0) for k in range(12)])

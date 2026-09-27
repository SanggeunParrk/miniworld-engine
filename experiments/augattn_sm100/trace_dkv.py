import sys, numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make, H, D
from ops import Fwd2, Dkv
L = int(sys.argv[2]); A = 48
q, k, v, bias = make(A, L)
do = torch.randn_like(q, dtype=torch.float32).to(torch.bfloat16)
frun, O, LSE = Fwd2("build/attn_fwd2.cubin").bind(q, k, v, bias); frun()
Dd = torch.randn(A, H, L, device="cuda") * 0.1
f = Dkv(sys.argv[1]); run, DK, DV = f.bind(q, k, v, do, bias.transpose(1, 2).contiguous(), LSE, Dd)
for _ in range(3): run()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(f.k.module, b"g_tr"), "g")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
t = buf.cpu().numpy().reshape(8, 256).astype(np.int64); t0 = t[1][0]
# events: 0 prod issue(G) 1 S(G) issued (w0 blocks) 2 dV/dK(G) issued (w0) 3 wg0 s_full seen 4 wg0 ds_full 5 wg0 loop top 6 wg0 full seen 7 wg0 epi done
G = np.arange(0, 120, 2)
md = lambda x: float(np.median(x))
print("wg0 per block (G even):")
print(f"  top->full seen {md(t[6][G]-t[5][G]):.0f}  full->s_full {md(t[3][G]-t[6][G]):.0f}  s_full->ds_full {md(t[4][G]-t[3][G]):.0f}  ds_full->next top {md(t[5][G+2]-t[4][G]):.0f}")
print(f"  block period {md(t[5][G+2]-t[5][G]):.0f}   S issued -> wg0 sees {md(t[3][G]-t[1][G]):.0f}   prod issue -> wg0 full seen {md(t[6][G]-t[0][G]):.0f}")
for g in range(10, 26, 2):
    print(g, [int(t[e][g] - t0) for e in range(8)])

import numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make_inputs
from fwd_op import FusedFwd
x, wa, wb, ws, g, b = make_inputs(384)
f = FusedFwd("build/tfwd2_trace.cubin", "transition_fwd2_sm100", 230912, cluster=2); f.set_weights(wa, wb, ws)
run, *_ = f.bind(x, g, b, save=False)
for _ in range(3): run()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(f.k.module, b"g_trace2"), "g")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
t = buf.cpu().numpy().reshape(2, 4, 1024).astype(np.int64)
md = lambda v: float(np.median(v))
ex, sq = t[0, 0], t[0, 1]
nch = 64
r = range(8, nch - 2)
print("leader expand: wait wab %.0f  wait ab_empty %.0f  period %.0f" % (md([ex[4*c+1]-ex[4*c] for c in r]), md([ex[4*c+2]-ex[4*c+1] for c in r]), md(np.diff([ex[4*c] for c in r]))))
print("leader squeeze: wait ws %.0f  wait h_full %.0f  period %.0f" % (md([sq[4*c+1]-sq[4*c] for c in r]), md([sq[4*c+2]-sq[4*c+1] for c in r]), md(np.diff([sq[4*c] for c in r]))))
for cta in range(2):
    for role, name in ((2, "even WG"), (3, "odd WG")):
        sw = t[cta, role]
        cs = [c for c in r if sw[4*c+3] > 0]
        print(f"CTA{cta} {name}: wait ex_done {md([sw[4*c+1]-sw[4*c] for c in cs]):.0f}  compute {md([sw[4*c+2]-sw[4*c+1] for c in cs]):.0f}  wait sq+st {md([sw[4*c+3]-sw[4*c+2] for c in cs]):.0f}")

"""Clock64 per-tile stage breakdown of DW CTA 0 in a TRACE build of tbwd8."""
import sys, numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make_inputs
from fwd_op import FusedFwd2
from bwd8_op import FusedBwd8, Quant8, scales_for
x, wa, wb, ws, g, b = make_inputs(384)
dy = torch.randn_like(x) * 0.1
f = FusedFwd2(); f.set_weights(wa, wb, ws)
run_f, out, xn, rstd, c1 = f.bind(x, g, b, save=True); run_f(); torch.cuda.synchronize()
sc = scales_for(x, wa, wb, ws, g, b, dy)
xq = (xn.float() / sc[0]).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
name = sys.argv[1]
qrun, wab_q, wst_q, ws_q = Quant8(f"build/{name}.cubin").bind(wa, wb, ws, sc); qrun()
bb = FusedBwd8(f"build/{name}.cubin"); rb, _ = bb.bind(dy, xq, x, rstd, c1, g, wab_q, wst_q, sc, wa, wb, ws)
for _ in range(3): run_f(); rb()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(bb.k.module, b"g_tr8"), "g")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
t = buf.cpu().numpy().reshape(12, 1024).astype(np.int64)
n = int((t[2] > 0).sum()); rr = slice(4, n - 2)
md = lambda v: float(np.median(v))
d = lambda a, b_: md((t[b_] - t[a])[rr])
print(f"DW CTA 0: {n} tiles, per tile {md(np.diff(t[2][:n])):.0f} clk (median), span {t[2][n-1]-t[2][0]} clk, mean {(t[2][n-1]-t[2][0])/(n-1):.0f}")
fw = t[10][:n] - t[9][:n]
print(f"  loader flag wait: median {md(fw):.0f}  mean {fw.mean():.0f}  max {fw.max()}  total {fw.sum()}  (tiles waiting > 1000 clk: {(fw > 1000).sum()})")
print(f"  w1: xn+gate_read seen -> dy_full seen {d(0,1):.0f}  -> issued {d(1,2):.0f}")
print(f"  gate: dhab seen -> g_empty seen {d(3,4):.0f}  -> loads done {d(4,5):.0f}  -> done {d(5,6):.0f}   period {md(np.diff(t[6][:n])):.0f}")
print(f"  wgrad: g_full seen -> issued {d(7,8):.0f}")
print(f"  chains: w1 issued(i) -> gate dhab seen(i) {d(2,3):.0f}   gate done(i) -> wgrad g_full seen(i) {d(6,7):.0f}")
x_ = t
print("  w1 xn seen(i) - w1 issued(i-1): %.0f" % md([x_[0][i] - x_[2][i - 1] for i in range(4, n - 2)]))
print("  w1 xn seen(i) - gate loads done(i-1): %.0f" % md([x_[0][i] - x_[5][i - 1] for i in range(4, n - 2)]))

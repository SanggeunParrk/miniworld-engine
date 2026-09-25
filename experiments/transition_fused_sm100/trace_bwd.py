"""Clock64 stage breakdown of the fused backward (TRACE build): DW CTA 0 and the first DX CTA."""
import argparse
import numpy as np, torch
from cuda.bindings import driver as cu
import drv
from common import make_inputs
from fwd_op import FusedFwd
from bwd_op import FusedTrain

p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=384); p.add_argument("--cubin", default="build/tbwd_trace.cubin")
p.add_argument("--repl", type=int, default=10)
a = p.parse_args()
x, wa, wb, ws, gamma, beta = make_inputs(a.length)
dy = torch.randn_like(x) * 0.1
f = FusedFwd(); f.set_weights(wa, wb, ws)
tr = FusedTrain(f, repl=a.repl, cubin=a.cubin)
step = tr.bind(x, gamma, beta, dy)
for _ in range(3):
    step()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(tr.b.k.module, b"g_trace"), "global")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda")
drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "copy")
t = buf.cpu().numpy().reshape(2, 4, 2048).astype(np.int64)
md = lambda v: float(np.median(v)) if len(v) else float("nan")
# ---- DW
w1, w2, g = t[0, 0], t[0, 1], t[0, 2]
n = int((w1[3::4] > 0).sum())
print(f"DW CTA 0: tiles {n}, span {w2[4*(n-1)+2]-w1[0]} clk, per tile {(w2[4*(n-1)+2]-w1[0])/n:.0f}")
print(f"  warp1 (dh+ab): wait in_full {md(w1[1::4][:n]-w1[0::4][:n]):.0f}  wait gate_read {md(w1[2::4][:n]-w1[1::4][:n]):.0f}  issue {md(w1[3::4][:n]-w1[2::4][:n]):.0f}  period {md(np.diff(w1[0::4][:n])):.0f}")
print(f"  warp2 (wgrad): wait g_full {md(w2[1::4][:n]-w2[0::4][:n]):.0f}  issue {md(w2[2::4][:n]-w2[1::4][:n]):.0f}")
pr = t[0, 3]
rr = range(4, n - 2)
print(f"  input chain: wgrad(i) issued -> in(i+2) load issued {md([pr[i+2] - w2[4*i+2] for i in rr]):.0f}  load issued -> dh+ab(i+2) may start {md([w1[4*(i+2)+1] - pr[i+2] for i in rr]):.0f}")
print(f"  per tile: dhab issue done -> gate sees dhab {md([g[4*i+1] - w1[4*i+3] for i in rr]):.0f}  gate done -> wgrad issued {md([w2[4*i+2] - g[4*i+3] for i in rr]):.0f}")
print(f"  gate: wait dhab {md(g[1::4][:n]-g[0::4][:n]):.0f}  wait g_empty {md(g[2::4][:n]-g[1::4][:n]):.0f}  compute+store {md(g[3::4][:n]-g[2::4][:n]):.0f}")
# ---- DX
m1, m2, g, e = t[1, 0], t[1, 1], t[1, 2], t[1, 3]
nch = int((m1[4::8] > 0).sum()); nt = nch // 8
print(f"DX CTA 0: tiles {nt}, chunks {nch}, span {e[8*(nt-1)+6]-m1[0]} clk, per chunk {(e[8*(nt-1)+6]-m1[0])/nch:.0f}")
r = slice(8, nch)
print(f"  warp1 (dh+ab): wait in {md((m1[1::8]-m1[0::8])[r]):.0f}  w_full {md((m1[2::8]-m1[1::8])[r]):.0f}  ab_free {md((m1[3::8]-m1[2::8])[r]):.0f}  issue {md((m1[4::8]-m1[3::8])[r]):.0f}  period {md(np.diff(m1[0::8][:nch])[8:]):.0f}")
print(f"  warp2 (dxn): wait g_full {md((m2[1::8]-m2[0::8])[r]):.0f}  dxn_empty {md((m2[2::8]-m2[1::8])[r]):.0f}  issue {md((m2[3::8]-m2[2::8])[r]):.0f}")
print(f"  gate: wait abdh {md((g[1::4]-g[0::4])[r]):.0f}  ld+compute {md((g[2::4]-g[1::4])[r]):.0f}  store+arrive {md((g[3::4]-g[2::4])[r]):.0f}  period {md(np.diff(g[0::4][:nch])[8:]):.0f}")
wp = e[1024:]
rr = range(8, nch - 3)
print(f"  WAB chain: dxn(c) issued -> WAB(c+2) load issued {md([wp[c+2] - m2[8*c+3] for c in rr]):.0f}  load issued -> ab(c+2) sees it {md([m1[8*(c+2)+2] - wp[c+2] for c in rr]):.0f}")
print(f"  gate(c) done -> dxn(c) issued {md([m2[8*c+3] - g[4*c+3] for c in rr]):.0f}   ab(c) issued -> gate sees abdh(c) {md([g[4*c+1] - m1[8*c+4] for c in rr]):.0f}")
gB = g[1024:]
print(f"  gate WG B done - WG A done {md([gB[c] - g[4*c+3] for c in rr]):.0f}   WG B done -> dxn issued {md([m2[8*c+3] - gB[c] for c in rr]):.0f}   dxn wait g_full end -> issued {md([m2[8*c+3] - m2[8*c+2] for c in rr]):.0f}")
print(f"  dxn: wait g_full end -> all issued {md([m2[8*c+3] - m2[8*c+2] for c in rr]):.0f}   all issued -> complete {md([m2[8*c+4] - m2[8*c+3] for c in rr]):.0f}")
print(f"  ab(c+1) issued (warp1 stamp) relative to dxn(c) issue start: {md([m1[8*(c+1)+4] - m2[8*c+2] for c in rr]):.0f}")
for i in range(min(nt, 3)):
    q = e[8*i:8*i+7] - m1[0]
    print(f"  epi tile {i}: wait dxn {q[1]-q[0]}  dn ld {q[2]-q[1]}  wait x {q[3]-q[2]}  pass1 {q[4]-q[3]}  pass2 {q[5]-q[4]}  store {q[6]-q[5]}")
base = m1[8 * 40]
print("gate A: loads done (ab_free arrive) - gate start:", md([g[1536 + c] - g[4*c+1] for c in rr]))
print("raw DX timeline (clk rel. to chunk 40 start): c | ab wait-end | ab/dh issued | gateA start | gateA done | gateB done | dxn start | dxn issued | dxn done")
for c in range(40, 46):
    print(c, 'ldA', g[1536+c]-base, m1[8*c+3]-base, m1[8*c+4]-base, g[4*c+1]-base, g[4*c+3]-base, g[1024+c]-base, m2[8*c+2]-base, m2[8*c+3]-base, m2[8*c+4]-base)
gs = np.array([g[4*c+1] for c in range(nch)])
d = np.diff(gs)
print("gate-start gaps within tiles (median):", md([d[c] for c in range(8, nch-1) if (c + 1) % 8 != 0]), " at tile boundaries (median):", md([d[c] for c in range(8, nch-1) if (c + 1) % 8 == 0]))
print("boundary gaps:", [int(d[c]) for c in range(8, nch-1) if (c + 1) % 8 == 0][:8])

"""Clock64 timeline of the first DX CTA in a TRACE build of tbwd8x."""
import sys, numpy as np, torch, drv
from cuda.bindings import driver as cu
from common import make_inputs
from fwd8_op import Train8
R = int(sys.argv[2]); x, wa, wb, ws, g, b = make_inputs(384)
dy = torch.randn_like(x) * 0.1
tr = Train8(R, bcubin=sys.argv[1]); st = tr.bind(x, wa, wb, ws, g, b, dy)
for _ in range(3): st()
torch.cuda.synchronize()
dptr, size = drv._chk(cu.cuModuleGetGlobal(tr.b.k.module, b"g_trx"), "g")
buf = torch.empty(size // 8, dtype=torch.int64, device="cuda"); drv._chk(cu.cuMemcpyDtoD(buf.data_ptr(), dptr, size), "c")
t = buf.cpu().numpy().reshape(9, 1024).astype(np.int64)
nch = int((t[3] > 0).sum()); nt = nch // 8
t0 = t[2][0]
md = lambda v: float(np.median(v))
fw = t[1][:nch] - t[0][:nch]
print(f"DX CTA 0: {nt} tiles, span {t[6][nt-1] - t0} clk, per tile {(t[6][nt-1] - t0) / nt:.0f}")
print(f"  dab flag wait per chunk: median {md(fw):.0f} mean {fw.mean():.0f} total {fw.sum()} ({fw.sum() / (t[6][nt-1] - t0) * 100:.0f}% of span)")
mw = t[8][:nch] - t[2][:nch]
print(f"  mma wait for dab_full: median {md(mw):.0f} mean {mw.mean():.0f}; issue {md(t[3][:nch] - t[8][:nch]):.0f}")
for i in range(min(nt, 6)):
    print(f"  tile {i}: first flag seen {t[1][8*i] - t0}, last flag seen {t[1][8*i+7] - t0}, last mma {t[3][8*i+7] - t0}, epi dxn seen {t[4][i] - t0}, pre-store {t[5][i] - t0}, done {t[6][i] - t0}, conv done {t[7][i] - t0}")
print("  epi durations (dxn seen -> done):", [int(t[6][i] - t[4][i]) for i in range(min(nt, 12))])
print("  last tiles:")
for i in range(max(0, nt - 4), nt):
    print(f"  tile {i}: first flag wait start {t[0][8*i] - t0}, first flag seen {t[1][8*i] - t0}, last flag seen {t[1][8*i+7] - t0}, last mma {t[3][8*i+7] - t0}, epi dxn seen {t[4][i] - t0}, done {t[6][i] - t0}")
lag = [t[4][i] - t[3][8*i+7] for i in range(nt)]
print("  mma done -> epi sees dxn (backlog):", [int(v) for v in lag[-8:]])

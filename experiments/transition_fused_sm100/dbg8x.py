import sys, time, ctypes, torch, drv
from cuda.bindings import driver as cu
from common import make_inputs
from fwd8_op import Train8
L = int(sys.argv[1]); R = int(sys.argv[2]); cub = sys.argv[3] if len(sys.argv) > 3 else "build/tbwd8x.cubin"
x, wa, wb, ws, g, b = make_inputs(L)
dy = torch.randn_like(x) * 0.1
tr = Train8(R, bcubin=cub)
st = tr.bind(x, wa, wb, ws, g, b, dy)
run_q, run_f, run_b = st.keep
prog = torch.zeros(148 * 16, dtype=torch.int32).pin_memory()
try:
    dptr, size = drv._chk(cu.cuModuleGetGlobal(tr.b.k.module, b"g_dbg"), "g")
    hp = ctypes.c_uint64(prog.data_ptr())
    drv._chk(cu.cuMemcpyHtoD(dptr, ctypes.addressof(hp), 8), "set")
except Exception as e:
    print("no dbg", e)
run_q(); run_f(); torch.cuda.synchronize(); print("fwd ok", flush=True)
run_b()
t0 = time.time(); ev = torch.cuda.Event(); ev.record()
while not ev.query():
    if time.time() - t0 > 8:
        print("HANG"); break
    time.sleep(0.1)
else:
    print("bwd ok", flush=True)
p = prog.view(148, 16).numpy()
ndw = 8 * R
print("DW rows [loader-flagwait, w1, gate, wgrad, publisher] (value = last index + 1):")
for c in range(0, ndw, max(1, ndw // 12)): print(" cta", c, p[c, :5].tolist())
print("DX rows [in, wab, dab-flagwait, mma chunk, epi tile, convert, dab got]:")
for c in range(ndw, 148, 4): print(" cta", c, p[c, :8].tolist(), "epi arrive", p[c, 8:12].tolist(), "epi prestore", p[c, 12:16].tolist())
